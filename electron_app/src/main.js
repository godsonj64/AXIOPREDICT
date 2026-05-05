const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");

const PORT = 5001; // 5000 is reserved by macOS AirPlay on macOS 12+
let mainWindow = null;
let pythonProcess = null;

function getResourcePath(...parts) {
  if (app.isPackaged) return path.join(process.resourcesPath, ...parts);
  return path.join(__dirname, "..", "..", ...parts);
}

// ── Find compatible Python ────────────────────────────────────────────────────
function findPython() {
  const home    = process.env.HOME || process.env.USERPROFILE || "";
  const appRoot = getResourcePath();
  const venv    = process.platform === "win32"
    ? path.join(appRoot, ".venv", "Scripts", "python.exe")
    : path.join(appRoot, ".venv", "bin", "python");

  return [
    venv,
    // EL-03: include Python 3.12 and 3.11 as well as 3.10–3.8
    "python3.12", "python3.11", "python3.10", "python3.9", "python3.8",
    "/opt/homebrew/bin/python3.12",
    "/opt/homebrew/bin/python3.11",
    "/opt/homebrew/bin/python3.10",
    "/opt/homebrew/bin/python3.9",
    "/opt/homebrew/bin/python3.8",
    "/usr/local/bin/python3.12",
    "/usr/local/bin/python3.11",
    "/usr/local/bin/python3.10",
    "/usr/local/bin/python3.9",
    "/usr/local/bin/python3.8",
    "/usr/bin/python3.12",
    "/usr/bin/python3.11",
    "/usr/bin/python3.10",
    "/usr/bin/python3.9",
    `${home}/.pyenv/shims/python3.12`,
    `${home}/.pyenv/shims/python3.11`,
    `${home}/.pyenv/shims/python3.10`,
    `${home}/.pyenv/shims/python3.9`,
    `${home}/miniconda3/bin/python3.12`,
    `${home}/miniconda3/bin/python3.11`,
    `${home}/miniconda3/bin/python3.10`,
    `${home}/miniconda3/bin/python3.9`,
    "python3", "python",
  ];
}

// ── Start Python backend ──────────────────────────────────────────────────────
function startPythonBackend() {
  return new Promise((resolve, reject) => {
    const serverScript = getResourcePath("python_backend", "server.py");
    const candidates   = findPython();
    let resolved       = false;

    const tryNext = (idx) => {
      if (idx >= candidates.length) {
        reject(new Error("Python 3.8–3.10 not found.\n\nRun: brew install python@3.10\nThen: bash setup.sh"));
        return;
      }
      const pyCmd = candidates[idx];
      if (idx === 0 && !fs.existsSync(pyCmd)) { tryNext(1); return; }

      const env = { ...process.env };
      env.PYTORCH_ENABLE_MPS_FALLBACK = "1";
      env.PYTHONPATH = getResourcePath("sybil-source") +
        (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : "");
      env.SYBIL_PORT = String(PORT);

      const proc = spawn(pyCmd, [serverScript], {
        env, stdio: ["ignore", "pipe", "pipe"], cwd: getResourcePath(),
      });

      proc.on("error", () => tryNext(idx + 1));

      proc.stdout.on("data", (d) => {
        const txt = d.toString();
        console.log("[Python]", txt.trim());
        if (!resolved && txt.includes("Starting Sybil backend")) {
          resolved = true;
          setTimeout(() => resolve(proc), 600);
        }
      });

      proc.stderr.on("data", (d) => {
        const txt = d.toString().trim();
        if (txt && !txt.includes("WARNING") && !txt.includes(" * Running"))
          console.error("[Python ERR]", txt);
      });

      proc.on("exit", (code) => {
        if (!resolved && code !== null && code !== 0) tryNext(idx + 1);
      });

      setTimeout(() => {
        if (!resolved && !proc.killed) { resolved = true; resolve(proc); }
      }, 15000);  // EL-01: 15s fallback; waitForBackend() is the true readiness gate
    };

    tryNext(0);
  });
}

function waitForBackend(retries = 30, interval = 600) {
  return new Promise((resolve, reject) => {
    const attempt = (n) => {
      if (n <= 0) { reject(new Error("Backend failed to respond")); return; }
      const req = http.get(
        { hostname: "127.0.0.1", port: PORT, path: "/health", timeout: 5000 },
        (res) => {
          // Drain the response so the socket closes properly
          res.resume();
          if (res.statusCode === 200) resolve();
          else setTimeout(() => attempt(n - 1), interval);
        }
      );
      // Kill a stalled socket so the retry timer fires promptly
      req.on("timeout", () => req.destroy());
      req.on("error", () => setTimeout(() => attempt(n - 1), interval));
    };
    attempt(retries);
  });
}

// ── Core HTTP helper ──────────────────────────────────────────────────────────
// deadlineMs: hard wall-clock deadline (0 = no deadline).
// socketIdleMs: kill if socket goes IDLE for this long (0 = no idle check).
// For model load: no deadline, no idle timeout (weights can be large & slow).
// For short calls: 60s deadline.
// For polling: 10s deadline.
function httpCall(method, endpoint, body, deadlineMs, socketIdleMs) {
  return new Promise((resolve, reject) => {
    const bodyStr  = body ? JSON.stringify(body) : "";
    let   settled  = false;
    let   deadline = null;

    const done = (val, isErr) => {
      if (settled) return;
      settled = true;
      if (deadline) clearTimeout(deadline);
      if (isErr) reject(val);
      else resolve(val);
    };

    // Hard wall-clock deadline (independent of socket activity)
    if (deadlineMs > 0) {
      deadline = setTimeout(() => {
        req.destroy();
        done(new Error(`Request to ${endpoint} timed out after ${deadlineMs / 1000}s`), true);
      }, deadlineMs);
    }

    const req = http.request({
      hostname: "127.0.0.1",
      port:     PORT,
      path:     endpoint,
      method:   method || "GET",
      headers: {
        "Content-Type":   "application/json",
        "Content-Length": Buffer.byteLength(bodyStr),
        "Connection":     "keep-alive",
      },
    }, (res) => {
      let data = "";
      res.on("data",  (c) => (data += c));
      res.on("end",   ()  => {
        try   { done({ status: res.statusCode, data: JSON.parse(data) }); }
        catch { done({ status: res.statusCode, data }); }
      });
      res.on("error", (e) => done(e, true));
    });

    req.on("error", (e) => done(e, true));

    // Optional socket-idle timeout (only for short calls where network stall = problem)
    if (socketIdleMs > 0) {
      req.setTimeout(socketIdleMs, () => {
        req.destroy();
        done(new Error(`Socket idle timeout for ${endpoint}`), true);
      });
    }

    if (bodyStr) req.write(bodyStr);
    req.end();
  });
}

// ── Window ────────────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400, height: 900, minWidth: 1000, minHeight: 700,
    backgroundColor: "#FAFAFA",
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
  });
  mainWindow.loadFile(path.join(__dirname, "index.html"));
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => { mainWindow = null; });
}

// ── IPC handlers ─────────────────────────────────────────────────────────────
ipcMain.handle("select-directory", async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"], title: "Select DICOM / Image Directory",
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle("select-checkpoint-dir", async () => {
  const r = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory"], title: "Select Checkpoint Directory",
  });
  return r.canceled ? null : r.filePaths[0];
});

ipcMain.handle("open-folder", async (_, p) => {
  if (p && fs.existsSync(p)) shell.openPath(p);
});

// Short calls: 60s wall-clock, 30s socket-idle
ipcMain.handle("api-call", async (_, { method, endpoint, body }) => {
  return httpCall(method, endpoint, body, 60_000, 30_000);
});

// Model load: very long deadline (15min) + 2min socket-idle — catches genuine hangs (EL-04)
ipcMain.handle("api-call-long", async (_, { method, endpoint, body }) => {
  return httpCall(method, endpoint, body, 900_000, 120_000);
});

// Poll for inference job status: 10s wall-clock
ipcMain.handle("api-poll", async (_, { endpoint }) => {
  return httpCall("GET", endpoint, null, 10_000, 5_000);
});

ipcMain.handle("get-backend-status", async () => {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${PORT}/health`, (res) => {
      let d = "";
      res.on("data", (c) => (d += c));
      res.on("end", () => {
        try   { resolve({ running: true,  info: JSON.parse(d) }); }
        catch { resolve({ running: true,  info: {} }); }
      });
    });
    req.on("error", () => resolve({ running: false }));
    req.setTimeout(5000, () => { req.destroy(); resolve({ running: false }); });
  });
});

// ── Lifecycle ─────────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  createWindow();
  try {
    pythonProcess = await startPythonBackend();
    await waitForBackend();
    console.log("Backend ready on port", PORT);
    mainWindow?.webContents.send("backend-ready");
  } catch (err) {
    console.error("Backend startup failed:", err.message);
    mainWindow?.webContents.send("backend-error", err.message);
  }
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (pythonProcess) { killPython(pythonProcess); pythonProcess = null; }
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => { if (pythonProcess) killPython(pythonProcess); });

// EL-02: platform-safe process termination
function killPython(proc) {
  try {
    if (process.platform === "win32") {
      require("child_process").execSync(`taskkill /pid ${proc.pid} /T /F`, { stdio: "ignore" });
    } else {
      proc.kill("SIGTERM");
    }
  } catch (_) { /* already exited */ }
}
