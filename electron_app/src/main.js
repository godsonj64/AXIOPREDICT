const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");
const fs = require("fs");

const PORT = 5001; // 5000 is reserved by macOS AirPlay on macOS 12+
let mainWindow = null;
let pythonProcess = null;

// ── Crash-resilient logging ──────────────────────────────────────────────────
// Write startup events + uncaught errors to %APPDATA%\AXIO PREDICT\logs\
// so we can diagnose silent first-launch failures even when no window appears.
const LOG_PATH = (() => {
  try {
    const dir = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(dir, { recursive: true });
    return path.join(dir, "startup.log");
  } catch (_) { return null; }
})();

function logLine(tag, msg) {
  const line = `[${new Date().toISOString()}] [${tag}] ${msg}\n`;
  try { console.log(line.trim()); } catch (_) {}
  if (LOG_PATH) { try { fs.appendFileSync(LOG_PATH, line); } catch (_) {} }
}

process.on("uncaughtException", (err) => {
  logLine("FATAL", `uncaughtException: ${err && err.stack || err}`);
});
process.on("unhandledRejection", (reason) => {
  logLine("FATAL", `unhandledRejection: ${reason && reason.stack || reason}`);
});

function getResourcePath(...parts) {
  if (app.isPackaged) return path.join(process.resourcesPath, ...parts);
  return path.join(__dirname, "..", "..", ...parts);
}

// ── Find compatible Python ────────────────────────────────────────────────────
// Returns an array of { cmd, args } candidates. The Windows `py` launcher
// resolves to whatever Python the user installed, regardless of the .X version,
// so we try it before falling back to bare `python.exe` and the usual install
// locations.
function findPython() {
  const home     = process.env.HOME || process.env.USERPROFILE || "";
  const localApp = process.env.LOCALAPPDATA || path.join(home, "AppData", "Local");
  const appRoot  = getResourcePath();
  const isWin    = process.platform === "win32";

  const exeName  = isWin ? "python.exe" : "python";
  const venvBin  = isWin ? "Scripts"    : "bin";
  const sourceVenv = path.join(appRoot, ".venv", venvBin, exeName);
  const userVenv   = isWin
    ? path.join(localApp, "AXIO_PREDICT", ".venv", "Scripts", "python.exe")
    : path.join(home, ".axio_predict", ".venv", "bin", "python");

  const C = (cmd, ...args) => ({ cmd, args });
  const list = [
    C(userVenv),       // per-user venv (preferred — has all deps)
    C(sourceVenv),     // dev / source-tree venv
  ];

  if (isWin) {
    list.push(
      C("py", "-3.12"), C("py", "-3.11"), C("py", "-3.10"),
      C("py", "-3.9"),  C("py", "-3"),    C("py"),
      C("python.exe"), C("python"), C("python3"),
    );
    for (const ver of ["312","311","310","39","38"]) {
      list.push(
        C(path.join(localApp, "Programs", "Python", `Python${ver}`, "python.exe")),
        C(path.join(home, "AppData", "Local", "Programs", "Python", `Python${ver}`, "python.exe")),
        C(`C:\\Python${ver}\\python.exe`),
        C(`C:\\Program Files\\Python${ver}\\python.exe`),
        C(`C:\\Program Files (x86)\\Python${ver}\\python.exe`),
      );
    }
  } else {
    list.push(
      C("python3.12"), C("python3.11"), C("python3.10"), C("python3.9"), C("python3.8"),
      C("/opt/homebrew/bin/python3.12"),
      C("/opt/homebrew/bin/python3.11"),
      C("/opt/homebrew/bin/python3.10"),
      C("/opt/homebrew/bin/python3.9"),
      C("/opt/homebrew/bin/python3.8"),
      C("/usr/local/bin/python3.12"),
      C("/usr/local/bin/python3.11"),
      C("/usr/local/bin/python3.10"),
      C("/usr/local/bin/python3.9"),
      C("/usr/local/bin/python3.8"),
      C("/usr/bin/python3.12"),
      C("/usr/bin/python3.11"),
      C("/usr/bin/python3.10"),
      C("/usr/bin/python3.9"),
      C(`${home}/.pyenv/shims/python3.12`),
      C(`${home}/.pyenv/shims/python3.11`),
      C(`${home}/.pyenv/shims/python3.10`),
      C(`${home}/.pyenv/shims/python3.9`),
      C(`${home}/miniconda3/bin/python3.12`),
      C(`${home}/miniconda3/bin/python3.11`),
      C(`${home}/miniconda3/bin/python3.10`),
      C(`${home}/miniconda3/bin/python3.9`),
      C("python3"), C("python"),
    );
  }
  return list;
}

// ── Start Python backend ──────────────────────────────────────────────────────
function startPythonBackend() {
  return new Promise((resolve, reject) => {
    const serverScript = getResourcePath("python_backend", "server.py");
    const candidates   = findPython();
    let resolved       = false;
    let lastStderr     = "";   // last meaningful stderr from a Python that *did* execute
    let lastTried      = "";   // command label for the last attempt

    const tryNext = (idx) => {
      if (idx >= candidates.length) {
        const isWin  = process.platform === "win32";
        const setupCmd = isWin ? "setup.bat" : "bash setup.sh";
        const installHint = isWin
          ? "Install Python 3.10+ from https://python.org (check \"Add to PATH\")."
          : "Install Python 3.10+ (e.g. brew install python@3.10).";
        const detail = lastStderr ? `\n\nLast error from ${lastTried}:\n${lastStderr}` : "";
        reject(new Error(
          `Could not start the Python backend.\n\n${installHint}\nThen run ${setupCmd} to install dependencies.${detail}`
        ));
        return;
      }
      const { cmd: pyCmd, args: pyArgs = [] } = candidates[idx];
      // For absolute paths, skip the entry if the file doesn't exist (saves a spawn).
      if (path.isAbsolute(pyCmd) && !fs.existsSync(pyCmd)) { tryNext(idx + 1); return; }

      lastTried = [pyCmd, ...pyArgs].join(" ");

      const env = { ...process.env };
      env.PYTORCH_ENABLE_MPS_FALLBACK = "1";

      // Prefer the per-user writable copy of sybil-source if the in-app setup
      // created one. Otherwise fall back to the bundled (possibly read-only)
      // copy under resources/.
      const home2     = process.env.HOME || process.env.USERPROFILE || "";
      const localApp2 = process.env.LOCALAPPDATA || path.join(home2, "AppData", "Local");
      const userCopy  = process.platform === "win32"
        ? path.join(localApp2, "AXIO_PREDICT", "sybil-source")
        : path.join(home2, ".axio_predict", "sybil-source");
      const sybilForPath = fs.existsSync(userCopy) ? userCopy : getResourcePath("sybil-source");
      env.PYTHONPATH = sybilForPath +
        (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : "");
      env.SYBIL_PORT = String(PORT);

      let proc;
      try {
        proc = spawn(pyCmd, [...pyArgs, serverScript], {
          env, stdio: ["ignore", "pipe", "pipe"], cwd: getResourcePath(),
          // shell:false on Windows so we can detect ENOENT for missing commands
          shell: false,
        });
      } catch (e) {
        tryNext(idx + 1);
        return;
      }

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
        if (!txt) return;
        if (!txt.includes("WARNING") && !txt.includes(" * Running")) {
          console.error("[Python ERR]", txt);
          // Keep the last 1.5KB of stderr — likely contains the import error
          lastStderr = (lastStderr + "\n" + txt).slice(-1500);
        }
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
// Title-bar overlay palette. Native Windows requires a solid (opaque) color for
// the overlay strip — fully-transparent values render as black on Win10 and
// have caused first-launch crashes in past Electron versions. We use the same
// surface color the page background uses, so the strip blends invisibly.
const TITLE_BAR_OVERLAY = {
  light: { color: "#FAFAFA", symbolColor: "#1F1F1F", height: 40 },
  dark:  { color: "#13131A", symbolColor: "#E3E2E6", height: 40 },
};

function createWindow() {
  const isMac = process.platform === "darwin";
  const isWin = process.platform === "win32";

  const opts = {
    width: 1400, height: 900, minWidth: 1000, minHeight: 700,
    backgroundColor: "#FAFAFA",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
  };

  if (isMac) {
    opts.titleBarStyle      = "hiddenInset";
    opts.trafficLightPosition = { x: 14, y: 13 };
  } else if (isWin) {
    opts.titleBarStyle   = "hidden";
    opts.titleBarOverlay = TITLE_BAR_OVERLAY.light;
  } else {
    opts.frame = false;
  }

  logLine("WINDOW", `creating (platform=${process.platform}, packaged=${app.isPackaged})`);
  mainWindow = new BrowserWindow(opts);

  const indexPath = path.join(__dirname, "index.html");
  mainWindow.loadFile(indexPath).catch((e) => {
    logLine("LOAD-ERR", `loadFile(${indexPath}) failed: ${e && e.message}`);
  });

  // Force-show the window if ready-to-show never fires (3s grace period).
  // Better to show a partially-rendered window than leave the user staring
  // at a missing app icon with no feedback.
  let shown = false;
  const showOnce = (why) => {
    if (shown || !mainWindow) return;
    shown = true;
    logLine("WINDOW", `showing (${why})`);
    try { mainWindow.show(); } catch (e) { logLine("WINDOW", `show() threw: ${e.message}`); }
  };
  mainWindow.once("ready-to-show", () => showOnce("ready-to-show"));
  setTimeout(() => showOnce("fallback-3s-timeout"), 3000);

  // Surface renderer failures into the log so silent blank windows are debuggable
  mainWindow.webContents.on("did-fail-load", (_e, code, desc, url) => {
    logLine("DID-FAIL-LOAD", `code=${code} desc=${desc} url=${url}`);
  });
  mainWindow.webContents.on("render-process-gone", (_e, details) => {
    logLine("RENDERER-GONE", `reason=${details.reason} exitCode=${details.exitCode}`);
    showOnce("renderer-gone");
  });
  mainWindow.webContents.on("preload-error", (_e, file, err) => {
    logLine("PRELOAD-ERR", `${file}: ${err && err.stack || err}`);
  });
  mainWindow.webContents.on("console-message", (_e, level, message, line, sourceId) => {
    if (level >= 2) logLine("RENDERER", `[${level}] ${sourceId}:${line} — ${message}`);
  });

  mainWindow.on("unresponsive", () => logLine("WINDOW", "unresponsive"));
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

// ── In-app environment setup ─────────────────────────────────────────────────
// Builds a per-user Python venv at the same location findPython() searches
// first, so the next backend startup just works. Streams stdout/stderr to the
// renderer so the user sees pip download progress in real time.
function spawnStream(cmd, args, env, onLine) {
  return new Promise((resolve, reject) => {
    let proc;
    try {
      proc = spawn(cmd, args, { env, stdio: ["ignore", "pipe", "pipe"], shell: false });
    } catch (e) { reject(e); return; }

    const forward = (chunk) => {
      const str = chunk.toString();
      // Forward as-is — pip uses \r for progress bars, renderer handles them
      onLine(str);
    };
    proc.stdout.on("data", forward);
    proc.stderr.on("data", forward);
    proc.on("error", reject);
    proc.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${path.basename(cmd)} ${args.join(" ")} exited with code ${code}`));
    });
  });
}

// Sybil's setup.cfg pins `python_requires = >=3.8,<3.11`, so a usable
// interpreter must be Python 3.8, 3.9, or 3.10. We probe explicitly for those
// minor versions and verify with `--version` before returning, instead of
// trusting whatever `py` or `python` happens to default to (the user may have
// 3.14 installed, which would only fail later during pip install).
const SUPPORTED_MINORS = ["3.10", "3.9", "3.8"]; // newest first
function isSupportedPyVer(verString) {
  // verString like "Python 3.10.11" — accept if minor is in [8, 10] and major is 3
  const m = /Python\s+(\d+)\.(\d+)\.(\d+)/.exec(verString || "");
  if (!m) return false;
  const major = +m[1], minor = +m[2];
  return major === 3 && minor >= 8 && minor <= 10;
}

function probePythonVersion(cmd, args) {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let proc;
    try {
      proc = spawn(cmd, [...args, "--version"], { stdio: ["ignore", "pipe", "pipe"], shell: false });
    } catch (_) { resolve(null); return; }
    proc.stdout.on("data", (d) => stdout += d.toString());
    proc.stderr.on("data", (d) => stderr += d.toString());
    proc.on("error", () => resolve(null));
    proc.on("exit", (code) => {
      if (code !== 0) { resolve(null); return; }
      // Older Pythons print version to stderr; newer to stdout. Try both.
      const v = (stdout + stderr).trim();
      resolve(v || null);
    });
  });
}

async function findSystemPython() {
  const home     = process.env.HOME || process.env.USERPROFILE || "";
  const localApp = process.env.LOCALAPPDATA || path.join(home, "AppData", "Local");
  const progFiles = process.env.PROGRAMFILES || "C:\\Program Files";
  const isWin    = process.platform === "win32";

  // Order: prefer the newest supported minor (3.10 > 3.9 > 3.8). For each,
  // try the Windows `py` launcher form first, then absolute install paths.
  const candidates = [];
  if (isWin) {
    for (const minor of SUPPORTED_MINORS) {
      const verNoDot = minor.replace(".", "");  // "310"
      candidates.push({ cmd: "py", args: [`-${minor}`] });
      candidates.push({ cmd: path.join(localApp, "Programs", "Python", `Python${verNoDot}`, "python.exe"), args: [] });
      candidates.push({ cmd: path.join(home, "AppData", "Local", "Programs", "Python", `Python${verNoDot}`, "python.exe"), args: [] });
      candidates.push({ cmd: path.join(progFiles, `Python${verNoDot}`, "python.exe"), args: [] });
      candidates.push({ cmd: `C:\\Python${verNoDot}\\python.exe`, args: [] });
    }
    // Last resorts that may or may not match — only kept if their version checks out
    candidates.push({ cmd: "python.exe", args: [] });
    candidates.push({ cmd: "python",     args: [] });
    candidates.push({ cmd: "python3",    args: [] });
  } else {
    for (const minor of SUPPORTED_MINORS) {
      candidates.push({ cmd: `python${minor}`, args: [] });
      candidates.push({ cmd: `/usr/local/bin/python${minor}`, args: [] });
      candidates.push({ cmd: `/usr/bin/python${minor}`, args: [] });
      candidates.push({ cmd: `/opt/homebrew/bin/python${minor}`, args: [] });
      candidates.push({ cmd: `${home}/.pyenv/shims/python${minor}`, args: [] });
    }
    candidates.push({ cmd: "python3", args: [] });
    candidates.push({ cmd: "python",  args: [] });
  }

  for (const cand of candidates) {
    if (path.isAbsolute(cand.cmd) && !fs.existsSync(cand.cmd)) continue;
    const ver = await probePythonVersion(cand.cmd, cand.args);
    if (ver && isSupportedPyVer(ver)) {
      return { ...cand, version: ver };
    }
  }
  return null;
}

let setupRunning = false;

// Download `url` to `destPath`, following one level of HTTP redirects.
// Calls onProgress(pct, downloadedBytes, totalBytes) for each 5% milestone.
function downloadFile(url, destPath, onProgress) {
  const https = require("https");
  return new Promise((resolve, reject) => {
    const fetch = (currentUrl, hopsLeft) => {
      if (hopsLeft < 0) { reject(new Error("Too many HTTP redirects")); return; }
      const req = https.get(currentUrl, (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode)) {
          res.resume();
          fetch(res.headers.location, hopsLeft - 1);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} from ${currentUrl}`));
          res.resume();
          return;
        }
        const total = parseInt(res.headers["content-length"] || "0", 10);
        let dl = 0;
        let lastPct = -1;
        const file = fs.createWriteStream(destPath);
        res.on("data", (chunk) => {
          dl += chunk.length;
          if (total > 0 && onProgress) {
            const pct = Math.floor((dl / total) * 100);
            if (pct !== lastPct && pct % 5 === 0) {
              lastPct = pct;
              onProgress(pct, dl, total);
            }
          }
        });
        res.pipe(file);
        file.on("finish", () => file.close((err) => err ? reject(err) : resolve()));
        file.on("error", (e) => { fs.unlink(destPath, () => {}); reject(e); });
      });
      req.on("error", (e) => { fs.unlink(destPath, () => {}); reject(e); });
      req.setTimeout(60_000, () => { req.destroy(new Error("Download stalled (60s)")); });
    };
    fetch(url, 5);
  });
}

// Auto-install Python 3.10 (Windows only). Per-user install — no admin/UAC
// required. The installer lands at %LOCALAPPDATA%\Programs\Python\Python310\,
// which findSystemPython() already searches.
const PYTHON_INSTALLER_URL = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe";
async function autoInstallPython(onLine) {
  if (process.platform !== "win32") {
    throw new Error("Automatic Python install is only supported on Windows.");
  }
  const home     = process.env.HOME || process.env.USERPROFILE || "";
  const localApp = process.env.LOCALAPPDATA || path.join(home, "AppData", "Local");
  const cacheDir = path.join(localApp, "AXIO_PREDICT");
  const installerPath = path.join(cacheDir, "python-3.10.11-amd64.exe");
  const targetDir = path.join(localApp, "Programs", "Python", "Python310");
  const targetPy  = path.join(targetDir, "python.exe");

  fs.mkdirSync(cacheDir, { recursive: true });

  if (fs.existsSync(targetPy)) {
    onLine(`✓ Python 3.10 already installed at ${targetDir}\n\n`);
    return { cmd: targetPy, args: [], version: "Python 3.10" };
  }

  onLine(`▶ Downloading Python 3.10.11 from python.org (~28 MB)...\n`);
  try {
    await downloadFile(PYTHON_INSTALLER_URL, installerPath, (pct, dl, total) => {
      onLine(`   ${pct}% (${(dl / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB)\n`);
    });
  } catch (e) {
    throw new Error(
      `Could not download Python 3.10.11: ${e.message}\n` +
      `Install manually from https://python.org (any 3.8–3.10 release) and retry.`
    );
  }
  onLine(`✓ Downloaded\n\n▶ Installing Python 3.10.11 (per-user, no admin prompt)...\n`);

  // Silent per-user install. Quiet mode + InstallAllUsers=0 avoids UAC.
  await spawnStream(
    installerPath,
    [
      "/quiet",
      "InstallAllUsers=0",
      "PrependPath=0",
      "Include_launcher=0",
      "Include_test=0",
      "Include_doc=0",
      "Include_dev=0",
      "Include_tcltk=0",
      "SimpleInstall=1",
      `TargetDir=${targetDir}`,
    ],
    process.env,
    onLine,
  );

  // The /quiet installer occasionally returns before the file is in place.
  // Poll briefly for the python.exe to appear.
  for (let i = 0; i < 20 && !fs.existsSync(targetPy); i++) {
    await new Promise((r) => setTimeout(r, 250));
  }
  if (!fs.existsSync(targetPy)) {
    throw new Error(
      `Python 3.10 installer finished but ${targetPy} is missing.\n` +
      `Antivirus or group policy may have blocked it. Install manually from https://python.org.`
    );
  }
  onLine(`✓ Python 3.10.11 installed at ${targetDir}\n\n`);

  try { fs.unlinkSync(installerPath); } catch (_) { /* leave the installer for debugging */ }
  return { cmd: targetPy, args: [], version: "Python 3.10.11" };
}

async function runEnvironmentSetup(onLine) {
  if (setupRunning) throw new Error("Setup already running.");
  setupRunning = true;
  try {
    const home     = process.env.HOME || process.env.USERPROFILE || "";
    const localApp = process.env.LOCALAPPDATA || path.join(home, "AppData", "Local");
    const isWin    = process.platform === "win32";
    const venvDir  = isWin
      ? path.join(localApp, "AXIO_PREDICT", ".venv")
      : path.join(home, ".axio_predict", ".venv");
    const venvPy   = isWin
      ? path.join(venvDir, "Scripts", "python.exe")
      : path.join(venvDir, "bin", "python");

    // Validate any existing venv before reusing it. A prior run (or the old
    // setup.bat using `py -3`) may have created the venv with an unsupported
    // Python like 3.14, which would silently break the Sybil install later.
    let needCreate = !fs.existsSync(venvPy);
    if (!needCreate) {
      onLine(`▶ Checking existing venv at ${venvDir}\n`);
      const ver = await probePythonVersion(venvPy, []);
      if (isSupportedPyVer(ver)) {
        onLine(`✓ Reusing venv (${ver})\n\n`);
      } else {
        onLine(`!  Existing venv has incompatible Python (${ver || "unknown"}); recreating...\n`);
        try { fs.rmSync(venvDir, { recursive: true, force: true }); } catch (_) {}
        needCreate = true;
      }
    }

    if (needCreate) {
      onLine(`▶ Looking for a system Python 3.8–3.10...\n`);
      let sys = await findSystemPython();
      if (!sys) {
        if (isWin) {
          onLine(`!  No compatible Python found. Installing Python 3.10.11 automatically.\n\n`);
          sys = await autoInstallPython(onLine);
        } else {
          throw new Error(
            "No Python 3.8–3.10 interpreter was found.\n" +
            "Install one (e.g. brew install python@3.10) and try again."
          );
        }
      } else {
        onLine(`✓ Using ${[sys.cmd, ...sys.args].join(" ")}  (${sys.version})\n\n`);
      }

      onLine(`▶ Creating virtual environment at ${venvDir}\n`);
      fs.mkdirSync(path.dirname(venvDir), { recursive: true });
      await spawnStream(sys.cmd, [...sys.args, "-m", "venv", venvDir], process.env, onLine);
      onLine(`✓ venv created\n\n`);
    }

    onLine(`▶ Upgrading pip\n`);
    await spawnStream(venvPy, ["-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"], process.env, onLine);
    onLine(`\n▶ Installing Flask, flask-cors, waitress\n`);
    await spawnStream(venvPy, ["-m", "pip", "install", "flask", "flask-cors", "waitress", "--disable-pip-version-check"], process.env, onLine);

    // The bundled sybil-source lives inside the install folder, which is
    // read-only when the user installed under "Program Files". `pip install
    // -e` writes egg-info into the source dir and fails there with EACCES.
    // Copy to a per-user writable location and install from the copy.
    const bundledSybilSrc = getResourcePath("sybil-source");
    const userSybilSrc    = isWin
      ? path.join(localApp, "AXIO_PREDICT", "sybil-source")
      : path.join(home, ".axio_predict", "sybil-source");
    onLine(`\n▶ Copying sybil-source to ${userSybilSrc}\n`);
    try {
      // Best-effort wipe so a stale copy doesn't poison the install
      if (fs.existsSync(userSybilSrc)) fs.rmSync(userSybilSrc, { recursive: true, force: true });
      fs.cpSync(bundledSybilSrc, userSybilSrc, {
        recursive: true,
        force: true,
        // Skip pre-existing build artifacts so we copy a clean tree
        filter: (src) => {
          const base = path.basename(src);
          if (base === "__pycache__" || base === ".git") return false;
          if (base.endsWith(".pyc")) return false;
          if (base.endsWith(".egg-info")) return false;
          return true;
        },
      });
    } catch (e) {
      throw new Error(`Failed to copy sybil-source: ${e.message}`);
    }

    onLine(`▶ Installing Sybil from ${userSybilSrc}\n  (this downloads PyTorch — can take 5–15 min)\n`);
    await spawnStream(venvPy, ["-m", "pip", "install", "-e", userSybilSrc, "--disable-pip-version-check"], process.env, onLine);

    onLine(`\n✓ Setup complete. Restarting backend...\n`);
    return venvPy;
  } finally {
    setupRunning = false;
  }
}

ipcMain.handle("app:run-setup", async (event) => {
  const send = (line) => { try { event.sender.send("setup:progress", line); } catch (_) {} };
  try {
    await runEnvironmentSetup(send);

    // Tear down a stale process if any, then start backend with the fresh venv
    if (pythonProcess) { try { killPython(pythonProcess); } catch (_) {} pythonProcess = null; }
    try {
      pythonProcess = await startPythonBackend();
      await waitForBackend();
      mainWindow?.webContents.send("backend-ready");
      send(`✓ Backend ready on port ${PORT}\n`);
      logLine("APP", "backend ready after in-app setup");
    } catch (err) {
      const msg = err && err.message || String(err);
      send(`\n✗ Setup finished but backend still failed: ${msg}\n`);
      mainWindow?.webContents.send("backend-error", msg);
      return { success: false, error: msg };
    }
    return { success: true };
  } catch (err) {
    const msg = err && err.message || String(err);
    send(`\n✗ Setup failed: ${msg}\n`);
    logLine("SETUP-ERR", msg);
    return { success: false, error: msg };
  }
});

// ── Window controls (custom title bar) ──────────────────────────────────────
ipcMain.handle("window:minimize", () => { mainWindow?.minimize(); });
ipcMain.handle("window:maximize", () => {
  if (!mainWindow) return false;
  if (mainWindow.isMaximized()) mainWindow.unmaximize();
  else mainWindow.maximize();
  return mainWindow.isMaximized();
});
ipcMain.handle("window:close", () => { mainWindow?.close(); });
ipcMain.handle("window:is-maximized", () => !!mainWindow?.isMaximized());

// Renderer reports the active theme so we can recolor the native overlay buttons
ipcMain.handle("window:set-theme", (_evt, theme) => {
  if (process.platform !== "win32" || !mainWindow) return;
  const palette = theme === "dark" ? TITLE_BAR_OVERLAY.dark : TITLE_BAR_OVERLAY.light;
  try { mainWindow.setTitleBarOverlay(palette); } catch (_) { /* older electron */ }
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
  logLine("APP", `whenReady (version=${app.getVersion()}, electron=${process.versions.electron})`);
  try {
    createWindow();
  } catch (err) {
    logLine("FATAL", `createWindow failed: ${err && err.stack || err}`);
    return;
  }
  try {
    pythonProcess = await startPythonBackend();
    await waitForBackend();
    logLine("APP", `backend ready on port ${PORT}`);
    mainWindow?.webContents.send("backend-ready");
  } catch (err) {
    logLine("BACKEND-ERR", err && err.message || String(err));
    mainWindow?.webContents.send("backend-error", err.message);
  }
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
}).catch((err) => {
  logLine("FATAL", `whenReady chain rejected: ${err && err.stack || err}`);
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
