const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("sybilAPI", {
  platform:            process.platform,

  selectDirectory:     ()              => ipcRenderer.invoke("select-directory"),
  selectCheckpointDir: ()              => ipcRenderer.invoke("select-checkpoint-dir"),
  openFolder:          (p)             => ipcRenderer.invoke("open-folder", p),

  // Short requests (≤60s): health, scan, ct_preview
  apiCall:     (method, endpoint, body) => ipcRenderer.invoke("api-call",      { method, endpoint, body }),
  // Long requests (≤5min): model load
  apiCallLong: (method, endpoint, body) => ipcRenderer.invoke("api-call-long", { method, endpoint, body }),
  // Poll (≤10s): predict/status/:id
  apiPoll:     (endpoint)               => ipcRenderer.invoke("api-poll",      { endpoint }),

  getBackendStatus: ()  => ipcRenderer.invoke("get-backend-status"),
  onBackendReady:  (cb) => ipcRenderer.once("backend-ready", cb),  // EL-05: once, not on
  onBackendError:  (cb) => ipcRenderer.once("backend-error", (_, msg) => cb(msg)),  // EL-05

  // Custom title-bar window controls + theme sync
  windowMinimize: ()       => ipcRenderer.invoke("window:minimize"),
  windowMaximize: ()       => ipcRenderer.invoke("window:maximize"),
  windowClose:    ()       => ipcRenderer.invoke("window:close"),
  windowIsMaximized: ()    => ipcRenderer.invoke("window:is-maximized"),
  setTitleBarTheme: (mode) => ipcRenderer.invoke("window:set-theme", mode),

  // First-run / on-demand environment setup
  runSetup: () => ipcRenderer.invoke("app:run-setup"),
  onSetupProgress: (cb) => {
    const handler = (_evt, line) => { try { cb(line); } catch (_) {} };
    ipcRenderer.on("setup:progress", handler);
    return () => ipcRenderer.removeListener("setup:progress", handler);
  },
});
