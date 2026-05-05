const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("sybilAPI", {
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
});
