// Narrow, typed bridge between the renderer and the main process. Nothing else is exposed.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('anotara', {
  isDesktop: true,
  getAppInfo: () => ipcRenderer.invoke('anotara:app-info'),
  openLogs: () => ipcRenderer.invoke('anotara:open-logs'),
  retryStart: () => ipcRenderer.invoke('anotara:retry-start'),
  update: {
    getState: () => ipcRenderer.invoke('anotara:update:get-state'),
    check: () => ipcRenderer.invoke('anotara:update:check'),
    download: () => ipcRenderer.invoke('anotara:update:download'),
    install: () => ipcRenderer.invoke('anotara:update:install'),
    onState: listener => {
      const handler = (_event, state) => listener(state);
      ipcRenderer.on('anotara:update:state', handler);
      return () => ipcRenderer.removeListener('anotara:update:state', handler);
    },
  },
});
