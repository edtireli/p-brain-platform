const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('pbrainLauncher', {
  pickFolder: () => ipcRenderer.invoke('pick-folder'),
});
