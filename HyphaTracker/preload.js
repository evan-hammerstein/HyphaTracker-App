const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electron", {
    ipcRenderer: {
        invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
        send: (channel, ...args) => ipcRenderer.send(channel, ...args),
        on: (channel, listener) => ipcRenderer.on(channel, listener),
        once: (channel, listener) => ipcRenderer.once(channel, listener),
        removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel),
    },
    navigateTo: (screenName) => {
        console.log(`Navigating to screen: ${screenName}`); // Debug log
        ipcRenderer.send("navigate-to", screenName);
    },
});
