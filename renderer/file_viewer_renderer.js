document.addEventListener("DOMContentLoaded", async () => {
  const queryString = new URLSearchParams(window.location.search);
  const filePath = queryString.get("path");

  if (!filePath) {
      document.getElementById("file-content").innerHTML = "<p>Error: No file path provided.</p>";
      return;
  }

  const fileContent = document.getElementById("file-content");
  const downloadButton = document.getElementById("download-button");
  downloadButton.href = `file://${filePath}`;
  downloadButton.download = filePath.split("/").pop();

  const fileType = filePath.split(".").pop().toLowerCase();
  if (["png", "jpg", "jpeg"].includes(fileType)) {
      fileContent.innerHTML = `<img src="file://${filePath}" style="max-width: 100%;">`;
  } else if (fileType === "txt") {
      const text = await window.electron.ipcRenderer.invoke("read-text-file", filePath);
      fileContent.textContent = text;
  } else {
      fileContent.innerHTML = "<p>Unsupported file type for preview.</p>";
  }
});

document.getElementById("close-viewer").addEventListener("click", () => {
  window.history.back();
});
