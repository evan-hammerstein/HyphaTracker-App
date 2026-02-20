document.addEventListener("DOMContentLoaded", () => {
    const uploadButton = document.getElementById("upload-button");
    const goButton = document.getElementById("go-button");
    let filePath = null;

    // File Upload Logic
    uploadButton.addEventListener("click", async () => {
        console.log("Upload button clicked for New Analysis");
        try {
            const filePaths = await window.electron.ipcRenderer.invoke("open-file-dialog");
            if (filePaths && filePaths.length > 0) {
                filePath = filePaths[0];
                document.getElementById("uploaded-file-name").textContent = filePath.split("/").pop();
            } else {
                console.log("No file selected");
            }
        } catch (error) {
            console.error("Error during file upload:", error);
        }
    });

    // GO Button Logic
    goButton.addEventListener("click", () => {
        if (!filePath) {
            alert("Please upload a .tif file before proceeding.");
            return;
        }

        const Dimension = document.getElementById("dimension").value;
        const background = document.getElementById("background").value;
        const sensitivity = document.getElementById("sensitivity").value;

        // Send analysis request to the main process
        window.electron.ipcRenderer.send("start-analysis", {
            filePath,
            Dimension,
            background,
            sensitivity,
        });

        window.electron.ipcRenderer.once("analysis-completed", (event, data) => {
            if (data.success) {
                alert("Analysis Success!");
                window.electron.ipcRenderer.send("navigate-to", "previous_analysis");
            } else {
                alert("Analysis Failed. Please try again.");
            }
        });
    });
});
