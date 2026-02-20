document.addEventListener("DOMContentLoaded", async () => {
    const folderContents = document.getElementById("folder-contents");
    const queryString = new URLSearchParams(window.location.search);
    const folderPath = queryString.get("path");

    const items = await window.electron.ipcRenderer.invoke("get-folder-contents", folderPath);

    folderContents.innerHTML = ""; // Clear current contents

    if (items.length === 0) {
        folderContents.innerHTML = "<tr><td colspan='3'>No contents found.</td></tr>";
        return;
    }

    items.forEach(item => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${item.name}</td>
            <td>${item.type}</td>
            <td>
                <button class="action-button" data-type="${item.type}" data-path="${item.path}">View</button>
            </td>
        `;
        folderContents.appendChild(row);
    });

    document.querySelectorAll(".action-button").forEach(button => {
        button.addEventListener("click", (e) => {
            const type = e.target.getAttribute("data-type");
            const path = e.target.getAttribute("data-path");
            if (type === "folder") {
                window.electron.ipcRenderer.send("navigate-to-folder", path);
            } else {
                window.electron.ipcRenderer.send("navigate-to-file", path);
            }
        });
    });
});
