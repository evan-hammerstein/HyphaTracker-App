document.addEventListener("DOMContentLoaded", () => {
    const dismissButtons = document.querySelectorAll(".dismiss-notification");
    const viewButtons = document.querySelectorAll(".view-analysis");
  
    dismissButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        const notificationItem = event.target.closest(".notification-item");
        notificationItem.remove();
        console.log("Notification dismissed");
      });
    });
  
    viewButtons.forEach((button) => {
      button.addEventListener("click", () => {
        console.log("Opening analysis details");
        // Implement viewing functionality here
      });
    });
  });
  