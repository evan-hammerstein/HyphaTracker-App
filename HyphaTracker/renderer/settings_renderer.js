document.addEventListener("DOMContentLoaded", () => {
  const toggleModeButton = document.querySelector(".toggle-mode");
  const manageNotificationsButton = document.querySelector(".manage-notifications");
  const accountSettingsButton = document.querySelector(".account-settings");

  if (toggleModeButton) {
    toggleModeButton.addEventListener("click", () => {
      console.log("Toggling light/dark mode");
      document.body.classList.toggle("dark-mode");
    });
  }

  if (manageNotificationsButton) {
    manageNotificationsButton.addEventListener("click", () => {
      console.log("Opening notifications management");
    });
  }

  if (accountSettingsButton) {
    accountSettingsButton.addEventListener("click", () => {
      console.log("Opening account settings");
    });
  }
});
