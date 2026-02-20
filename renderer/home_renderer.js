document.addEventListener("DOMContentLoaded", () => {
    // Display the current date
    const setCurrentDate = () => {
      const currentDate = new Date();
      const options = { weekday: "long", year: "numeric", month: "long", day: "numeric" };
      document.getElementById("current-date").textContent = currentDate.toLocaleDateString(undefined, options);
    };
    setCurrentDate();
  
    // Populate the calendar
    const populateCalendar = () => {
      const calendarGrid = document.getElementById("calendar-grid");
      const currentDate = new Date();
      const year = currentDate.getFullYear();
      const month = currentDate.getMonth();
  
      const firstDay = new Date(year, month, 1).getDay();
      const daysInMonth = new Date(year, month + 1, 0).getDate();
  
      // Clear the calendar grid
      calendarGrid.innerHTML = "";
  
      // Fill in blank days before the first day of the month
      for (let i = 0; i < firstDay; i++) {
        const blankDay = document.createElement("div");
        blankDay.classList.add("blank-day");
        calendarGrid.appendChild(blankDay);
      }
  
      // Fill in the days of the month
      for (let day = 1; day <= daysInMonth; day++) {
        const dayElement = document.createElement("div");
        dayElement.textContent = day;
        dayElement.classList.add("day");
  
        // Highlight the current day
        if (day === currentDate.getDate()) {
          dayElement.classList.add("current-day");
        }
  
        calendarGrid.appendChild(dayElement);
      }
    };
    populateCalendar();
  });
  