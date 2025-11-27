 // Get modal and buttons
    var modal = document.getElementById("assessmentModal");
    var btn = document.getElementById("assessmentBtn");
    var span = document.getElementById("closeModalBtn");

    // Open the modal when the button is clicked
    btn.onclick = function() {
      modal.style.display = "block";
    }

    // Close the modal when the "x" button is clicked
    span.onclick = function() {
      modal.style.display = "none";
    }

    // Close the modal if the user clicks outside the modal
    window.onclick = function(event) {
      if (event.target == modal) {
        modal.style.display = "none";
      }
    }
