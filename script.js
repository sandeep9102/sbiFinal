// Smooth Scroll for Navbar Links
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href.startsWith('#')) {
        e.preventDefault();
        const targetId = href.substring(1);
        const targetSection = document.getElementById(targetId);
        if (targetSection) {
            targetSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
    });
});

// Navbar Button Click
const navButton = document.querySelector('.nav-button');
navButton.addEventListener('click', () => {
    alert('Get Started button clicked!');
});

// Contact Form Validation
const contactForm = document.querySelector('.contact-form');

contactForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const name = this.querySelector('input[type="text"]').value.trim();
    const email = this.querySelector('input[type="email"]').value.trim();
    const message = this.querySelector('textarea').value.trim();

    if (name === '' || email === '' || message === '') {
        alert('Please fill out all fields.');
    } else {
        alert('Thank you for contacting us!');
        this.reset();
    }
});

  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.querySelector('.dropdown-toggle');
    const menu = document.querySelector('.dropdown-menu');

    toggle.addEventListener('click', (e) => {
      e.preventDefault();
      menu.classList.toggle('show');
    });

    document.addEventListener('click', (e) => {
      if (!e.target.closest('.dropdown')) {
        menu.classList.remove('show');
      }
    });
  });
// Select elements
const dropdownBtn = document.getElementById('dropdownBtn');
const dropdownMenu = document.getElementById('dropdownMenu');

// Toggle dropdown on button click
dropdownBtn.addEventListener('click', function(event) {
  event.preventDefault(); // Prevents jumping to top
  dropdownMenu.style.display = (dropdownMenu.style.display === 'block') ? 'none' : 'block';
});

// Close dropdown if clicked outside
window.addEventListener('click', function(event) {
  if (!event.target.matches('#dropdownBtn')) {
    dropdownMenu.style.display = 'none';
  }
});
// Add this to your existing script.js file or create a new one
document.addEventListener('DOMContentLoaded', function() {
  // Dropdown functionality
  const dropdownBtn = document.getElementById('dropdownBtn');
  const dropdownMenu = document.getElementById('dropdownMenu');
  
  dropdownBtn.addEventListener('click', function() {
    dropdownMenu.classList.toggle('show');
  });
  
  // Close dropdown when clicking outside
  window.addEventListener('click', function(event) {
    if (!event.target.matches('#dropdownBtn')) {
      if (dropdownMenu.classList.contains('show')) {
        dropdownMenu.classList.remove('show');
      }
    }
  });
});

