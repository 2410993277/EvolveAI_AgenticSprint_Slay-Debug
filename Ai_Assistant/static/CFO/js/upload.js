// minimal: show filename in console / future enhancements
document.addEventListener('DOMContentLoaded', function(){
  const fileInput = document.getElementById('fileInput');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (f) {
        console.log('Selected file', f.name);
      }
    });
  }
});
