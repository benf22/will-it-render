document.addEventListener("DOMContentLoaded", function () {
  var btn = document.getElementById("copy-bibtex");
  var text = document.getElementById("bibtex-text");
  if (!btn || !text) return;
  btn.addEventListener("click", function () {
    navigator.clipboard.writeText(text.textContent).then(function () {
      btn.classList.add("copied");
      btn.querySelector(".copy-text").textContent = "Copied";
      setTimeout(function () {
        btn.classList.remove("copied");
        btn.querySelector(".copy-text").textContent = "Copy";
      }, 1500);
    });
  });
});
