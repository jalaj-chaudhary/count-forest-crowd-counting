/* Small progressive enhancements for the report. Everything here is optional:
   if this script fails, the page still reads correctly. */
(function () {
  "use strict";

  function numberChips() {
    // "1. Introduction" -> [1] Introduction
    document.querySelectorAll(".md-typeset h2").forEach(function (h) {
      if (h.querySelector(".secnum")) return;
      var node = h.firstChild;
      if (!node || node.nodeType !== 3) return;
      var m = node.nodeValue.match(/^\s*(\d+)\.\s+([\s\S]*)$/);
      if (!m) return;
      var chip = document.createElement("span");
      chip.className = "secnum";
      chip.textContent = m[1];
      node.nodeValue = m[2];
      h.insertBefore(chip, node);
    });
  }

  function progressBar() {
    if (document.getElementById("cf-progress")) return;
    var bar = document.createElement("div");
    bar.id = "cf-progress";
    document.body.appendChild(bar);
    var ticking = false;
    function update() {
      var doc = document.documentElement;
      var max = doc.scrollHeight - doc.clientHeight;
      var p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      bar.style.transform = "scaleX(" + p + ")";
      ticking = false;
    }
    window.addEventListener(
      "scroll",
      function () {
        if (!ticking) {
          ticking = true;
          window.requestAnimationFrame(update);
        }
      },
      { passive: true }
    );
    update();
  }

  function lightbox() {
    var box = document.getElementById("cf-lightbox");
    if (!box) {
      box = document.createElement("div");
      box.id = "cf-lightbox";
      box.appendChild(document.createElement("img"));
      document.body.appendChild(box);
      box.addEventListener("click", function () {
        box.classList.remove("open");
      });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape") box.classList.remove("open");
      });
    }
    document.querySelectorAll(".md-typeset figure img").forEach(function (img) {
      if (img.dataset.zoom) return;
      img.dataset.zoom = "1";
      img.title = "Click to enlarge";
      img.addEventListener("click", function () {
        var big = box.querySelector("img");
        big.src = img.currentSrc || img.src;
        big.alt = img.alt;
        box.classList.add("open");
      });
    });
  }

  function init() {
    numberChips();
    progressBar();
    lightbox();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
