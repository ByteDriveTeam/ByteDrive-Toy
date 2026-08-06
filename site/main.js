const tocLinks = [...document.querySelectorAll('.toc a[href^="#"]')];
const observedSections = tocLinks.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);

if ("IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (!visible.length) return;
    const id = `#${visible[0].target.id}`;
    tocLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === id));
  }, { rootMargin: "-22% 0px -68% 0px", threshold: 0.01 });
  observedSections.forEach((section) => sectionObserver.observe(section));
}

const dialog = document.querySelector("[data-lightbox-dialog]");
const dialogImage = dialog.querySelector("img");
document.querySelectorAll("[data-lightbox]").forEach((button) => button.addEventListener("click", () => {
  dialogImage.src = button.dataset.lightbox;
  dialog.showModal();
}));
document.querySelector("[data-lightbox-close]").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});
