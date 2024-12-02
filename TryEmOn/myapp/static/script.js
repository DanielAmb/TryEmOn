document.querySelectorAll(".clothes-options input").forEach((input) => {
    input.addEventListener("change", function () {
        document.querySelectorAll(".clothes-options img").forEach((img) => {
            img.style.border = "2px solid #ccc";
        });
        this.nextElementSibling.style.border = "2px solid blue";
    });
});