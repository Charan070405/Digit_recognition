// static/js/app.js
const canvas = document.getElementById("drawArea");
const ctx = canvas.getContext("2d");

// choose canvas size you used in HTML
canvas.width = canvas.width || 400;
canvas.height = canvas.height || 200;

// black background (white strokes)
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// brush
ctx.strokeStyle = "white";
ctx.lineWidth = 24;
ctx.lineCap = "round";
ctx.lineJoin = "round";

let drawing = false;

function getXY(e) {
    const rect = canvas.getBoundingClientRect();
    if (e.touches) {
        return { x: e.touches[0].clientX - rect.left, y: e.touches[0].clientY - rect.top };
    }
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    const { x, y } = getXY(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
});
canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const { x, y } = getXY(e);
    ctx.lineTo(x, y);
    ctx.stroke();
});
canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener("mouseleave", () => {
    drawing = false;
    ctx.beginPath();
});

// touch support
canvas.addEventListener("touchstart", (e) => {
    e.preventDefault();
    drawing = true;
    const { x, y } = getXY(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
});
canvas.addEventListener("touchmove", (e) => {
    e.preventDefault();
    if (!drawing) return;
    const { x, y } = getXY(e);
    ctx.lineTo(x, y);
    ctx.stroke();
});
canvas.addEventListener("touchend", () => {
    drawing = false;
    ctx.beginPath();
});

document.getElementById("clear-btn").addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
});

document.getElementById("predict-draw-btn").addEventListener("click", () => {
    const dataURL = canvas.toDataURL("image/png");
    const formData = new FormData();
    formData.append("image", dataURL);

    fetch("/predict_canvas", { method: "POST", body: formData })
      .then(r => r.json())
      .then(data => {
          if (data.error) { alert("Prediction error"); return; }
          window.location.href = `/result_from_canvas?label=${data.label}&image=${data.image_url}`;
      })
      .catch(err => { console.error(err); alert("Prediction failed"); });
});
