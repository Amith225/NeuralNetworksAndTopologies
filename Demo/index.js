const canvas = document.getElementById('drawing-canvas');
const toolbar = document.getElementById('toolbar');
const ctx = canvas.getContext('2d');

let isDown = false;
let beginPoint = null;
let points = [];
let lineWidth = 10;
let lineWidthInput = document.getElementById('lineWidth');
lineWidthInput.value = lineWidth

canvas.width = 500;
canvas.height = 500;
$html_body_pad = 2;  // from style.css
function resize(evt) {
    canvas.width = Math.min(500, window.innerWidth - toolbar.offsetWidth - 4 * $html_body_pad)
}

resize()
window.addEventListener("resize", resize)

// function isTouchDevice() {
//   return (('ontouchstart' in window) || (navigator.maxTouchPoints > 0));
// }

function convertCanvasToImage() {
    let py = document.createElement('py-script');
    py.innerText = " print('hii') ";
    document.body.appendChild(py);
    // window.location.href = canvas.toDataURL("image/png").replace(
    //     "image/png", "image/octet-stream");
}

toolbar.addEventListener('change', e => {
    if (e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }
});

canvas.addEventListener('mousedown', down, false);
document.addEventListener('mousemove', move, false);
document.addEventListener('mouseup', up, false);
canvas.addEventListener('touchstart', down, false);
document.addEventListener('touchmove', move, false);
document.addEventListener('touchend', up, false)

function down(evt) {
    isDown = true;
    points.push(beginPoint = getPos(evt));
}

function move(evt) {
    if (!isDown) return;
    points.push(getPos(evt));
    if (points.length > 3) {
        const lastTwoPoints = points.slice(-2);
        const controlPoint = lastTwoPoints[0];
        const endPoint = {
            x: (lastTwoPoints[0].x + lastTwoPoints[1].x) / 2,
            y: (lastTwoPoints[0].y + lastTwoPoints[1].y) / 2,
        }
        drawLine(beginPoint, controlPoint, endPoint);
        beginPoint = endPoint;
    }
}

function up(evt) {
    if (!isDown) return;
    beginPoint = null;
    isDown = false;
    points = [];
}

function getPos(evt) {
    if (evt.type === 'touchstart' || evt.type === 'touchmove' || evt.type === 'touchend') {
        let touch = evt.touches[0] || evt.changedTouches[0];
        evt.clientX = touch.pageX;
        evt.clientY = touch.pageY;
    }
    let rect = canvas.getBoundingClientRect();
    return {
        x: Math.min(Math.max(evt.clientX - rect.left, lineWidth / 2), canvas.width - lineWidth / 2),
        y: Math.min(Math.max(evt.clientY - rect.top, lineWidth / 2), canvas.height - lineWidth / 2)
    }
}

function drawLine(beginPoint, controlPoint, endPoint) {
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(beginPoint.x, beginPoint.y);
    ctx.quadraticCurveTo(controlPoint.x, controlPoint.y, endPoint.x, endPoint.y);
    ctx.stroke();
    ctx.closePath();
}