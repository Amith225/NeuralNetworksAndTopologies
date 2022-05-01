const canvas = document.getElementById('drawing-canvas');
const toolbar = document.getElementById('toolbar');
const ctx = canvas.getContext('2d');

let isDown = false;
let beginPoint = null;
let points = [];
let lineWidth = 10;

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

function convertCanvasToArray() {
    let matrix = []
    let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let pos = null;
    for (let j = 0; j < canvas.height; j++) {
        let my = []
        for (let i = 0; i < canvas.width; i++) {
            pos = (j * canvas.width + i) * 4;
            my[i] = imgData[pos + 3]
        }
        matrix[j] = my
    }
    return matrix
}

toolbar.addEventListener('change', e => {
    if (e.target.id === 'stroke') {
        ctx.strokeStyle = e.target.value;
    }

    if (e.target.id === 'lineWidth') {
        lineWidth = e.target.value;
    }

});

document.addEventListener('mousedown', down, false);
document.addEventListener('mousemove', move, false);
document.addEventListener('mouseup', up, false);
document.addEventListener('touchstart', down, false);
document.addEventListener('touchmove', move, false);
document.addEventListener('touchend', up, false)

function down(evt) {
    if (evt.x )
    isDown = true;
    const {x, y} = getPos(evt);
    points.push({x, y});
    beginPoint = {x, y};
}

function move(evt) {
    if (!isDown) return;
    const {x, y} = getPos(evt);
    points.push({x, y});

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
    const {x, y} = getPos(evt);
    points.push({x, y});

    if (points.length > 3) {
        const lastTwoPoints = points.slice(-2);
        const controlPoint = lastTwoPoints[0];
        const endPoint = lastTwoPoints[1];
        drawLine(beginPoint, controlPoint, endPoint);
    }
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
