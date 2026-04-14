const canvas = document.getElementById('paintCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultContainer = document.getElementById('result');
const digitVal = document.getElementById('digitVal');
const confidenceVal = document.getElementById('confidenceVal');

// Initialize Canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15; // Similar to Tkinter brush size roughly
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'white';

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// To handle mouse/touch seamlessly
function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches && e.touches.length > 0) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    isDrawing = true;
    const { x, y } = getCoordinates(e);
    lastX = x;
    lastY = y;
    
    // Draw a single dot if user just clicks
    ctx.beginPath();
    ctx.fillStyle = 'white';
    ctx.arc(x, y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    
    // For smooth lines
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault(); // Prevent scrolling on touch
    
    const { x, y } = getCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    lastX = x;
    lastY = y;
}

function stopDrawing() {
    isDrawing = false;
    ctx.closePath();
}

// Event Listeners for drawing
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);

// Clear Canvas
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultContainer.classList.add('hidden');
});

// Predict
predictBtn.addEventListener('click', async () => {
    // Show a loading state
    const originalText = predictBtn.textContent;
    predictBtn.textContent = 'Predicting...';
    predictBtn.disabled = true;

    try {
        const dataURL = canvas.toDataURL('image/png'); // Sends what's exactly on canvas
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: dataURL })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            digitVal.textContent = result.digit;
            confidenceVal.textContent = (result.confidence * 100).toFixed(2) + '%';
            resultContainer.classList.remove('hidden');
        } else {
            alert(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Could not connect to the server.');
    } finally {
        predictBtn.textContent = originalText;
        predictBtn.disabled = false;
    }
});
