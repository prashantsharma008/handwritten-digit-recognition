/* ============================================
   JAVASCRIPT — Digit Recognition App
   ============================================ */

document.addEventListener("DOMContentLoaded", () => {
    // ── DOM Elements ──
    const drawCanvas = document.getElementById("drawCanvas");
    const ctx = drawCanvas.getContext("2d");
    const canvasHint = document.getElementById("canvasHint");
    const brushSlider = document.getElementById("brushSize");
    const brushPreview = document.getElementById("brushPreview");
    const clearBtn = document.getElementById("clearBtn");
    const predictDrawBtn = document.getElementById("predictDrawBtn");

    const tabSwitcher = document.getElementById("tabSwitcher");
    const uploadTab = document.getElementById("uploadTab");
    const drawTab = document.getElementById("drawTab");

    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadPreview = document.getElementById("uploadPreview");
    const uploadPlaceholder = document.getElementById("uploadPlaceholder");
    const clearUploadBtn = document.getElementById("clearUploadBtn");
    const predictUploadBtn = document.getElementById("predictUploadBtn");

    // Result elements
    const resultCard = document.getElementById("resultCard");
    const resultCardEmpty = document.getElementById("resultCardEmpty");
    const resultCardLoading = document.getElementById("resultCardLoading");
    const resultCardContent = document.getElementById("resultCardContent");
    const predictedDigit = document.getElementById("predictedDigit");
    const confidenceText = document.getElementById("confidenceText");
    const resultsRow = document.getElementById("resultsRow");
    const topDigit = document.getElementById("topDigit");
    const modelPreview = document.getElementById("modelPreview");
    const probBars = document.getElementById("probBars");

    // ── Tab Switching ──
    tabSwitcher.addEventListener("click", (e) => {
        const tab = e.target.closest(".tab");
        if (!tab) return;
        tabSwitcher.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const target = tab.dataset.tab;
        uploadTab.classList.toggle("active", target === "upload");
        drawTab.classList.toggle("active", target === "draw");
    });

    // ── Canvas Drawing ──
    let isDrawing = false;
    let hasDrawn = false;
    let lastX = 0, lastY = 0;

    function initCanvas() {
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    }
    initCanvas();

    function getCanvasCoords(e) {
        const rect = drawCanvas.getBoundingClientRect();
        const scaleX = drawCanvas.width / rect.width;
        const scaleY = drawCanvas.height / rect.height;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        hasDrawn = true;
        canvasHint.classList.add("hidden");
        const coords = getCanvasCoords(e);
        lastX = coords.x;
        lastY = coords.y;

        // Draw initial point so single clicks/taps register immediately
        const brushSize = parseInt(brushSlider.value);
        ctx.beginPath();
        ctx.fillStyle = "#fff";
        ctx.arc(coords.x, coords.y, brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCanvasCoords(e);
        const brushSize = parseInt(brushSlider.value);

        ctx.beginPath();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = brushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        lastX = coords.x;
        lastY = coords.y;
    }

    function stopDrawing() {
        isDrawing = false;
    }

    // Mouse events
    drawCanvas.addEventListener("mousedown", startDrawing);
    drawCanvas.addEventListener("mousemove", draw);
    drawCanvas.addEventListener("mouseup", stopDrawing);
    drawCanvas.addEventListener("mouseleave", stopDrawing);

    // Touch events
    drawCanvas.addEventListener("touchstart", startDrawing, { passive: false });
    drawCanvas.addEventListener("touchmove", draw, { passive: false });
    drawCanvas.addEventListener("touchend", stopDrawing);

    // Brush preview
    function updateBrushPreview() {
        const size = parseInt(brushSlider.value);
        brushPreview.style.width = size + "px";
        brushPreview.style.height = size + "px";
    }
    brushSlider.addEventListener("input", updateBrushPreview);
    updateBrushPreview();

    // Clear canvas
    clearBtn.addEventListener("click", () => {
        initCanvas();
        hasDrawn = false;
        canvasHint.classList.remove("hidden");
        showResultState("empty");
    });

    // Predict drawn digit
    predictDrawBtn.addEventListener("click", () => {
        if (!hasDrawn) return;
        const dataURL = drawCanvas.toDataURL("image/png");
        sendPrediction(dataURL);
    });

    // ── File Upload ──
    uploadZone.addEventListener("click", () => {
        if (!uploadZone.classList.contains("has-image")) {
            fileInput.click();
        }
    });

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("dragover");
    });
    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });
    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    let uploadedDataURL = null;

    function handleFile(file) {
        if (!file.type.startsWith("image/")) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedDataURL = e.target.result;
            uploadPreview.src = uploadedDataURL;
            uploadPreview.style.display = "block";
            uploadZone.classList.add("has-image");
        };
        reader.readAsDataURL(file);
    }

    clearUploadBtn.addEventListener("click", () => {
        uploadedDataURL = null;
        uploadPreview.style.display = "none";
        uploadPreview.src = "";
        uploadZone.classList.remove("has-image");
        fileInput.value = "";
        showResultState("empty");
    });

    predictUploadBtn.addEventListener("click", () => {
        if (!uploadedDataURL) return;
        sendPrediction(uploadedDataURL);
    });

    // ── Result State Management ──
    function showResultState(state) {
        resultCardEmpty.style.display = state === "empty" ? "block" : "none";
        resultCardLoading.style.display = state === "loading" ? "flex" : "none";
        resultCardContent.style.display = state === "result" ? "block" : "none";
        resultsRow.style.display = state === "result" ? "grid" : "none";
    }

    // ── Prediction API Call ──
    async function sendPrediction(imageDataURL) {
        showResultState("loading");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imageDataURL })
            });

            const data = await response.json();

            if (!data.success) {
                alert("Prediction failed: " + (data.error || "Unknown error"));
                showResultState("empty");
                return;
            }

            displayResult(data);
        } catch (error) {
            alert("Error connecting to server: " + error.message);
            showResultState("empty");
        }
    }

    // ── Display Result ──
    function displayResult(data) {
        showResultState("result");

        // Main digit card
        predictedDigit.textContent = data.digit;
        predictedDigit.style.animation = "none";
        requestAnimationFrame(() => {
            predictedDigit.style.animation = "digitPop 0.4s cubic-bezier(0.16, 1, 0.3, 1)";
        });

        confidenceText.textContent = data.confidence.toFixed(1) + "%";

        // Model preview
        modelPreview.src = data.preview;

        // Top digit in predictions box
        topDigit.textContent = data.digit;

        // Probability bars — show top 3
        probBars.innerHTML = "";
        const sorted = [...data.confidences];
        const top3 = sorted.slice(0, 3);

        top3.forEach((item, index) => {
            const row = document.createElement("div");
            row.classList.add("prob-row");

            const isFirst = index === 0;

            row.innerHTML = `
                <span class="prob-digit">${item.digit}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill ${isFirst ? '' : 'secondary'}" id="bar-${item.digit}"></div>
                </div>
                <span class="prob-value ${isFirst ? 'highlight' : ''}">${item.confidence.toFixed(0)}%</span>
            `;

            probBars.appendChild(row);

            // Animate bar width
            setTimeout(() => {
                const bar = document.getElementById(`bar-${item.digit}`);
                if (bar) bar.style.width = item.confidence + "%";
            }, 100 + index * 80);
        });
    }
});
