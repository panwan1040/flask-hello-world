// YOLO Object Detection with ONNX.js
class YOLODetector {
    constructor() {
        this.session = null;
        this.modelLoaded = false;
        this.confidenceThreshold = 0.5;
        this.iouThreshold = 0.45;
        this.inputShape = [1, 3, 640, 640]; // YOLOv8 default
        this.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ];
    }

    async loadModel() {
        try {
            updateStatus('กำลังโหลดโมเดล YOLO...', 'loading');

            // สร้าง ONNX session
            this.session = await ort.InferenceSession.create('./model.onnx', {
                executionProviders: ['wasm'] // ใช้ WebAssembly สำหรับความเข้ากันได้สูงสุด
            });

            this.modelLoaded = true;
            updateStatus('✅ โมเดลพร้อมใช้งาน', 'success');

            console.log('✅ โมเดล YOLO โหลดสำเร็จ');
            return true;
        } catch (error) {
            console.error('❌ เกิดข้อผิดพลาดในการโหลดโมเดล:', error);
            updateStatus('❌ ไม่สามารถโหลดโมเดลได้: ' + error.message, 'error');
            return false;
        }
    }

    async detect(imageElement) {
        if (!this.modelLoaded) {
            throw new Error('โมเดลยังไม่ได้โหลด');
        }

        const startTime = performance.now();

        try {
            // เตรียมรูปภาพ
            const tensor = await this.preprocessImage(imageElement);

            // ทำการ inference
            const feeds = { images: tensor };
            const results = await this.session.run(feeds);

            // ประมวลผลผลลัพธ์
            const detections = this.processResults(results);

            const endTime = performance.now();
            const processingTime = Math.round(endTime - startTime);

            return {
                detections: detections,
                processingTime: processingTime
            };
        } catch (error) {
            console.error('❌ เกิดข้อผิดพลาดในการตรวจจับ:', error);
            throw error;
        }
    }

    async preprocessImage(imageElement) {
        // สร้าง canvas สำหรับประมวลผลรูปภาพ
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // ตั้งขนาด canvas
        canvas.width = this.inputShape[3];
        canvas.height = this.inputShape[2];

        // คำนวณ aspect ratio และ resize
        const aspectRatio = imageElement.width / imageElement.height;
        let drawWidth, drawHeight, offsetX, offsetY;

        if (aspectRatio > 1) {
            // รูปภาพกว้างกว่าสูง
            drawWidth = canvas.width;
            drawHeight = canvas.width / aspectRatio;
            offsetX = 0;
            offsetY = (canvas.height - drawHeight) / 2;
        } else {
            // รูปภาพสูงกว่ากว้าง
            drawWidth = canvas.height * aspectRatio;
            drawHeight = canvas.height;
            offsetX = (canvas.width - drawWidth) / 2;
            offsetY = 0;
        }

        // วาดรูปภาพลงบน canvas
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(imageElement, offsetX, offsetY, drawWidth, drawHeight);

        // แปลงเป็น tensor
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const { data, width, height } = imageData;

        // Normalize และแปลงเป็น RGB
        const input = new Float32Array(width * height * 3);
        for (let i = 0; i < width * height; i++) {
            const pixelIndex = i * 4;
            // RGB to BGR และ normalize
            input[i] = data[pixelIndex + 2] / 255.0;     // R
            input[i + width * height] = data[pixelIndex + 1] / 255.0;     // G
            input[i + 2 * width * height] = data[pixelIndex] / 255.0;     // B
        }

        // สร้าง ONNX tensor
        return new ort.Tensor('float32', input, [1, 3, height, width]);
    }

    processResults(results) {
        const output = results.output0; // YOLOv8 output tensor
        const detections = [];

        // YOLOv8 output format: [batch, num_boxes, 84] for COCO dataset (80 classes + 4 bbox)
        const numBoxes = output.dims[1];
        const numClasses = 80; // COCO dataset has 80 classes

        for (let i = 0; i < numBoxes; i++) {
            const offset = i * (numClasses + 4);

            // Extract bounding box
            const x = output.data[offset];
            const y = output.data[offset + 1];
            const w = output.data[offset + 2];
            const h = output.data[offset + 3];

            // Find class with highest confidence
            let maxConf = 0;
            let maxClass = 0;
            for (let j = 0; j < numClasses; j++) {
                const conf = output.data[offset + 4 + j];
                if (conf > maxConf) {
                    maxConf = conf;
                    maxClass = j;
                }
            }

            // Filter by confidence threshold
            if (maxConf > this.confidenceThreshold) {
                detections.push({
                    bbox: [x, y, w, h],
                    confidence: maxConf,
                    class: maxClass,
                    className: this.classes[maxClass] || `class_${maxClass}`
                });
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        return this.nms(detections);
    }

    nms(detections) {
        // Sort by confidence (descending)
        detections.sort((a, b) => b.confidence - a.confidence);

        const selected = [];

        for (const detection of detections) {
            let shouldAdd = true;

            for (const selectedDet of selected) {
                const iou = this.calculateIoU(detection.bbox, selectedDet.bbox);
                if (iou > this.iouThreshold) {
                    shouldAdd = false;
                    break;
                }
            }

            if (shouldAdd) {
                selected.push(detection);
            }
        }

        return selected;
    }

    calculateIoU(box1, box2) {
        // Convert from center format to corner format
        const box1Left = box1[0] - box1[2] / 2;
        const box1Top = box1[1] - box1[3] / 2;
        const box1Right = box1[0] + box1[2] / 2;
        const box1Bottom = box1[1] + box1[3] / 2;

        const box2Left = box2[0] - box2[2] / 2;
        const box2Top = box2[1] - box2[3] / 2;
        const box2Right = box2[0] + box2[2] / 2;
        const box2Bottom = box2[1] + box2[3] / 2;

        // Calculate intersection
        const interLeft = Math.max(box1Left, box2Left);
        const interTop = Math.max(box1Top, box2Top);
        const interRight = Math.min(box1Right, box2Right);
        const interBottom = Math.min(box1Bottom, box2Bottom);

        if (interRight <= interLeft || interBottom <= interTop) {
            return 0;
        }

        const interArea = (interRight - interLeft) * (interBottom - interTop);

        // Calculate union
        const box1Area = (box1Right - box1Left) * (box1Bottom - box1Top);
        const box2Area = (box2Right - box2Left) * (box2Bottom - box2Top);
        const unionArea = box1Area + box2Area - interArea;

        return interArea / unionArea;
    }

    setConfidenceThreshold(threshold) {
        this.confidenceThreshold = threshold;
    }

    setIoUThreshold(threshold) {
        this.iouThreshold = threshold;
    }
}

// Global variables
let detector = null;
let currentImage = null;
let currentDetections = [];

// Camera variables
let cameraStream = null;
let cameraActive = false;
let cameraCanvas = null;
let cameraContext = null;
let animationFrame = null;
let lastDetectionTime = 0;

// Initialize
async function init() {
    detector = new YOLODetector();

    // โหลดโมเดล
    await detector.loadModel();

    // ตั้งค่าเริ่มต้นเป็นโหมดอัปโหลด
    switchMode('upload');

    // ตั้งค่า event listeners
    setupEventListeners();
}

function setupEventListeners() {
    // File input
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    const uploadSection = document.getElementById('upload-section');
    uploadSection.addEventListener('dragover', handleDragOver);
    uploadSection.addEventListener('drop', handleDrop);

    // Settings
    const confidenceSlider = document.getElementById('confidence-slider');
    const iouSlider = document.getElementById('iou-slider');

    confidenceSlider.addEventListener('input', function() {
        const value = parseFloat(this.value);
        document.getElementById('confidence-value').textContent = (value * 100).toFixed(1) + '%';
        detector.setConfidenceThreshold(value);
    });

    iouSlider.addEventListener('input', function() {
        const value = parseFloat(this.value);
        document.getElementById('iou-value').textContent = (value * 100).toFixed(1) + '%';
        detector.setIoUThreshold(value);
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        processImage(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.style.background = 'rgba(76, 237, 196, 0.2)';
}

function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)';

    const files = event.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        processImage(files[0]);
    }
}

async function processImage(file) {
    try {
        updateStatus('⏳ กำลังประมวลผลรูปภาพ...', 'loading');

        // สร้าง Image object
        const img = new Image();
        img.onload = async function() {
            currentImage = img;
            await detectObjects();
        };

        // อ่านไฟล์เป็น data URL
        const reader = new FileReader();
        reader.onload = function(e) {
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);

    } catch (error) {
        console.error('❌ เกิดข้อผิดพลาดในการประมวลผลรูปภาพ:', error);
        updateStatus('❌ เกิดข้อผิดพลาด: ' + error.message, 'error');
    }
}

async function detectObjects() {
    if (!currentImage || !detector.modelLoaded) {
        return;
    }

    try {
        updateStatus('🔍 กำลังตรวจจับวัตถุ...', 'loading');

        // ทำการตรวจจับ
        const result = await detector.detect(currentImage);
        currentDetections = result.detections;

        // แสดงผลลัพธ์
        displayResults(result);

        updateStatus('✅ ตรวจจับวัตถุสำเร็จ!', 'success');

    } catch (error) {
        console.error('❌ เกิดข้อผิดพลาดในการตรวจจับ:', error);
        updateStatus('❌ เกิดข้อผิดพลาดในการตรวจจับ: ' + error.message, 'error');
    }
}

function displayResults(result) {
    const uploadSection = document.getElementById('upload-section');
    const resultSection = document.getElementById('result-section');
    const canvas = document.getElementById('result-canvas');

    // ซ่อนส่วนอัปโหลดและแสดงส่วนผลลัพธ์
    uploadSection.style.display = 'none';
    resultSection.style.display = 'block';

    // วาดผลลัพธ์ลงบน canvas
    drawDetections(canvas, currentImage, result.detections);

    // อัปเดตสถิติ
    updateStats(result);
}

function drawDetections(canvas, image, detections) {
    const ctx = canvas.getContext('2d');

    // ตั้งขนาด canvas ให้ตรงกับรูปภาพต้นฉบับ
    canvas.width = image.width;
    canvas.height = image.height;

    // วาดรูปภาพต้นฉบับ
    ctx.drawImage(image, 0, 0);

    // วาด bounding boxes
    detections.forEach((detection, index) => {
        const [x, y, w, h] = detection.bbox;

        // Convert from normalized coordinates to pixel coordinates
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;

        const boxX = (x - w/2) * canvasWidth;
        const boxY = (y - h/2) * canvasHeight;
        const boxWidth = w * canvasWidth;
        const boxHeight = h * canvasHeight;

        // วาด bounding box
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 3;
        ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

        // วาดพื้นหลังสำหรับ text
        ctx.fillStyle = 'rgba(255, 107, 107, 0.8)';
        const textHeight = 24;
        const textPadding = 8;
        const text = `${detection.className} ${(detection.confidence * 100).toFixed(1)}%`;

        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(boxX, boxY - textHeight - textPadding, textWidth + textPadding * 2, textHeight + textPadding);

        // วาด text
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.fillText(text, boxX + textPadding, boxY - textPadding);
    });
}

function updateStats(result) {
    const objectsCount = result.detections.length;
    const avgConfidence = objectsCount > 0
        ? (result.detections.reduce((sum, det) => sum + det.confidence, 0) / objectsCount * 100).toFixed(1)
        : 0;

    document.getElementById('objects-count').textContent = objectsCount;
    document.getElementById('avg-confidence').textContent = avgConfidence + '%';
    document.getElementById('processing-time').textContent = result.processingTime + 'ms';
}

function updateStatus(message, type = 'info') {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.className = 'status';

    if (type === 'success') {
        statusElement.classList.add('success');
    } else if (type === 'error') {
        statusElement.classList.add('error');
    }
}

// Utility functions
function detectAgain() {
    if (currentImage) {
        detectObjects();
    }
}

function clearResult() {
    const uploadSection = document.getElementById('upload-section');
    const resultSection = document.getElementById('result-section');
    const fileInput = document.getElementById('file-input');
    const canvas = document.getElementById('result-canvas');

    // ล้าง canvas
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // รีเซ็ต UI
    uploadSection.style.display = 'block';
    resultSection.style.display = 'none';
    fileInput.value = '';

    // รีเซ็ตตัวแปร
    currentImage = null;
    currentDetections = [];

    updateStatus('พร้อมสำหรับการอัปโหลดรูปภาพใหม่', 'success');
}

function downloadResult() {
    const canvas = document.getElementById('result-canvas');
    if (canvas.width > 0 && canvas.height > 0) {
        // สร้าง download link
        const link = document.createElement('a');
        link.download = 'detection_result.png';
        link.href = canvas.toDataURL();
        link.click();
    }
}

// Mode switching
function switchMode(mode) {
    const uploadModeBtn = document.getElementById('upload-mode-btn');
    const cameraModeBtn = document.getElementById('camera-mode-btn');
    const uploadSection = document.getElementById('upload-section');
    const cameraSection = document.getElementById('camera-section');
    const resultSection = document.getElementById('result-section');

    // Reset active states
    uploadModeBtn.classList.remove('active');
    cameraModeBtn.classList.remove('active');

    // Hide all sections
    uploadSection.style.display = 'none';
    cameraSection.style.display = 'none';
    resultSection.style.display = 'none';

    // Show selected mode
    if (mode === 'upload') {
        uploadModeBtn.classList.add('active');
        uploadSection.style.display = 'block';
        updateStatus('พร้อมสำหรับการอัปโหลดรูปภาพ', 'success');

        // Stop camera if running
        if (cameraActive) {
            stopCamera();
        }
    } else if (mode === 'camera') {
        cameraModeBtn.classList.add('active');
        cameraSection.style.display = 'block';
        updateStatus('คลิก "เริ่มกล้อง" เพื่อเปิดกล้อง', 'info');

        // Hide result section when switching to camera mode
        document.getElementById('result-section').style.display = 'none';
    }
}

// Camera functions
async function startCamera() {
    if (cameraActive) return;

    try {
        updateStatus('กำลังเปิดกล้อง...', 'loading');

        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user' // Use front camera by default
            },
            audio: false
        });

        cameraStream = stream;
        const video = document.getElementById('camera-video');
        video.srcObject = stream;

        // Wait for video to load
        await new Promise(resolve => {
            video.onloadedmetadata = resolve;
        });

        // Initialize canvas for processing
        cameraCanvas = document.getElementById('camera-canvas');
        cameraContext = cameraCanvas.getContext('2d');
        cameraCanvas.width = video.videoWidth;
        cameraCanvas.height = video.videoHeight;

        cameraActive = true;

        // Update UI
        document.getElementById('start-camera-btn').style.display = 'none';
        document.getElementById('stop-camera-btn').style.display = 'inline-block';
        document.getElementById('capture-btn').style.display = 'inline-block';

        updateStatus('✅ กล้องพร้อมใช้งาน', 'success');

        // Start real-time detection
        startRealTimeDetection();

    } catch (error) {
        console.error('❌ เกิดข้อผิดพลาดในการเปิดกล้อง:', error);
        updateStatus('❌ ไม่สามารถเปิดกล้องได้: ' + error.message, 'error');

        // Check if it's a permission issue
        if (error.name === 'NotAllowedError') {
            updateStatus('❌ กรุณาอนุญาตการเข้าถึงกล้อง', 'error');
        }
    }
}

function stopCamera() {
    if (!cameraActive) return;

    // Stop camera stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    // Stop real-time detection
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }

    cameraActive = false;

    // Update UI
    document.getElementById('start-camera-btn').style.display = 'inline-block';
    document.getElementById('stop-camera-btn').style.display = 'none';
    document.getElementById('capture-btn').style.display = 'none';

    updateStatus('กล้องหยุดทำงาน', 'info');
}

function captureFrame() {
    if (!cameraActive || !cameraCanvas) return;

    const video = document.getElementById('camera-video');

    // Draw current frame to canvas
    cameraContext.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);

    // Convert canvas to image
    cameraCanvas.toBlob(async (blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });

        // Process the captured image
        await processImage(file);
    }, 'image/jpeg', 0.95);
}

async function startRealTimeDetection() {
    if (!cameraActive || !detector.modelLoaded) return;

    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');

    // Draw video frame to canvas
    cameraContext.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get current time
    const currentTime = Date.now();

    // Only run detection every 500ms to avoid overloading
    if (currentTime - lastDetectionTime > 500) {
        lastDetectionTime = currentTime;

        try {
            // Convert canvas to image for detection
            const imageData = cameraContext.getImageData(0, 0, canvas.width, canvas.height);

            // Create image element from canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = canvas.width;
            tempCanvas.height = canvas.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imageData, 0, 0);

            // Create image from canvas
            const img = new Image();
            img.onload = async () => {
                try {
                    const result = await detector.detect(img);

                    // Draw detections on video canvas
                    drawDetectionsOnVideo(result.detections);

                    // Update stats
                    updateCameraStats(result);

                } catch (error) {
                    console.error('Real-time detection error:', error);
                }
            };
            img.src = tempCanvas.toDataURL('image/jpeg', 0.8);

        } catch (error) {
            console.error('Real-time detection error:', error);
        }
    }

    // Continue loop
    animationFrame = requestAnimationFrame(startRealTimeDetection);
}

function drawDetectionsOnVideo(detections) {
    if (!cameraActive || !cameraCanvas) return;

    const video = document.getElementById('camera-video');

    // Clear canvas and draw video frame
    cameraContext.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);

    // Draw detections
    detections.forEach((detection, index) => {
        const [x, y, w, h] = detection.bbox;

        // Convert from normalized coordinates to pixel coordinates
        const canvasWidth = cameraCanvas.width;
        const canvasHeight = cameraCanvas.height;

        const boxX = (x - w/2) * canvasWidth;
        const boxY = (y - h/2) * canvasHeight;
        const boxWidth = w * canvasWidth;
        const boxHeight = h * canvasHeight;

        // Draw bounding box
        cameraContext.strokeStyle = '#FF6B6B';
        cameraContext.lineWidth = 3;
        cameraContext.strokeRect(boxX, boxY, boxWidth, boxHeight);

        // Draw label background
        cameraContext.fillStyle = 'rgba(255, 107, 107, 0.8)';
        const textHeight = 20;
        const textPadding = 6;
        const text = `${detection.className} ${(detection.confidence * 100).toFixed(1)}%`;

        const textWidth = cameraContext.measureText(text).width;
        cameraContext.fillRect(boxX, boxY - textHeight - textPadding, textWidth + textPadding * 2, textHeight + textPadding);

        // Draw text
        cameraContext.fillStyle = 'white';
        cameraContext.font = '14px Arial';
        cameraContext.fillText(text, boxX + textPadding, boxY - textPadding);
    });
}

function updateCameraStats(result) {
    const objectsCount = result.detections.length;
    const avgConfidence = objectsCount > 0
        ? (result.detections.reduce((sum, det) => sum + det.confidence, 0) / objectsCount * 100).toFixed(1)
        : 0;

    document.getElementById('objects-count').textContent = objectsCount;
    document.getElementById('avg-confidence').textContent = avgConfidence + '%';
    document.getElementById('processing-time').textContent = result.processingTime + 'ms';
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', init);
