from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import sys
import base64
from io import BytesIO
from PIL import Image

# เพิ่ม path สำหรับโมเดล
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Global variables
model = None
camera = None
output_frame = None
lock = threading.Lock()

# Detection settings
CONFIDENCE_THRESHOLD = 0.95  # 95% confidence threshold
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# สร้างโฟลเดอร์ upload ถ้ายังไม่มี
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_model():
    """โหลดโมเดล YOLO"""
    global model
    try:
        # ลองหลายๆ path สำหรับ online deployment
        possible_paths = [
            "best.pt",
            "./best.pt",
            os.path.join(os.path.dirname(__file__), '..', 'best.pt'),
            os.path.join(os.path.dirname(__file__), 'best.pt')
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print("❌ ไม่พบไฟล์โมเดล YOLO")
            return False

        print(f"กำลังโหลดโมเดลจาก: {model_path}")
        model = YOLO(model_path)
        print("✅ โหลดโมเดลสำเร็จ!")
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return False

def initialize_camera():
    """เริ่มต้นกล้อง"""
    global camera
    try:
        # สำหรับ Vercel/Serverless ไม่มีกล้องจริง ใช้ dummy camera
        print("⚠️ Serverless environment - ไม่มีกล้องจริง")
        return False
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเปิดกล้อง: {e}")
        return False

def allowed_file(filename):
    """ตรวจสอบนามสกุลไฟล์ที่อนุญาต"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# โหลดโมเดลเมื่อเริ่มต้น
load_model()
initialize_camera()

# HTML Template สำหรับ Web Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Object Detection - Web API</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .upload-section {
            text-align: center;
            padding: 50px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            border: 2px dashed rgba(255,255,255,0.3);
            margin: 20px 0;
        }
        .upload-section:hover {
            background: rgba(255,255,255,0.15);
        }
        .result-section {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        #result-image {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .settings {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .setting-group {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            min-width: 250px;
        }
        .setting-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #FFD93D;
        }
        .setting-group input[type="range"] {
            width: 100%;
            margin: 10px 0;
        }
        .setting-group .value-display {
            font-size: 1.2em;
            font-weight: bold;
            color: #4ECDC4;
            margin-top: 10px;
        }
        button {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .status {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            backdrop-filter: blur(5px);
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #FFD93D;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .api-info {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
        }
        .api-endpoint {
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 YOLO Object Detection Web API</h1>

        <div class="upload-section" id="upload-section">
            <div style="font-size: 3em; margin-bottom: 20px;">📸</div>
            <div style="font-size: 1.2em; margin-bottom: 20px;">ลากไฟล์รูปภาพมาที่นี่ หรือคลิกเพื่อเลือกไฟล์</div>
            <input type="file" id="file-input" accept="image/*" style="display: none;">
            <button onclick="document.getElementById('file-input').click()">เลือกไฟล์รูปภาพ</button>
        </div>

        <div class="result-section" id="result-section">
            <h3>ผลการตรวจจับวัตถุ</h3>
            <img id="result-image" src="" alt="Detection Result">
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number" id="objects">0</div>
                    <div class="stat-label">วัตถุที่ตรวจพบ</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="confidence">0%</div>
                    <div class="stat-label">ความมั่นใจเฉลี่ย</div>
                </div>
            </div>
        </div>

        <div class="controls">
            <button onclick="detectAgain()">🔄 ตรวจจับใหม่</button>
            <button onclick="clearResult()">🗑️ ล้างผลลัพธ์</button>
        </div>

        <div class="settings">
            <div class="setting-group">
                <label for="confidence-slider">🎯 ความมั่นใจในการตรวจจับ</label>
                <input type="range" id="confidence-slider" min="0.1" max="1.0" step="0.05" value="0.95">
                <div class="value-display" id="confidence-value">95%</div>
            </div>
        </div>

        <div class="status" id="status">
            กำลังโหลดโมเดล YOLO...
        </div>

        <div class="api-info">
            <h3>📋 API Endpoints</h3>
            <div class="api-endpoint">POST /api/upload - อัปโหลดและตรวจจับวัตถุ</div>
            <div class="api-endpoint">GET /api/status - สถานะโมเดลและระบบ</div>
            <div class="api-endpoint">GET /api/health - Health check</div>
            <div class="api-endpoint">GET /api/stats - สถิติการทำงาน</div>
            <div class="api-endpoint">POST /api/set_confidence/&lt;value&gt; - ตั้งค่าความมั่นใจ</div>
        </div>
    </div>

    <script>
        let currentFile = null;

        function detectAgain() {
            if (currentFile) {
                uploadAndDetect(currentFile);
            }
        }

        function clearResult() {
            document.getElementById('upload-section').style.display = 'block';
            document.getElementById('result-section').style.display = 'none';
            document.getElementById('file-input').value = '';
            currentFile = null;
        }

        // Handle file input
        document.getElementById('file-input').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                currentFile = file;
                uploadAndDetect(file);
            }
        });

        // Handle drag and drop
        const uploadSection = document.getElementById('upload-section');
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.background = 'rgba(76, 237, 196, 0.2)';
        });

        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.background = 'rgba(255, 255, 255, 0.1)';
        });

        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.background = 'rgba(255, 255, 255, 0.1)';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    currentFile = file;
                    uploadAndDetect(file);
                } else {
                    alert('กรุณาเลือกไฟล์รูปภาพเท่านั้น');
                }
            }
        });

        function uploadAndDetect(file) {
            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('status').innerHTML = '⏳ กำลังอัปโหลดและตรวจจับวัตถุ...';
            document.getElementById('status').style.background = 'rgba(255, 193, 7, 0.2)';

            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // แสดงผลลัพธ์
                    document.getElementById('upload-section').style.display = 'none';
                    document.getElementById('result-section').style.display = 'block';
                    document.getElementById('result-image').src = 'data:image/jpeg;base64,' + data.image;

                    // อัปเดตสถิติ
                    document.getElementById('objects').textContent = data.objects;
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';

                    document.getElementById('status').innerHTML = '✅ ตรวจจับวัตถุสำเร็จ!';
                    document.getElementById('status').style.background = 'rgba(76, 237, 196, 0.2)';
                } else {
                    throw new Error(data.error);
                }
            })
            .catch(error => {
                console.error('Upload error:', error);
                document.getElementById('status').innerHTML = '❌ เกิดข้อผิดพลาด: ' + error.message;
                document.getElementById('status').style.background = 'rgba(255, 107, 107, 0.2)';
            });
        }

        // Check model status
        setInterval(() => {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.model_loaded) {
                        document.getElementById('status').innerHTML = '✅ โมเดลพร้อมใช้งาน';
                        document.getElementById('status').style.background = 'rgba(76, 237, 196, 0.2)';
                    } else {
                        document.getElementById('status').innerHTML = '❌ โมเดลยังไม่ได้โหลด';
                        document.getElementById('status').style.background = 'rgba(255, 107, 107, 0.2)';
                    }
                });
        }, 2000);

        // Load initial confidence value
        fetch('/api/get_confidence')
            .then(response => response.json())
            .then(data => {
                const slider = document.getElementById('confidence-slider');
                const valueDisplay = document.getElementById('confidence-value');
                slider.value = data.confidence;
                valueDisplay.textContent = (data.confidence * 100).toFixed(1) + '%';
            });

        // Handle confidence slider change
        document.getElementById('confidence-slider').addEventListener('input', function() {
            const value = parseFloat(this.value);
            const valueDisplay = document.getElementById('confidence-value');
            valueDisplay.textContent = (value * 100).toFixed(1) + '%';

            // Send new confidence value to server
            fetch(`/api/set_confidence/${value}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Confidence updated:', data.confidence);
                })
                .catch(err => console.log('Error updating confidence:', err));
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """หน้าแรก - Web Interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def get_status():
    """ส่งสถานะโมเดล"""
    return jsonify({
        'model_loaded': model is not None,
        'camera_active': False,  # Serverless ไม่มีกล้อง
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })

@app.route('/api/stats')
def get_stats():
    """ส่งสถิติการตรวจจับ"""
    return jsonify({
        'fps': 0,  # Serverless ไม่มี real-time processing
        'objects': 0,
        'confidence': 0.0
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_ready': model is not None,
        'timestamp': time.time()
    })

@app.route('/api/set_confidence/<float:confidence>')
def set_confidence(confidence):
    """ตั้งค่า confidence threshold"""
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = max(0.1, min(1.0, confidence))  # จำกัดค่าให้อยู่ระหว่าง 0.1-1.0
    return jsonify({'status': 'success', 'confidence': CONFIDENCE_THRESHOLD})

@app.route('/api/get_confidence')
def get_confidence():
    """ส่งค่า confidence threshold ปัจจุบัน"""
    return jsonify({'confidence': CONFIDENCE_THRESHOLD})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """อัปโหลดไฟล์และทำการตรวจจับวัตถุ"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # อ่านไฟล์เป็น numpy array
            image = Image.open(file.stream)
            frame = np.array(image)

            # แปลงเป็น RGB ถ้าจำเป็น
            if frame.shape[-1] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            elif len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # ทำการตรวจจับวัตถุ
            if model is not None:
                results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                annotated_frame = results[0].plot()

                # สร้าง base64 string สำหรับส่งกลับ
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(buffer).decode('utf-8')

                # นับจำนวนวัตถุ
                objects_detected = len(results[0].boxes) if len(results[0].boxes) > 0 else 0
                avg_confidence = 0.0
                if objects_detected > 0:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    avg_confidence = float(np.mean(confidences))

                return jsonify({
                    'success': True,
                    'image': img_str,
                    'objects': objects_detected,
                    'confidence': avg_confidence
                })
            else:
                return jsonify({'error': 'Model not loaded'}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

# Vercel handler
def handler(event, context):
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)

    from werkzeug.serving import WSGIRequestHandler
    return app