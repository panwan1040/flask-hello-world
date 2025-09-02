#!/usr/bin/env python3
"""
YOLO Object Detection Web App
รองรับทั้ง Camera และ File Upload
ใช้โมเดล: runs/detect/train7/weights/best.pt
"""

from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import base64
from io import BytesIO
from PIL import Image

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

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Object Detection - Camera</title>
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
        .video-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        #video-stream {
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            max-width: 100%;
            height: auto;
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
        button:active {
            transform: translateY(0);
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 YOLO Object Detection</h1>

        <div class="video-container">
            <!-- Camera Mode -->
            <div id="camera-mode" style="display: block;">
                <img id="video-stream" src="/video_feed" alt="Video Stream">
            </div>

            <!-- Upload Mode -->
            <div id="upload-mode" style="display: none;">
                <div id="upload-result" style="display: none;">
                    <img id="result-image" src="" alt="Detection Result" style="max-width: 100%; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
                </div>
                <div id="upload-placeholder" style="text-align: center; padding: 50px; background: rgba(255,255,255,0.1); border-radius: 15px; border: 2px dashed rgba(255,255,255,0.3);">
                    <div style="font-size: 3em; margin-bottom: 20px;">📸</div>
                    <div style="font-size: 1.2em; margin-bottom: 20px;">ลากไฟล์รูปภาพมาที่นี่ หรือคลิกเพื่อเลือกไฟล์</div>
                    <input type="file" id="file-input" accept="image/*" style="display: none;">
                    <button onclick="document.getElementById('file-input').click()" style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); border: none; padding: 15px 30px; border-radius: 25px; color: white; font-size: 16px; font-weight: bold; cursor: pointer;">เลือกไฟล์รูปภาพ</button>
                </div>
            </div>
        </div>

        <div class="controls">
            <button onclick="switchMode('camera')">📹 โหมดกล้อง</button>
            <button onclick="switchMode('upload')">📸 อัปโหลดรูปภาพ</button>
            <button onclick="startDetection()">▶️ เริ่มการตรวจจับ</button>
            <button onclick="stopDetection()">⏹️ หยุดการตรวจจับ</button>
            <button onclick="toggleFullscreen()">🔍 เต็มจอ</button>
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

        <div class="stats">
            <div class="stat-card">
                <div class="stat-number" id="fps">0</div>
                <div class="stat-label">FPS</div>
            </div>
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

    <script>
        let detectionActive = false;
        let fpsCounter = 0;
        let fpsTimer = Date.now();
        let currentMode = 'camera';

        function switchMode(mode) {
            currentMode = mode;
            const cameraMode = document.getElementById('camera-mode');
            const uploadMode = document.getElementById('upload-mode');
            const uploadResult = document.getElementById('upload-result');
            const uploadPlaceholder = document.getElementById('upload-placeholder');

            if (mode === 'camera') {
                cameraMode.style.display = 'block';
                uploadMode.style.display = 'none';
                document.getElementById('status').innerHTML = '📹 โหมดกล้อง - กำลังโหลดวิดีโอ...';
            } else {
                cameraMode.style.display = 'none';
                uploadMode.style.display = 'block';
                uploadResult.style.display = 'none';
                uploadPlaceholder.style.display = 'block';
                document.getElementById('status').innerHTML = '📸 โหมดอัปโหลด - เลือกรูปภาพเพื่อตรวจจับวัตถุ';
                document.getElementById('status').style.background = 'rgba(76, 237, 196, 0.2)';
            }
        }

        function startDetection() {
            detectionActive = true;
            if (currentMode === 'camera') {
                document.getElementById('status').innerHTML = '🔍 กำลังตรวจจับวัตถุ...';
            }
            document.getElementById('status').style.background = 'rgba(76, 237, 196, 0.2)';
        }

        function stopDetection() {
            detectionActive = false;
            document.getElementById('status').innerHTML = '⏸️ หยุดการตรวจจับ';
            document.getElementById('status').style.background = 'rgba(255, 107, 107, 0.2)';
        }

        function toggleFullscreen() {
            const video = document.getElementById('video-stream');
            if (!document.fullscreenElement) {
                video.requestFullscreen().catch(err => {
                    console.log('Error attempting to enable fullscreen:', err);
                });
            } else {
                document.exitFullscreen();
            }
        }

        // Update stats every second
        setInterval(() => {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('objects').textContent = data.objects;
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                })
                .catch(err => console.log('Error fetching stats:', err));
        }, 1000);

        // Check model status
        setInterval(() => {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.model_loaded) {
                        document.getElementById('status').innerHTML = '✅ โมเดลพร้อมใช้งาน';
                        document.getElementById('status').style.background = 'rgba(76, 237, 196, 0.2)';
                    }
                });
        }, 2000);

        // Load initial confidence value
        fetch('/get_confidence')
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
            fetch(`/set_confidence/${value}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Confidence updated:', data.confidence);
                })
                .catch(err => console.log('Error updating confidence:', err));
        });

        // Handle file upload
        document.getElementById('file-input').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                uploadAndDetect(file);
            }
        });

        // Handle drag and drop
        const uploadPlaceholder = document.getElementById('upload-placeholder');
        uploadPlaceholder.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.background = 'rgba(76, 237, 196, 0.2)';
        });

        uploadPlaceholder.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.background = 'rgba(255, 255, 255, 0.1)';
        });

        uploadPlaceholder.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.background = 'rgba(255, 255, 255, 0.1)';

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
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

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // แสดงผลลัพธ์
                    document.getElementById('upload-placeholder').style.display = 'none';
                    document.getElementById('upload-result').style.display = 'block';
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
    </script>
</body>
</html>
"""

def allowed_file(filename):
    """ตรวจสอบนามสกุลไฟล์ที่อนุญาต"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """โหลดโมเดล YOLO"""
    global model
    try:
        # ลองหลายๆ path สำหรับ online deployment
        possible_paths = [
            r"D:\code project\Part-Counter.v1i.yolov8\runs\detect\train7\weights\best.pt",
            "runs/detect/train7/weights/best.pt",
            "./runs/detect/train7/weights/best.pt",
            "best.pt"
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
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ ไม่สามารถเปิดกล้องได้")
            return False

        # ตั้งค่าคุณภาพกล้อง
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        print("✅ เปิดกล้องสำเร็จ!")
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการเปิดกล้อง: {e}")
        return False

def generate_frames():
    """สร้างเฟรมวิดีโอสำหรับ streaming"""
    global output_frame, lock, model

    fps_counter = 0
    fps_timer = time.time()
    objects_detected = 0
    avg_confidence = 0.0

    while True:
        if camera is None:
            break

        success, frame = camera.read()
        if not success:
            break

        # ทำการตรวจจับวัตถุด้วย YOLO (confidence threshold 95%)
        if model is not None:
            try:
                results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

                # วาดผลลัพธ์
                annotated_frame = results[0].plot()

                # นับจำนวนวัตถุและคำนวณความมั่นใจเฉลี่ย
                if len(results[0].boxes) > 0:
                    objects_detected = len(results[0].boxes)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    avg_confidence = float(np.mean(confidences))
                else:
                    objects_detected = 0
                    avg_confidence = 0.0

                # คำนวณ FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_timer >= 1.0:
                    fps = fps_counter / (current_time - fps_timer)
                    fps_counter = 0
                    fps_timer = current_time

                    # เก็บข้อมูลสถิติ
                    with lock:
                        app.config['current_fps'] = round(fps, 1)
                        app.config['current_objects'] = objects_detected
                        app.config['current_confidence'] = avg_confidence

                frame = annotated_frame

            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการตรวจจับ: {e}")

        # แปลงเป็น JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # ส่งเฟรม
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """หน้าแรก"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """ส่งสถานะโมเดล"""
    return {
        'model_loaded': model is not None,
        'camera_active': camera is not None and camera.isOpened()
    }

@app.route('/stats')
def get_stats():
    """ส่งสถิติการตรวจจับ"""
    return {
        'fps': app.config.get('current_fps', 0),
        'objects': app.config.get('current_objects', 0),
        'confidence': app.config.get('current_confidence', 0.0)
    }

@app.route('/start_detection')
def start_detection():
    """เริ่มการตรวจจับ"""
    return {'status': 'started'}

@app.route('/stop_detection')
def stop_detection():
    """หยุดการตรวจจับ"""
    return {'status': 'stopped'}

@app.route('/set_confidence/<float:confidence>')
def set_confidence(confidence):
    """ตั้งค่า confidence threshold"""
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = max(0.1, min(1.0, confidence))  # จำกัดค่าให้อยู่ระหว่าง 0.1-1.0
    return {'status': 'success', 'confidence': CONFIDENCE_THRESHOLD}

@app.route('/get_confidence')
def get_confidence():
    """ส่งค่า confidence threshold ปัจจุบัน"""
    return {'confidence': CONFIDENCE_THRESHOLD}

@app.route('/upload', methods=['POST'])
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

def cleanup():
    """ทำความสะอาดทรัพยากร"""
    global camera
    try:
        if camera is not None:
            camera.release()
            print("🔄 ปิดกล้องแล้ว")
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการปิดกล้อง: {e}")

if __name__ == '__main__':
    try:
        print("🚀 กำลังเริ่ม YOLO Object Detection Web App...")
        print("=" * 50)

        # โหลดโมเดล
        if not load_model():
            print("❌ ไม่สามารถโหลดโมเดลได้ โปรแกรมจะหยุดทำงาน")
            exit(1)

        # พยายามเปิดกล้อง (ไม่บังคับสำหรับ online deployment)
        camera_available = initialize_camera()
        if camera_available:
            print("✅ กล้องพร้อมใช้งาน")
        else:
            print("⚠️ กล้องไม่พร้อมใช้งาน (ปกติสำหรับ online deployment)")

        # ตั้งค่า Flask
        app.config['current_fps'] = 0
        app.config['current_objects'] = 0
        app.config['current_confidence'] = 0.0

        print("✅ ทุกอย่างพร้อมแล้ว!")
        print("🌐 เปิดเบราว์เซอร์ไปที่: http://localhost:5000")
        print("📸 รองรับทั้งโหมดกล้องและอัปโหลดรูปภาพ")
        if camera_available:
            print("📹 โหมดกล้อง: ตรวจจับวัตถุแบบเรียลไทม์")
        print("💡 กด Ctrl+C เพื่อหยุดโปรแกรม")
        print("=" * 50)

        # เริ่มเซิร์ฟเวอร์
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\n👋 กำลังปิดโปรแกรม...")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
    finally:
        cleanup()
