# 🎯 YOLO Object Detection - GitHub Pages

เว็บแอปพลิเคชันตรวจจับวัตถุด้วย AI ที่ทำงานในเบราว์เซอร์โดยใช้ YOLO และ ONNX.js

## ✨ คุณสมบัติ

- 🚀 **ทำงานในเบราว์เซอร์** - ไม่ต้องติดตั้งอะไรเพิ่มเติม
- 🎯 **AI ตรวจจับวัตถุ** - ใช้โมเดล YOLO ที่แปลงเป็น ONNX
- 📸 **รองรับหลายรูปแบบ** - JPG, PNG, WebP และรูปภาพอื่นๆ
- 📹 **กล้องแบบเรียลไทม์** - ตรวจจับวัตถุจากกล้องแบบสด
- 📱 **รองรับทุกอุปกรณ์** - คอมพิวเตอร์และมือถือ
- ⚡ **ประสิทธิภาพสูง** - ประมวลผลในเครื่อง ไม่ส่งข้อมูลไปเซิร์ฟเวอร์
- 🎨 **UI สวยงาม** - ออกแบบมาอย่างเป็นมิตรต่อผู้ใช้
- 🔧 **ปรับแต่งได้** - ปรับความมั่นใจและ IoU threshold

## 🌐 สาธิต

เมื่อ deploy เสร็จแล้ว เว็บไซต์จะอยู่ที่: `https://panwan1040.github.io/`

## 🚀 การใช้งาน

### วิธีที่ 1: อัปโหลดรูปภาพ
1. เลือกโหมด "อัปโหลดรูปภาพ" (ค่าเริ่มต้น)
2. คลิกปุ่ม "เลือกไฟล์รูปภาพ" หรือลากไฟล์รูปภาพมาวาง
3. รอให้โมเดล AI ประมวลผล
4. ดูผลลัพธ์พร้อม bounding boxes และความมั่นใจ

### วิธีที่ 2: ใช้กล้องแบบเรียลไทม์
1. เลือกโหมด "เปิดกล้อง"
2. คลิกปุ่ม "เริ่มกล้อง" และอนุญาตการเข้าถึงกล้อง
3. AI จะตรวจจับวัตถุแบบเรียลไทม์อัตโนมัติ
4. คลิก "ถ่ายภาพ" เพื่อบันทึกภาพปัจจุบัน
5. คลิก "หยุดกล้อง" เพื่อปิดกล้อง

### วิธีที่ 3: ปรับแต่งการตั้งค่า
- **ความมั่นใจ** - ปรับระดับความมั่นใจในการตรวจจับ (0.1 - 1.0)
- **IoU Threshold** - ปรับค่า Intersection over Union สำหรับ NMS

## 🛠️ เทคโนโลยีที่ใช้

- **YOLOv8** - โมเดล AI สำหรับ object detection
- **ONNX.js** - รันโมเดลในเบราว์เซอร์
- **HTML5 Canvas** - วาดผลลัพธ์
- **CSS3** - ออกแบบ UI
- **JavaScript (ES6+)** - ตรรกะการทำงาน

## 📁 โครงสร้างโปรเจค

```
├── index.html          # หน้าเว็บหลัก
├── styles.css          # สไตล์ CSS
├── script.js           # JavaScript หลัก
├── model.onnx          # โมเดล YOLO ที่แปลงแล้ว
├── .nojekyll          # บอก GitHub Pages ว่าเป็น static site
├── .github/
│   └── workflows/
│       └── deploy.yml  # GitHub Actions สำหรับ deployment
└── README.md           # ไฟล์นี้
```

## 🔧 การติดตั้งและรันในเครื่อง

### วิธีที่ 1: ใช้ Python HTTP Server
```bash
python -m http.server 8000
```
แล้วเปิดเบราว์เซอร์ไปที่ `http://localhost:8000`

### วิธีที่ 2: ใช้ Node.js HTTP Server
```bash
npx http-server
```
แล้วเปิดเบราว์เซอร์ไปที่ `http://localhost:8080`

## 🚀 Deployment ไปยัง GitHub Pages

### ขั้นตอนอัตโนมัติ (แนะนำ)
1. Fork หรือ clone repository นี้
2. ไปที่ Settings > Pages ใน GitHub repository
3. เลือก source เป็น "GitHub Actions"
4. Push การเปลี่ยนแปลงใดๆ ไปยัง branch main
5. GitHub Actions จะ deploy อัตโนมัติ

### ขั้นตอนด้วยตนเอง
1. ไปที่ Settings > Pages ใน GitHub repository
2. เลือก source เป็น "Deploy from a branch"
3. เลือก branch main และโฟลเดอร์ /
4. Save และรอให้ deploy เสร็จ

## 🤝 การมีส่วนร่วม

ยินดีรับ pull requests! กรุณา:

1. Fork repository นี้
2. สร้าง feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add some AmazingFeature'`)
4. Push ไปยัง branch (`git push origin feature/AmazingFeature`)
5. เปิด Pull Request

## 📄 ใบอนุญาต

โปรเจคนี้ใช้ใบอนุญาต MIT - ดูรายละเอียดได้ที่ [LICENSE](LICENSE) file

## 🙏 เครดิต

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - โมเดล AI
- [ONNX.js](https://github.com/microsoft/onnxjs) - Runtime สำหรับโมเดลในเบราว์เซอร์
- [GitHub Pages](https://pages.github.com/) - Hosting แบบฟรี

## 📞 ติดต่อ

หากมีคำถามหรือพบปัญหา สามารถเปิด [Issue](https://github.com/[username]/[repository-name]/issues) ได้

---

⭐ ถ้าชอบโปรเจคนี้ กรุณาให้ดาว (star)!
