# Face Detection & Recognition – Offline Android Demo

![Android](https://img.shields.io/badge/Android-5.0%2B-brightgreen)
![Kotlin](https://img.shields.io/badge/Kotlin-1.9+-blue)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-2.14+-orange)
![ML Kit](https://img.shields.io/badge/ML%20Kit-Face%20Detection-FF6F00)

A fully **offline, on‑device** face recognition system for Android.  
No internet connection required. Embeddings are stored **persistently** in local storage.

---

## ✨ Features

- **Real‑time face detection** with green bounding box overlay
- **Face recognition** using FaceNet (128‑dim L2‑normalised embeddings)
- **Two operating modes**:
    - `RECOGNIZE` – continuously identifies known faces
    - `REGISTER` – one‑tap capture to enroll a new face
- **Persistent storage** – embeddings saved via `SharedPreferences` (JSON), survive app restarts
- **Comprehensive performance logging** – view in Logcat:
    - Device model, CPU cores, RAM, heap
    - Camera sensor resolution & rotated bitmap size
    - Face detection time, embedding time, total frame time
    - **Live FPS** (frames per second)
    - Memory usage (heap & native)
    - CPU load

---

## 📦 Requirements

- **Android 5.0+** (API level 21)
- **Camera** (front camera used by default)
- **TensorFlow Lite model**: `facenet.tflite` (must be placed in `app/src/main/assets/`)

---

## 🚀 Installation

### Option 1 – Build & Run with Android Studio

1. Clone this repository or copy the source files.
2. Open the project in **Android Studio**.
3. Connect your device via USB (enable **Developer options** and **USB debugging**).
4. Click **Run** (▶) or press `Shift+F10`.

### Option 2 – Install APK directly

1. Generate a signed or debug APK:
    - **Debug**: `Build → Build Bundle(s) / APK → Build APK`
    - **Release**: `Build → Generate Signed Bundle / APK`
2. Transfer the APK to your device and install it.

---

## 🎮 Usage

1. **Launch the app** – camera preview appears, green box follows your face.
2. **Recognition mode** (default):
    - If a face is already enrolled, the **name and similarity score** appear at the top.
    - Similarity **> 0.5** = known person; **< 0.5** = `"Unknown"`.
3. **Register a new face**:
    - Tap **`REGISTER FACE`**
    - Enter the person’s name
    - Look at the camera – **one frame is captured**, the face is cropped, embedded, and saved.
    - A toast confirms success.
4. **Switch back to recognition**:
    - Tap **`RECOGNIZE`** at any time.

> 💾 **Persistence**: Enrolled faces remain stored even after closing the app.  
> To clear all data: go to **Settings → Apps → FaceDetectionApp → Storage → Clear Storage**.

---

## 📊 Performance & Benchmarks

All measurements taken with **FaceNet (Inception ResNet v1)** on **1280×720 → 960×1280 rotated** input.  
*Emulator numbers are for reference only – real devices are significantly faster.*

| Device                          | Android | CPU                          | Face Detection | Embedding | Total  | FPS   |
|---------------------------------|--------|------------------------------|----------------|-----------|--------|-------|
| Pixel 5 API 36.0 Emulator (x86) | 16     | 4× emulated                 | 190 ms        | 170 ms    | 360 ms | 2.8   |


*\*With quantised MobileFaceNet model – **recommended for embedded deployment**.*

### 🔬 Example Log Output (Emulator)
D/SystemInfo: Device Model: sdk_gphone64_x86_64
D/SystemInfo: CPU Cores: 4
D/SystemInfo: Total RAM: 1971 MB
D/FaceRecognitionAnalyzer: 📸 Detected 1 face(s) in 195.29 ms
D/FaceRecognitionAnalyzer: Cropped: 160x160, Embedding time: 136.01 ms
D/FaceEmbeddingDatabase: 🔍 Compare with 'known_faces' → similarity = 0.6786
D/FaceRecognitionAnalyzer: 🏷️ Recognized as: 'known_face' (score: 0.679)
D/FaceRecognitionAnalyzer: ⏱️ Total frame processing time: 350.52 ms
D/FaceRecognitionAnalyzer: 📊 FPS: 2.3

---

## 🧱 Project Architecture

| Module                    | Responsibility                                           |
|---------------------------|----------------------------------------------------------|
| `MainActivity`            | UI, camera lifecycle, mode switching, one‑shot enrollment |
| `FaceRecognitionAnalyzer` | CameraX `ImageAnalysis.Analyzer` – face detection, embedding extraction, performance logging |
| `FaceNetModel`            | Loads TFLite model, preprocesses image, L2 normalisation |
| `FaceEmbeddingDatabase`   | In‑memory + persistent storage of embeddings           |
| `EmbeddingStorage`        | JSON serialisation via `SharedPreferences`              |
| `FaceContourView`         | Custom view to draw bounding boxes (manual scale + mirror) |
| `SystemInfo`              | Hardware & performance logging (CPU, RAM, heap, freq)   |

---

## 🛠️ Technical Details

- **Face Detection**: [ML Kit Face Detection](https://developers.google.com/ml-kit/vision/face-detection) – `ACCURATE` mode, `CONTOUR_MODE_ALL`
- **Face Recognition**: TensorFlow Lite – FaceNet (128‑dim embeddings)
- **Preprocessing**:
    - Bitmap rotation to portrait using `rotationDegrees` from `ImageProxy`
    - Crop with **30% margin**, then **center square** → resize to 160×160
    - Normalisation: `(pixel - 127.5f) / 128.0f` (identical to Python)
- **Similarity**: Cosine similarity = dot product of L2‑normalised embeddings
- **Persistence**: Embeddings stored as JSON in `SharedPreferences` (≈ 2 KB per face)

---

