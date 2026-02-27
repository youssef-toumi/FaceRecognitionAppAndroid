package com.example.facedetectionapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import java.io.FileNotFoundException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var resultTextView: TextView
    private lateinit var btnRegister: Button
    private lateinit var btnRecognize: Button
    private lateinit var btnResetAll: Button
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraProvider: ProcessCameraProvider
    private lateinit var cameraSelector: CameraSelector

    private lateinit var faceNetModel: FaceNetModel
    private var livenessModel: LivenessModel? = null
    private lateinit var embeddingDatabase: FaceEmbeddingDatabase
    private var faceRecognitionAnalyzer: FaceRecognitionAnalyzer? = null

    // Stereo depth anti-spoofing (null if calibration file is missing)
    private var stereoCalibData: StereoCalibrationData? = null
    private var depthAntiSpoofing: DepthAntiSpoofing? = null

    private val faceDetector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .build()
    )

    companion object {
        private const val REQUEST_CAMERA_PERMISSION = 1001
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        resultTextView = findViewById(R.id.resultTextView)
        btnRegister = findViewById(R.id.btnRegister)
        btnRecognize = findViewById(R.id.btnRecognize)
        btnResetAll = findViewById(R.id.btnResetAll)

        btnRegister.setOnClickListener { showNameInputDialog() }
        btnRecognize.setOnClickListener { switchToRecognitionMode() }
        btnResetAll.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Reset Database")
                .setMessage("Delete ALL stored face embeddings?\nThis cannot be undone.")
                .setPositiveButton("Yes") { _, _ ->
                    embeddingDatabase.clear()
                    resultTextView.text = "All faces deleted"
                    Toast.makeText(this, "✅ Database cleared", Toast.LENGTH_SHORT).show()
                    Log.d(TAG, "All embeddings cleared by user")
                }
                .setNegativeButton("No", null)
                .show()
        }

        SystemInfo.logDeviceInfo(this)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initModels()
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CAMERA_PERMISSION
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun initModels() {
        faceNetModel = FaceNetModel(this)
        livenessModel = try {
            LivenessModel(this)
        } catch (e: FileNotFoundException) {
            Log.w(TAG, "Liveness model not found (liveness_detection_model.tflite missing). Liveness disabled.")
            null
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load liveness model", e)
            null
        }
        embeddingDatabase = FaceEmbeddingDatabase(this)

        // Initialize stereo depth anti-spoofing (graceful if calibration missing)
        initStereoDepth()
    }

    private fun initStereoDepth() {
        try {
            // Load OpenCV native library
            org.opencv.android.OpenCVLoader.initLocal()
            Log.d(TAG, "OpenCV loaded")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load OpenCV", e)
            return
        }

        stereoCalibData = StereoCalibrationData.loadFromAssets(this)
        if (stereoCalibData == null) {
            Log.w(TAG, "Stereo calibration not found → depth anti-spoofing disabled")
            Log.w(TAG, "To enable: run tools/stereo_calibrate.py, copy stereo_calib.json to assets/")
            return
        }

        val stereoCapture = StereoCapture(this)
        val rectifier = StereoRectifier(stereoCalibData!!)
        val depthProcessor = StereoDepthProcessor(stereoCalibData!!)
        depthAntiSpoofing = DepthAntiSpoofing(stereoCapture, rectifier, depthProcessor)
        Log.d(TAG, "✅ Stereo depth anti-spoofing initialized")
    }

    /**
     * Perform a stereo depth liveness check on a detected face.
     * Call this from a coroutine when you want extra anti-spoofing.
     *
     * This pauses CameraX, captures a stereo pair, processes depth,
     * then resumes CameraX.
     */
    private suspend fun performDepthLivenessCheck(
        faceBoundingBox: android.graphics.Rect
    ): DepthAntiSpoofing.DepthLivenessResult? {
        val detector = depthAntiSpoofing ?: return null

        // Must pause CameraX for StereoCapture (exclusive camera access)
        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
            cameraProvider.unbindAll()
        }

        return try {
            detector.checkLiveness(faceBoundingBox)
        } finally {
            // Always resume CameraX
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                switchToRecognitionMode()
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            switchToRecognitionMode()

            previewView.post {
                Log.d(TAG, "📱 PreviewView resolution: ${previewView.width}x${previewView.height}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun switchToRecognitionMode() {
        resultTextView.text = "Mode: Recognition"
        bindAnalyzer(createRecognitionAnalyzer())
    }

    private fun createRecognitionAnalyzer(): FaceRecognitionAnalyzer {
        return FaceRecognitionAnalyzer(
            faceNetModel = faceNetModel,
            livenessModel = livenessModel,
            embeddingDatabase = embeddingDatabase,
            onRecognitionResults = { results ->
                if (results.isNotEmpty()) {
                    val best = results.maxByOrNull { it.similarity }
                    val name = best?.name ?: "Unknown"
                    val score = best?.similarity ?: 0f
                    val isReal = best?.isReal ?: true
                    val livenessProb = best?.livenessProbability ?: -1f

                    // Invert probability to show "real confidence" (100% - probability)
                    val confidence = if (livenessProb >= 0) (1 - livenessProb) * 100 else -1f

                    val text = if (confidence >= 0) {
                        if (isReal) {
                            "$name (${"%.2f".format(score)}) - Confidence: ${"%.0f".format(confidence)}%"
                        } else {
                            "Fake - Confidence: ${"%.0f".format(confidence)}%"
                        }
                    } else {
                        "$name (${"%.2f".format(score)})"
                    }

                    val spannable = SpannableString(text)
                    val color = if (isReal) Color.GREEN else Color.RED
                    spannable.setSpan(ForegroundColorSpan(color), 0, text.length, 0)

                    resultTextView.text = spannable
                } else {
                    resultTextView.text = "No face"
                }
            },
            similarityThreshold = 0.5f,
            livenessThreshold = 0.5f,
            processingInterval = 4
        )
    }

    private fun bindAnalyzer(analyzer: ImageAnalysis.Analyzer) {
        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(1280, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        imageAnalysis.setAnalyzer(cameraExecutor, analyzer)

        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)

        if (analyzer is FaceRecognitionAnalyzer) {
            faceRecognitionAnalyzer = analyzer
        }
    }

    private fun showNameInputDialog() {
        val input = android.widget.EditText(this)
        AlertDialog.Builder(this)
            .setTitle("Register Face")
            .setMessage("Enter the person's name:")
            .setView(input)
            .setPositiveButton("OK") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isNotEmpty()) {
                    captureSingleFrameForEnrollment(name)
                } else {
                    Toast.makeText(this, "Name cannot be empty", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun captureSingleFrameForEnrollment(name: String) {
        resultTextView.text = "Capturing face for '$name'..."
        Toast.makeText(this, "Look at the camera", Toast.LENGTH_SHORT).show()

        val enrollmentAnalyzer = ImageAnalysis.Analyzer { imageProxy ->
            processFrameForEnrollment(imageProxy, name)
        }

        val imageAnalysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(1280, 720))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        imageAnalysis.setAnalyzer(cameraExecutor, enrollmentAnalyzer)

        val preview = Preview.Builder().build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
    }

    private fun processFrameForEnrollment(imageProxy: ImageProxy, name: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bitmap = imageProxy.toBitmap()
                if (bitmap == null) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Failed to capture image", Toast.LENGTH_SHORT).show()
                        switchToRecognitionMode()
                    }
                    return@launch
                }

                val rotation = imageProxy.imageInfo.rotationDegrees
                val rotatedBitmap = rotateBitmap(bitmap, rotation)

                val image = imageProxy.image ?: return@launch
                val inputImage = InputImage.fromMediaImage(image, rotation)
                val faces = faceDetector.process(inputImage).await()

                if (faces.isEmpty()) {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "No face detected", Toast.LENGTH_SHORT).show()
                        switchToRecognitionMode()
                    }
                    return@launch
                }

                val faceCrop = faceNetModel.cropFace(rotatedBitmap, faces[0].boundingBox)
                val embedding = faceNetModel.getFaceEmbedding(faceCrop)

                embeddingDatabase.addEmbedding(name, embedding)

                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "✅ Registered $name", Toast.LENGTH_SHORT).show()
                    switchToRecognitionMode()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Enrollment failed", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Enrollment failed", Toast.LENGTH_SHORT).show()
                    switchToRecognitionMode()
                }
            } finally {
                imageProxy.close()
            }
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        val yuvImage = android.graphics.YuvImage(bytes, android.graphics.ImageFormat.NV21, width, height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initModels()
                startCamera()
            } else {
                Toast.makeText(this, "❌ Camera permission required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceRecognitionAnalyzer?.close()
        faceNetModel.close()
        livenessModel?.close()
        faceDetector.close()
        stereoCalibData?.release()
    }
}