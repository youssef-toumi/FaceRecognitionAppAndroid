package com.example.facedetectionapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.*
import kotlinx.coroutines.tasks.await
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicReference

@ExperimentalGetImage
class FaceRecognitionAnalyzer(
    private val faceNetModel: FaceNetModel,
    private val livenessModel: LivenessModel?,
    private val embeddingDatabase: FaceEmbeddingDatabase,
    private val onRecognitionResults: (List<FaceRecognitionResult>) -> Unit,
    private val similarityThreshold: Float = 0.5f,
    private val livenessThreshold: Float = 0.5f,
    private val processingInterval: Int = 4
) : ImageAnalysis.Analyzer {

    private val TAG = "FaceRecognitionAnalyzer"
    private val faceDetector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .build()
    )

    private var frameCounter = 0
    private val analyzerScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var lastFpsLogTime = System.currentTimeMillis()
    private var frameCount = 0

    // Store the latest liveness score (thread-safe)
    private val latestLivenessScore = AtomicReference(1f) // default 1 (fake, but safe)

    override fun analyze(imageProxy: ImageProxy) {
        frameCounter++
        if (frameCounter % processingInterval != 0) {
            imageProxy.close()
            return
        }

        analyzerScope.launch {
            val startTotal = System.nanoTime()

            val image = imageProxy.image ?: run {
                Log.e(TAG, "❌ ImageProxy.image is null")
                imageProxy.close()
                return@launch
            }
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            val inputImage = InputImage.fromMediaImage(image, rotationDegrees)

            val originalBitmap = imageProxy.toBitmap()
            if (originalBitmap == null) {
                Log.e(TAG, "❌ Failed to convert ImageProxy to Bitmap")
                imageProxy.close()
                return@launch
            }
            val rotatedBitmap = rotateBitmap(originalBitmap, rotationDegrees)
            Log.d(TAG, "✅ Rotated bitmap: ${rotatedBitmap.width}x${rotatedBitmap.height}, rotation: $rotationDegrees°")

            // Launch liveness inference asynchronously on the full bitmap
            val livenessJob = livenessModel?.let { model ->
                async {
                    val (score, _) = model.predict(rotatedBitmap)
                    latestLivenessScore.set(score)
                    score
                }
            }

            if (frameCounter % 30 == 0) {
                Log.d(TAG, "📷 Camera sensor resolution: ${imageProxy.width}x${imageProxy.height}")
                SystemInfo.logCpuFrequency()
            }

            try {
                val startDetect = System.nanoTime()
                val faces: List<Face> = faceDetector.process(inputImage).await()
                val detectTime = (System.nanoTime() - startDetect) / 1_000_000.0
                Log.d(TAG, "📸 Detected ${faces.size} face(s) in ${"%.2f".format(detectTime)} ms")

                if (faces.isNotEmpty()) {
                    Log.d(TAG, "   First face bounding box: ${faces[0].boundingBox}")
                }

                // Wait for liveness if it's still running
                livenessJob?.await()

                val results = mutableListOf<FaceRecognitionResult>()
                for ((index, face) in faces.withIndex()) {
                    Log.d(TAG, "🧬 Processing face #$index")

                    val startEmbed = System.nanoTime()
                    val faceCrop = faceNetModel.cropFace(rotatedBitmap, face.boundingBox)
                    val embedding = faceNetModel.getFaceEmbedding(faceCrop)
                    val embedTime = (System.nanoTime() - startEmbed) / 1_000_000.0
                    Log.d(TAG, "   Cropped: ${faceCrop.width}x${faceCrop.height}, Embedding time: ${"%.2f".format(embedTime)} ms")

                    val livenessScore = latestLivenessScore.get()
                    val isReal = livenessScore < livenessThreshold
                    val (name, similarity) = if (isReal) {
                        embeddingDatabase.recognize(embedding, similarityThreshold)
                    } else {
                        "Spoof" to 0f
                    }

                    Log.d(TAG, "   🏷️ Recognized as: '$name' (score: ${"%.3f".format(similarity)}), Liveness: ${"%.2f".format(livenessScore)} (real=$isReal)")

                    // Pass liveness probability to the result
                    results.add(FaceRecognitionResult(
                        boundingBox = face.boundingBox,
                        name = name,
                        similarity = similarity,
                        isReal = isReal,
                        livenessProbability = livenessScore
                    ))
                }

                withContext(Dispatchers.Main) {
                    onRecognitionResults(results)
                }

                val totalTime = (System.nanoTime() - startTotal) / 1_000_000.0
                Log.d(TAG, "⏱️ Total frame processing time: ${"%.2f".format(totalTime)} ms")

                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsLogTime >= 1000) {
                    val fps = frameCount * 1000.0 / (now - lastFpsLogTime)
                    Log.d(TAG, "📊 FPS: ${"%.1f".format(fps)}")
                    frameCount = 0
                    lastFpsLogTime = now
                }

                if (frameCounter % 30 == 0) {
                    SystemInfo.logMemoryUsage(TAG)
                    SystemInfo.logCpuLoad()
                }

            } catch (e: Exception) {
                Log.e(TAG, "❌ Face detection/recognition failed", e)
            } finally {
                imageProxy.close()
            }
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix()
        matrix.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    @ExperimentalGetImage
    @Suppress("EXTENSION_SHADOWED_BY_MEMBER")
    private fun ImageProxy.toBitmap(): Bitmap? {
        if (format != ImageFormat.YUV_420_888) {
            Log.w(TAG, "Unexpected image format: $format")
            return null
        }

        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        val yuvImage = YuvImage(bytes, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val jpegData = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)
    }

    fun close() {
        analyzerScope.cancel()
        faceDetector.close()
    }
}