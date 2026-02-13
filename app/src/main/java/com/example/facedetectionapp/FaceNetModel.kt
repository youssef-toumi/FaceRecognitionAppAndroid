package com.example.facedetectionapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.min
import kotlin.math.sqrt

class FaceNetModel(private val context: Context) {

    private val TAG = "FaceNetModel"
    private val interpreter: Interpreter
    private val inputImageWidth: Int
    private val inputImageHeight: Int
    val outputEmbeddingSize: Int

    init {
        try {
            val modelBuffer = loadModelFile("facenet.tflite")
            val options = Interpreter.Options().apply { setNumThreads(4) }
            interpreter = Interpreter(modelBuffer, options)

            val inputTensor = interpreter.getInputTensor(0)
            inputImageWidth = inputTensor.shape()[1]
            inputImageHeight = inputTensor.shape()[2]

            val outputTensor = interpreter.getOutputTensor(0)
            outputEmbeddingSize = outputTensor.shape()[1]

            Log.d(TAG, "✅ Model loaded. Input: ${inputImageWidth}x$inputImageHeight, Output size: $outputEmbeddingSize")
        } catch (e: Exception) {
            throw RuntimeException("Failed to load FaceNet model", e)
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val afd = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }

    /**
     * Crops a face from the bitmap, adds margin, makes it square, then resizes to model input.
     */
    fun cropFace(bitmap: Bitmap, boundingBox: Rect, marginFactor: Float = 0.3f): Bitmap {
        val width = boundingBox.width()
        val height = boundingBox.height()
        val marginW = (width * marginFactor).toInt()
        val marginH = (height * marginFactor).toInt()

        val left = (boundingBox.left - marginW).coerceAtLeast(0)
        val top = (boundingBox.top - marginH).coerceAtLeast(0)
        val right = (boundingBox.right + marginW).coerceAtMost(bitmap.width)
        val bottom = (boundingBox.bottom + marginH).coerceAtMost(bitmap.height)

        val cropped = Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)

        // Make it square (center crop)
        val size = min(cropped.width, cropped.height)
        val startX = (cropped.width - size) / 2
        val startY = (cropped.height - size) / 2
        val square = Bitmap.createBitmap(cropped, startX, startY, size, size)

        return Bitmap.createScaledBitmap(square, inputImageWidth, inputImageHeight, true)
    }

    /**
     * Converts a face Bitmap to normalized ByteBuffer and runs inference.
     * Returns L2-normalized embedding, with detailed logging.
     */
    fun getFaceEmbedding(faceBitmap: Bitmap): FloatArray {
        // Ensure correct size
        val resized = if (faceBitmap.width != inputImageWidth || faceBitmap.height != inputImageHeight) {
            Bitmap.createScaledBitmap(faceBitmap, inputImageWidth, inputImageHeight, true)
        } else faceBitmap

        val inputBuffer = ByteBuffer.allocateDirect(4 * inputImageWidth * inputImageHeight * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        resized.getPixels(pixels, 0, inputImageWidth, 0, 0, inputImageWidth, inputImageHeight)

        for (pixel in pixels) {
            // EXACT Python normalization: (value - 127.5) / 128.0
            val r = ((pixel shr 16 and 0xFF) - 127.5f) / 128.0f
            val g = ((pixel shr 8 and 0xFF) - 127.5f) / 128.0f
            val b = ((pixel and 0xFF) - 127.5f) / 128.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        val outputBuffer = ByteBuffer.allocateDirect(4 * outputEmbeddingSize)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val rawEmbedding = FloatArray(outputEmbeddingSize)
        outputBuffer.asFloatBuffer().get(rawEmbedding)

        // Log raw embedding stats
        val min = rawEmbedding.minOrNull() ?: 0f
        val max = rawEmbedding.maxOrNull() ?: 0f
        val avg = rawEmbedding.average()
        Log.d(TAG, "📊 Raw embedding - min: $min, max: $max, avg: $avg")

        // L2 normalize
        val normalized = l2Normalize(rawEmbedding)
        val norm = sqrt(normalized.map { it * it }.sum())
        Log.d(TAG, "✅ L2 norm after normalization: $norm (should be ~1.0)")

        return normalized
    }

    private fun l2Normalize(embedding: FloatArray): FloatArray {
        var norm = 0.0f
        for (v in embedding) norm += v * v
        norm = sqrt(norm)
        Log.d(TAG, "📏 L2 norm before normalization: $norm")
        if (norm < 1e-12f) return embedding
        return FloatArray(embedding.size) { embedding[it] / norm }
    }

    /**
     * Cosine similarity between two L2-normalized embeddings = dot product.
     */
    fun cosineSimilarity(e1: FloatArray, e2: FloatArray): Float {
        if (e1.size != e2.size) return 0f
        var dot = 0f
        for (i in e1.indices) dot += e1[i] * e2[i]
        return dot
    }

    fun close() {
        interpreter.close()
    }
}