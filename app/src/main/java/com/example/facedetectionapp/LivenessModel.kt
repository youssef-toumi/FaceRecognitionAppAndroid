package com.example.facedetectionapp

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.sqrt

class LivenessModel(private val context: Context) {

    private val TAG = "LivenessModel"
    private val interpreter: Interpreter
    private val inputImageWidth: Int
    private val inputImageHeight: Int
    private val isNCHW: Boolean

    init {
        try {
            val modelBuffer = loadModelFile("liveness_detection_model.tflite")
            val options = Interpreter.Options().apply { setNumThreads(2) }
            interpreter = Interpreter(modelBuffer, options)

            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape()

            if (inputShape.size == 4) {
                if (inputShape[1] == 3) {
                    isNCHW = true
                    inputImageHeight = inputShape[2]
                    inputImageWidth = inputShape[3]
                    Log.d(TAG, "✅ Model layout: NCHW, ${inputImageHeight}x${inputImageWidth}")
                } else {
                    isNCHW = false
                    inputImageHeight = inputShape[1]
                    inputImageWidth = inputShape[2]
                    Log.d(TAG, "✅ Model layout: NHWC, ${inputImageHeight}x${inputImageWidth}")
                }
            } else {
                throw RuntimeException("Unexpected input tensor shape: ${inputShape.joinToString()}")
            }

        } catch (e: Exception) {
            throw RuntimeException("Failed to load liveness model", e)
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val afd = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }

    fun predict(fullImage: Bitmap): Pair<Float, Boolean> {
        // Resize to model input size
        val resized = Bitmap.createScaledBitmap(fullImage, inputImageWidth, inputImageHeight, true)

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        resized.getPixels(pixels, 0, inputImageWidth, 0, 0, inputImageWidth, inputImageHeight)

        // Extract raw RGB values in [0,1] range
        val rawR = FloatArray(pixels.size)
        val rawG = FloatArray(pixels.size)
        val rawB = FloatArray(pixels.size)

        for ((i, pixel) in pixels.withIndex()) {
            rawR[i] = (pixel shr 16 and 0xFF) / 255.0f
            rawG[i] = (pixel shr 8 and 0xFF) / 255.0f
            rawB[i] = (pixel and 0xFF) / 255.0f
        }

        // Compute per‑channel mean and standard deviation
        val rMean = rawR.average().toFloat()
        val gMean = rawG.average().toFloat()
        val bMean = rawB.average().toFloat()

        val rStd = sqrt(rawR.map { (it - rMean) * (it - rMean) }.average().toFloat())
        val gStd = sqrt(rawG.map { (it - gMean) * (it - gMean) }.average().toFloat())
        val bStd = sqrt(rawB.map { (it - bMean) * (it - bMean) }.average().toFloat())

        // Normalize each channel to zero mean and unit variance
        val rNorm = FloatArray(pixels.size)
        val gNorm = FloatArray(pixels.size)
        val bNorm = FloatArray(pixels.size)

        for (i in pixels.indices) {
            rNorm[i] = (rawR[i] - rMean) / rStd
            gNorm[i] = (rawG[i] - gMean) / gStd
            bNorm[i] = (rawB[i] - bMean) / bStd
        }

        // Prepare input buffer
        val bufferSize = 4 * inputImageWidth * inputImageHeight * 3
        val inputBuffer = ByteBuffer.allocateDirect(bufferSize)
        inputBuffer.order(ByteOrder.nativeOrder())

        if (isNCHW) {
            // NCHW: all R, then all G, then all B
            for (value in rNorm) inputBuffer.putFloat(value)
            for (value in gNorm) inputBuffer.putFloat(value)
            for (value in bNorm) inputBuffer.putFloat(value)
        } else {
            // NHWC: interleaved RGB per pixel
            for (i in pixels.indices) {
                inputBuffer.putFloat(rNorm[i])
                inputBuffer.putFloat(gNorm[i])
                inputBuffer.putFloat(bNorm[i])
            }
        }

        val outputBuffer = ByteBuffer.allocateDirect(4)
        outputBuffer.order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val logit = outputBuffer.float

        // Sigmoid to get probability
        val probability = 1.0f / (1.0f + exp(-logit))
        // Interpretation: probability < 0.5 = real (live), > 0.5 = fake (spoof)
        val isReal = probability < 0.5f

        Log.d(TAG, "🔮 Liveness - logit: $logit, prob: $probability, isReal: $isReal")
        return Pair(probability, isReal)
    }

    fun close() {
        interpreter.close()
    }
}