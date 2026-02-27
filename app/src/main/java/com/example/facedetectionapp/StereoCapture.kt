package com.example.facedetectionapp

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Captures a synchronized stereo pair from the device's two cameras.
 *
 * Hardware constraints (MediaTek SoC, Alps kiosk):
 * - Simultaneous dual streaming is NOT supported (concurrent == false)
 * - Only "Full alternate 0→1" works: open Camera 0, capture, FULLY close,
 *   then open Camera 1, capture, FULLY close.
 * - Camera 1→0 order fails due to MediaTek HAL initialization quirk.
 * - Total capture time ≈ 1.5 seconds.
 *
 * Usage:
 *   // 1. Pause CameraX first
 *   cameraProvider.unbindAll()
 *   // 2. Capture stereo pair
 *   val pair = stereoCapture.capture()
 *   // 3. Resume CameraX
 *   cameraProvider.bindToLifecycle(...)
 */
class StereoCapture(private val context: Context) {

    companion object {
        private const val TAG = "StereoCapture"
        private const val CAMERA_0_ID = "0"
        private const val CAMERA_1_ID = "1"
        private const val CAPTURE_TIMEOUT_MS = 5000L
        private const val CAPTURE_WIDTH = 1280
        private const val CAPTURE_HEIGHT = 720
        // Delay after closing a camera to let the HAL fully release resources
        private const val HAL_RELEASE_DELAY_MS = 150L
    }

    /**
     * Result of a stereo capture.
     * @param left  Bitmap from Camera 0 (left lens)
     * @param right Bitmap from Camera 1 (right lens)
     * @param captureTimeMs Wall-clock time for the full stereo capture
     */
    data class StereoPair(
        val left: Bitmap,
        val right: Bitmap,
        val captureTimeMs: Long
    )

    private val cameraManager =
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    /**
     * Captures a stereo image pair. Must be called from a coroutine.
     *
     * The caller is responsible for:
     *   1. Pausing CameraX (unbindAll) BEFORE calling this
     *   2. Resuming CameraX AFTER this returns
     *
     * @return StereoPair with left and right bitmaps
     * @throws StereoException if either capture fails
     */
    suspend fun capture(): StereoPair = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()
        Log.d(TAG, "▶ Starting stereo capture...")

        val callbackThread = HandlerThread("StereoCallbackThread").apply { start() }
        val callbackHandler = Handler(callbackThread.looper)

        try {
            // --- Camera 0 (must always go first on MediaTek) ---
            Log.d(TAG, "  Capturing from Camera 0...")
            val leftBitmap = captureFromCamera(CAMERA_0_ID, callbackHandler)
                ?: throw StereoException("Failed to capture from Camera 0")
            Log.d(TAG, "  ✓ Camera 0: ${leftBitmap.width}x${leftBitmap.height}")

            // --- Camera 1 ---
            Log.d(TAG, "  Capturing from Camera 1...")
            val rightBitmap = captureFromCamera(CAMERA_1_ID, callbackHandler)
                ?: throw StereoException("Failed to capture from Camera 1")
            Log.d(TAG, "  ✓ Camera 1: ${rightBitmap.width}x${rightBitmap.height}")

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "◼ Stereo capture completed in ${elapsed}ms")

            StereoPair(leftBitmap, rightBitmap, elapsed)
        } finally {
            callbackThread.quitSafely()
        }
    }

    // -----------------------------------------------------------------------
    //  Internal: open one camera → capture one JPEG frame → fully close
    // -----------------------------------------------------------------------

    @SuppressLint("MissingPermission")
    private fun captureFromCamera(cameraId: String, callbackHandler: Handler): Bitmap? {
        var resultBitmap: Bitmap? = null
        val openLatch = CountDownLatch(1)
        val frameLatch = CountDownLatch(1)
        var cameraDevice: CameraDevice? = null
        var captureSession: CameraCaptureSession? = null
        var imageReader: ImageReader? = null

        try {
            // 1. Set up ImageReader (JPEG so we get a ready-to-decode frame)
            imageReader = ImageReader.newInstance(
                CAPTURE_WIDTH, CAPTURE_HEIGHT,
                ImageFormat.JPEG, 2
            )
            imageReader.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage()
                if (image != null) {
                    try {
                        val buffer = image.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)
                        resultBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        Log.d(TAG, "    Frame acquired from camera $cameraId")
                    } finally {
                        image.close()
                    }
                    frameLatch.countDown()
                }
            }, callbackHandler)

            // 2. Open the camera device
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    openLatch.countDown()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    Log.w(TAG, "Camera $cameraId disconnected")
                    camera.close()
                    openLatch.countDown()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Log.e(TAG, "Camera $cameraId open error: $error")
                    camera.close()
                    openLatch.countDown()
                }
            }, callbackHandler)

            // Block on the IO thread (NOT the callback thread) until open completes
            if (!openLatch.await(CAPTURE_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                Log.e(TAG, "Timeout opening camera $cameraId")
                return null
            }
            val device = cameraDevice ?: return null

            // 3. Create a capture session targeting the ImageReader surface
            val sessionLatch = CountDownLatch(1)
            val surface = imageReader.surface

            @Suppress("DEPRECATION")
            device.createCaptureSession(
                listOf(surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        captureSession = session
                        sessionLatch.countDown()
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e(TAG, "Session config failed for camera $cameraId")
                        sessionLatch.countDown()
                    }
                },
                callbackHandler
            )

            if (!sessionLatch.await(CAPTURE_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                Log.e(TAG, "Timeout configuring session for camera $cameraId")
                return null
            }
            val session = captureSession ?: return null

            // 4. Submit a single still-capture request
            val captureRequest = device.createCaptureRequest(
                CameraDevice.TEMPLATE_STILL_CAPTURE
            ).apply {
                addTarget(surface)
                set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                )
            }.build()

            session.capture(captureRequest, null, callbackHandler)

            // 5. Wait for the frame to arrive
            if (!frameLatch.await(CAPTURE_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                Log.e(TAG, "Timeout waiting for frame from camera $cameraId")
                return null
            }

            return resultBitmap

        } catch (e: SecurityException) {
            Log.e(TAG, "Camera permission denied for $cameraId", e)
            return null
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Camera access error for $cameraId", e)
            return null
        } finally {
            // CRITICAL: fully close everything so the next camera can use the ISP
            try { captureSession?.close() } catch (_: Exception) {}
            try { cameraDevice?.close() } catch (_: Exception) {}
            try { imageReader?.close() } catch (_: Exception) {}
            // Give the MediaTek HAL time to fully release
            Thread.sleep(HAL_RELEASE_DELAY_MS)
        }
    }

    /** Thrown when stereo capture fails. */
    class StereoException(message: String) : Exception(message)
}
