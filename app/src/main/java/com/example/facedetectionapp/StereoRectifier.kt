package com.example.facedetectionapp

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * Undistorts + rectifies a stereo image pair using pre-computed remap tables.
 *
 * After rectification, matching points in left/right images lie on the same
 * horizontal scanline — which is required for disparity computation.
 *
 * Usage:
 *   val rectifier = StereoRectifier(calibData)
 *   val (rectLeft, rectRight) = rectifier.rectify(leftBitmap, rightBitmap)
 */
class StereoRectifier(private val calib: StereoCalibrationData) {

    companion object {
        private const val TAG = "StereoRectifier"
    }

    /**
     * Rectified stereo pair result.
     */
    data class RectifiedPair(
        val left: Bitmap,
        val right: Bitmap,
        val rectifyTimeMs: Long
    )

    /**
     * Rectify a stereo image pair.
     *
     * @param leftBitmap  Bitmap from Camera 0
     * @param rightBitmap Bitmap from Camera 1
     * @return RectifiedPair with corrected images
     */
    fun rectify(leftBitmap: Bitmap, rightBitmap: Bitmap): RectifiedPair {
        val start = System.currentTimeMillis()

        // Convert Bitmaps → OpenCV Mats
        val matL = Mat()
        val matR = Mat()
        Utils.bitmapToMat(leftBitmap, matL)
        Utils.bitmapToMat(rightBitmap, matR)

        // Apply remap (undistort + rectify in one step)
        val rectL = Mat()
        val rectR = Mat()
        Imgproc.remap(matL, rectL, calib.map1x, calib.map1y, Imgproc.INTER_LINEAR)
        Imgproc.remap(matR, rectR, calib.map2x, calib.map2y, Imgproc.INTER_LINEAR)

        // Convert back to Bitmaps
        val resultL = Bitmap.createBitmap(rectL.cols(), rectL.rows(), Bitmap.Config.ARGB_8888)
        val resultR = Bitmap.createBitmap(rectR.cols(), rectR.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rectL, resultL)
        Utils.matToBitmap(rectR, resultR)

        // Release Mats
        matL.release()
        matR.release()
        rectL.release()
        rectR.release()

        val elapsed = System.currentTimeMillis() - start
        Log.d(TAG, "✅ Rectification done in ${elapsed}ms")

        return RectifiedPair(resultL, resultR, elapsed)
    }
}
