package com.example.facedetectionapp

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import org.opencv.android.Utils
import org.opencv.calib3d.StereoSGBM
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * Computes a depth map from a rectified stereo pair.
 *
 * Pipeline:
 *   rectified left/right → grayscale → SGBM disparity → depth (mm)
 *
 * Depth formula:  Z = (f * B) / d
 *   f = focal length (pixels, from P1[0][0])
 *   B = baseline (mm, from ||T||)
 *   d = disparity (pixels)
 *
 * Usage:
 *   val processor = StereoDepthProcessor(calibData)
 *   val depthMap = processor.computeDepthMap(rectifiedLeft, rectifiedRight)
 *   val faceDepth = processor.getDepthInRegion(depthMap, faceBoundingBox)
 */
class StereoDepthProcessor(private val calib: StereoCalibrationData) {

    companion object {
        private const val TAG = "StereoDepth"

        // SGBM parameters — tuned for face at ~1m with ~7cm baseline
        private const val MIN_DISPARITY = 0
        private const val NUM_DISPARITIES = 128   // must be divisible by 16
        private const val BLOCK_SIZE = 5           // odd number, 3-11
        private const val P1_MULTIPLIER = 8        // smoothness penalty
        private const val P2_MULTIPLIER = 32       // larger smoothness penalty
        private const val DISP_12_MAX_DIFF = 1
        private const val PRE_FILTER_CAP = 63
        private const val UNIQUENESS_RATIO = 10
        private const val SPECKLE_WINDOW_SIZE = 100
        private const val SPECKLE_RANGE = 32
    }

    private val sgbm: StereoSGBM

    init {
        val channels = 3 // color images
        sgbm = StereoSGBM.create(
            MIN_DISPARITY,
            NUM_DISPARITIES,
            BLOCK_SIZE,
            P1_MULTIPLIER * channels * BLOCK_SIZE * BLOCK_SIZE,
            P2_MULTIPLIER * channels * BLOCK_SIZE * BLOCK_SIZE,
            DISP_12_MAX_DIFF,
            PRE_FILTER_CAP,
            UNIQUENESS_RATIO,
            SPECKLE_WINDOW_SIZE,
            SPECKLE_RANGE,
            StereoSGBM.MODE_SGBM_3WAY  // good quality/speed tradeoff
        )
        Log.d(TAG, "SGBM created: numDisparities=$NUM_DISPARITIES, blockSize=$BLOCK_SIZE")
    }

    /**
     * Result of depth computation.
     *
     * @param disparityMat  Raw disparity (CV_16S, scaled by 16). Keep for debug.
     * @param depthMm       Depth map in millimeters (CV_32F). 0 = invalid.
     * @param computeTimeMs Processing time.
     */
    data class DepthResult(
        val disparityMat: Mat,
        val depthMm: Mat,
        val computeTimeMs: Long
    ) {
        fun release() {
            disparityMat.release()
            depthMm.release()
        }
    }

    /**
     * Compute depth map from a rectified stereo pair.
     */
    fun computeDepthMap(rectifiedLeft: Bitmap, rectifiedRight: Bitmap): DepthResult {
        val start = System.currentTimeMillis()

        // Convert to OpenCV Mats
        val matL = Mat()
        val matR = Mat()
        Utils.bitmapToMat(rectifiedLeft, matL)
        Utils.bitmapToMat(rectifiedRight, matR)

        // Convert RGBA → BGR (OpenCV default)
        val bgrL = Mat()
        val bgrR = Mat()
        Imgproc.cvtColor(matL, bgrL, Imgproc.COLOR_RGBA2BGR)
        Imgproc.cvtColor(matR, bgrR, Imgproc.COLOR_RGBA2BGR)

        // Compute disparity (output is CV_16S, values are disparity * 16)
        val disparity16 = Mat()
        sgbm.compute(bgrL, bgrR, disparity16)

        // Convert to float disparity (divide by 16)
        val disparityFloat = Mat()
        disparity16.convertTo(disparityFloat, CvType.CV_32F, 1.0 / 16.0)

        // Convert disparity → depth in mm:  Z = (f * B) / d
        val depthMm = Mat.zeros(disparityFloat.rows(), disparityFloat.cols(), CvType.CV_32F)
        val fB = calib.focalLengthPx.toDouble() * calib.baselineMm.toDouble()

        // Process each pixel (TODO: optimize with native code if too slow)
        for (r in 0 until disparityFloat.rows()) {
            for (c in 0 until disparityFloat.cols()) {
                val d = disparityFloat.get(r, c)[0]
                if (d > 0.5) { // valid disparity
                    val z = fB / d
                    if (z in 100.0..5000.0) { // 10cm to 5m range
                        depthMm.put(r, c, z)
                    }
                }
            }
        }

        // Cleanup
        matL.release()
        matR.release()
        bgrL.release()
        bgrR.release()
        disparityFloat.release()

        val elapsed = System.currentTimeMillis() - start
        Log.d(TAG, "✅ Depth computed in ${elapsed}ms")

        return DepthResult(disparity16, depthMm, elapsed)
    }

    /**
     * Analyze depth values inside a face bounding box.
     *
     * @param depthMm  Depth map from computeDepthMap()
     * @param faceRect Face bounding box (in the rectified left image coordinates)
     * @return DepthStats with min, max, mean, std, range — or null if no valid pixels
     */
    fun getDepthInRegion(depthMm: Mat, faceRect: Rect): DepthStats? {
        // Clamp bounding box to image bounds
        val x1 = faceRect.left.coerceIn(0, depthMm.cols() - 1)
        val y1 = faceRect.top.coerceIn(0, depthMm.rows() - 1)
        val x2 = faceRect.right.coerceIn(0, depthMm.cols() - 1)
        val y2 = faceRect.bottom.coerceIn(0, depthMm.rows() - 1)

        if (x2 <= x1 || y2 <= y1) return null

        // Extract ROI
        val roi = depthMm.submat(y1, y2, x1, x2)
        val validDepths = mutableListOf<Float>()

        for (r in 0 until roi.rows()) {
            for (c in 0 until roi.cols()) {
                val z = roi.get(r, c)[0].toFloat()
                if (z > 0f) validDepths.add(z)
            }
        }
        roi.release()

        if (validDepths.size < 20) {
            Log.w(TAG, "Too few valid depth pixels in face region: ${validDepths.size}")
            return null
        }

        val mean = validDepths.average().toFloat()
        val min = validDepths.min()
        val max = validDepths.max()
        val range = max - min
        val variance = validDepths.map { (it - mean) * (it - mean) }.average().toFloat()
        val std = kotlin.math.sqrt(variance.toDouble()).toFloat()
        val validRatio = validDepths.size.toFloat() / ((x2 - x1) * (y2 - y1))

        Log.d(TAG, "📏 Face depth: mean=${"%.0f".format(mean)}mm, " +
                "range=${"%.0f".format(range)}mm, std=${"%.1f".format(std)}mm, " +
                "valid=${(validRatio * 100).toInt()}%")

        return DepthStats(
            meanMm = mean,
            minMm = min,
            maxMm = max,
            rangeMm = range,
            stdMm = std,
            validPixelRatio = validRatio,
            validPixelCount = validDepths.size
        )
    }

    /**
     * Depth statistics for a face region.
     */
    data class DepthStats(
        val meanMm: Float,        // average depth (distance from camera)
        val minMm: Float,         // closest point (nose tip)
        val maxMm: Float,         // farthest point (ears/sides)
        val rangeMm: Float,       // max - min (should be 80-150mm for real face)
        val stdMm: Float,         // standard deviation (real face ~20-40mm)
        val validPixelRatio: Float, // % of pixels with valid depth
        val validPixelCount: Int
    )
}
