package com.example.facedetectionapp

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log

/**
 * Stereo-depth-based anti-spoofing.
 *
 * A real face at ~1m has clear 3D structure:
 *   - Nose tip is ~60-100mm closer than ears
 *   - Depth standard deviation ~20-40mm
 *   - Good % of valid depth pixels
 *
 * A flat photo/screen has:
 *   - Near-zero depth variation (< 15mm range)
 *   - Very low standard deviation (< 5mm)
 *
 * This class wraps the full pipeline:
 *   StereoCapture → StereoRectifier → StereoDepthProcessor → decision
 *
 * Usage:
 *   val spoofDetector = DepthAntiSpoofing(context)
 *   // When recognition triggers:
 *   val result = spoofDetector.checkLiveness(cameraProvider, lifecycle, faceBoundingBox)
 *   if (result.isReal) { ... }
 */
class DepthAntiSpoofing(
    private val stereoCapture: StereoCapture,
    private val rectifier: StereoRectifier,
    private val depthProcessor: StereoDepthProcessor,
) {
    companion object {
        private const val TAG = "DepthAntiSpoof"

        // ── Thresholds (tune after real-world testing) ────────
        // A real face at ~1m should have depth range of 60-150mm
        private const val MIN_DEPTH_RANGE_MM = 30f     // below this → flat → spoof
        // Standard deviation should be meaningful for a real face
        private const val MIN_DEPTH_STD_MM = 10f       // below this → too flat → spoof
        // Need enough valid pixels to be confident
        private const val MIN_VALID_PIXEL_RATIO = 0.3f // at least 30% valid depth
        // Face should be within reasonable distance (500mm to 2000mm)
        private const val MIN_FACE_DISTANCE_MM = 400f
        private const val MAX_FACE_DISTANCE_MM = 2000f
    }

    /**
     * Result of depth-based liveness check.
     */
    data class DepthLivenessResult(
        val isReal: Boolean,
        val confidence: Float,       // 0.0 = certainly fake, 1.0 = certainly real
        val reason: String,          // human-readable explanation
        val depthStats: StereoDepthProcessor.DepthStats?,
        val totalTimeMs: Long,

        // Breakdown timings
        val captureTimeMs: Long,
        val rectifyTimeMs: Long,
        val depthTimeMs: Long,
    )

    /**
     * Perform a stereo depth liveness check on a face.
     *
     * IMPORTANT: The caller must unbind CameraX BEFORE calling this,
     * and rebind AFTER it returns (StereoCapture needs exclusive camera access).
     *
     * @param faceBoundingBox Bounding box of the detected face (from ML Kit, in
     *                        the coordinate system of the camera frame).
     *                        The box will be applied to the rectified left image.
     */
    suspend fun checkLiveness(faceBoundingBox: Rect): DepthLivenessResult {
        val totalStart = System.currentTimeMillis()

        // ── Step 1: Capture stereo pair ───────────────────────
        val stereoPair = try {
            stereoCapture.capture()
        } catch (e: StereoCapture.StereoException) {
            Log.e(TAG, "Stereo capture failed", e)
            return DepthLivenessResult(
                isReal = true, // fail-open: don't block if stereo fails
                confidence = 0f,
                reason = "Stereo capture failed: ${e.message}",
                depthStats = null,
                totalTimeMs = System.currentTimeMillis() - totalStart,
                captureTimeMs = 0, rectifyTimeMs = 0, depthTimeMs = 0
            )
        }
        val captureTime = stereoPair.captureTimeMs

        // ── Step 2: Rectify ───────────────────────────────────
        val rectified = rectifier.rectify(stereoPair.left, stereoPair.right)
        val rectifyTime = rectified.rectifyTimeMs

        // ── Step 3: Compute depth ─────────────────────────────
        val depthResult = depthProcessor.computeDepthMap(rectified.left, rectified.right)
        val depthTime = depthResult.computeTimeMs

        // ── Step 4: Analyze face region ───────────────────────
        val stats = depthProcessor.getDepthInRegion(depthResult.depthMm, faceBoundingBox)
        depthResult.release()

        val totalTime = System.currentTimeMillis() - totalStart

        if (stats == null) {
            Log.w(TAG, "Could not compute depth stats for face region")
            return DepthLivenessResult(
                isReal = true, // fail-open
                confidence = 0f,
                reason = "Insufficient depth data in face region",
                depthStats = null,
                totalTimeMs = totalTime,
                captureTimeMs = captureTime,
                rectifyTimeMs = rectifyTime,
                depthTimeMs = depthTime
            )
        }

        // ── Step 5: Make liveness decision ────────────────────
        val (isReal, confidence, reason) = evaluateDepth(stats)

        Log.d(TAG, "🛡️ Depth liveness: real=$isReal, confidence=${"%.0f".format(confidence * 100)}%, " +
                "reason=$reason, total=${totalTime}ms " +
                "(capture=${captureTime}ms, rectify=${rectifyTime}ms, depth=${depthTime}ms)")

        return DepthLivenessResult(
            isReal = isReal,
            confidence = confidence,
            reason = reason,
            depthStats = stats,
            totalTimeMs = totalTime,
            captureTimeMs = captureTime,
            rectifyTimeMs = rectifyTime,
            depthTimeMs = depthTime
        )
    }

    /**
     * Core decision logic. Returns (isReal, confidence, reason).
     */
    private fun evaluateDepth(
        stats: StereoDepthProcessor.DepthStats
    ): Triple<Boolean, Float, String> {

        // Check 1: Is the face at a reasonable distance?
        if (stats.meanMm < MIN_FACE_DISTANCE_MM || stats.meanMm > MAX_FACE_DISTANCE_MM) {
            return Triple(
                false, 0.2f,
                "Face distance out of range: ${"%.0f".format(stats.meanMm)}mm " +
                        "(expected ${MIN_FACE_DISTANCE_MM.toInt()}-${MAX_FACE_DISTANCE_MM.toInt()}mm)"
            )
        }

        // Check 2: Do we have enough valid depth pixels?
        if (stats.validPixelRatio < MIN_VALID_PIXEL_RATIO) {
            return Triple(
                true, 0.1f, // fail-open, low confidence
                "Low depth coverage: ${"%.0f".format(stats.validPixelRatio * 100)}%"
            )
        }

        // Check 3: Depth range — real face has significant nose-to-ear depth range
        if (stats.rangeMm < MIN_DEPTH_RANGE_MM) {
            return Triple(
                false, 0.9f,
                "Too flat: range=${"%.0f".format(stats.rangeMm)}mm (need >${MIN_DEPTH_RANGE_MM.toInt()}mm)"
            )
        }

        // Check 4: Standard deviation — real face has meaningful depth variation
        if (stats.stdMm < MIN_DEPTH_STD_MM) {
            return Triple(
                false, 0.8f,
                "Too uniform: std=${"%.1f".format(stats.stdMm)}mm (need >${MIN_DEPTH_STD_MM.toInt()}mm)"
            )
        }

        // All checks passed → real face
        // Confidence scales with how clearly 3D the face looks
        val rangeScore = ((stats.rangeMm - MIN_DEPTH_RANGE_MM) / 100f).coerceIn(0f, 1f)
        val stdScore = ((stats.stdMm - MIN_DEPTH_STD_MM) / 30f).coerceIn(0f, 1f)
        val confidence = (rangeScore * 0.6f + stdScore * 0.4f).coerceIn(0.5f, 1f)

        return Triple(
            true, confidence,
            "3D face verified: range=${"%.0f".format(stats.rangeMm)}mm, " +
                    "std=${"%.1f".format(stats.stdMm)}mm, " +
                    "dist=${"%.0f".format(stats.meanMm)}mm"
        )
    }
}
