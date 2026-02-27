package com.example.facedetectionapp

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.IOException

/**
 * Loads stereo calibration data produced by tools/stereo_calibrate.py.
 *
 * Place the file at: app/src/main/assets/stereo_calib.json
 *
 * Contains:
 *  - Camera intrinsics (K1, D1, K2, D2)
 *  - Stereo extrinsics (R, T)
 *  - Rectification (R1, R2, P1, P2, Q)
 *  - Pre-computed remap LUTs (map1x, map1y, map2x, map2y)
 */
class StereoCalibrationData private constructor(
    val imageWidth: Int,
    val imageHeight: Int,
    val baselineMm: Float,
    val stereoError: Float,

    // Remap lookup tables for fast rectification
    val map1x: Mat,
    val map1y: Mat,
    val map2x: Mat,
    val map2y: Mat,

    // Disparity-to-depth matrix
    val Q: Mat,

    // Focal length from P1 (needed for depth = f*B/d)
    val focalLengthPx: Float,
) {
    companion object {
        private const val TAG = "StereoCalibData"
        private const val ASSET_FILE = "stereo_calib.json"

        /**
         * Load calibration from assets. Returns null if file is missing.
         * Call this once at startup, keep the result.
         */
        fun loadFromAssets(context: Context): StereoCalibrationData? {
            return try {
                val jsonStr = context.assets.open(ASSET_FILE).bufferedReader().readText()
                parseJson(jsonStr)
            } catch (e: IOException) {
                Log.w(TAG, "Stereo calibration file not found ($ASSET_FILE). Stereo depth disabled.")
                null
            } catch (e: Exception) {
                Log.e(TAG, "Failed to parse stereo calibration", e)
                null
            }
        }

        private fun parseJson(jsonStr: String): StereoCalibrationData {
            val json = JSONObject(jsonStr)

            val size = json.getJSONArray("image_size")
            val w = size.getInt(0)
            val h = size.getInt(1)

            val baselineMm = json.getDouble("baseline_mm").toFloat()
            val stereoError = json.getDouble("stereo_reprojection_error").toFloat()

            val map1x = jsonArrayToMat(json.getJSONArray("map1x"), h, w, CvType.CV_32FC1)
            val map1y = jsonArrayToMat(json.getJSONArray("map1y"), h, w, CvType.CV_32FC1)
            val map2x = jsonArrayToMat(json.getJSONArray("map2x"), h, w, CvType.CV_32FC1)
            val map2y = jsonArrayToMat(json.getJSONArray("map2y"), h, w, CvType.CV_32FC1)

            val Q = jsonArrayToMat(json.getJSONArray("Q"), 4, 4, CvType.CV_64FC1)

            // P1 is 3x4; focal length is P1[0][0]
            val p1 = json.getJSONArray("P1")
            val focalLength = p1.getJSONArray(0).getDouble(0).toFloat()

            Log.d(TAG, "✅ Loaded stereo calibration: ${w}x${h}, baseline=${baselineMm}mm, f=${focalLength}px, error=${stereoError}px")

            return StereoCalibrationData(
                imageWidth = w,
                imageHeight = h,
                baselineMm = baselineMm,
                stereoError = stereoError,
                map1x = map1x,
                map1y = map1y,
                map2x = map2x,
                map2y = map2y,
                Q = Q,
                focalLengthPx = focalLength,
            )
        }

        /**
         * Convert a JSON 2D array [[row0], [row1], ...] to an OpenCV Mat.
         */
        private fun jsonArrayToMat(arr: JSONArray, rows: Int, cols: Int, type: Int): Mat {
            val mat = Mat(rows, cols, type)
            for (r in 0 until rows) {
                val rowArr = arr.getJSONArray(r)
                for (c in 0 until cols) {
                    when (type) {
                        CvType.CV_32FC1 -> mat.put(r, c, floatArrayOf(rowArr.getDouble(c).toFloat()))
                        CvType.CV_64FC1 -> mat.put(r, c, doubleArrayOf(rowArr.getDouble(c)))
                    }
                }
            }
            return mat
        }
    }

    fun release() {
        map1x.release()
        map1y.release()
        map2x.release()
        map2y.release()
        Q.release()
    }
}
