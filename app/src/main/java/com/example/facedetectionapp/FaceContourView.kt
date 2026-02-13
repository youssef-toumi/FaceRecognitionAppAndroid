package com.example.facedetectionapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.util.Log
import android.view.View
import com.google.mlkit.vision.face.Face

class FaceContourView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val faces = mutableListOf<Face>()
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0

    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    fun updateFaces(newFaces: List<Face>, width: Int, height: Int) {
        Log.d("FaceContourView", "📥 updateFaces: ${newFaces.size} faces, image size: ${width}x${height}")
        faces.clear()
        faces.addAll(newFaces)
        imageWidth = width
        imageHeight = height
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (faces.isEmpty() || imageWidth == 0 || imageHeight == 0) {
            return
        }

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        val imageAspectRatio = imageWidth.toFloat() / imageHeight.toFloat()
        val viewAspectRatio = viewWidth / viewHeight

        val scale: Float
        var offsetX = 0f
        var offsetY = 0f

        if (imageAspectRatio > viewAspectRatio) {
            scale = viewHeight / imageHeight.toFloat()
            offsetX = (viewWidth - imageWidth * scale) / 2f
        } else {
            scale = viewWidth / imageWidth.toFloat()
            offsetY = (viewHeight - imageHeight * scale) / 2f
        }

        for (face in faces) {
            val bounds = face.boundingBox

            // Mirror X for front camera
            val mirroredLeft = imageWidth - bounds.right
            val mirroredRight = imageWidth - bounds.left

            val scaledLeft = mirroredLeft * scale + offsetX
            val scaledTop = bounds.top * scale + offsetY
            val scaledRight = mirroredRight * scale + offsetX
            val scaledBottom = bounds.bottom * scale + offsetY

            val rect = Rect(
                scaledLeft.toInt(),
                scaledTop.toInt(),
                scaledRight.toInt(),
                scaledBottom.toInt()
            )
            canvas.drawRect(rect, paint)
        }
    }
}