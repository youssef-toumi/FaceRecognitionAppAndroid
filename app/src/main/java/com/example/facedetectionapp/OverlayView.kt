// File: app/src/main/java/com/example/facedetectionapp/OverlayView.kt
package com.example.facedetectionapp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val results = mutableListOf<FaceRecognitionResult>()
    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val textPaint = Paint().apply {
        color = Color.GREEN
        textSize = 40f
        style = Paint.Style.FILL
    }

    fun setResults(newResults: List<FaceRecognitionResult>) {
        results.clear()
        results.addAll(newResults)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (result in results) {
            canvas.drawRect(result.boundingBox, paint)
            val text = "${result.name} (${"%.2f".format(result.similarity)})"
            canvas.drawText(text, result.boundingBox.left.toFloat(), result.boundingBox.top - 10f, textPaint)
        }
    }
}