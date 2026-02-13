package com.example.facedetectionapp

import android.graphics.Rect

data class FaceRecognitionResult(
    val boundingBox: Rect,
    val name: String,
    val similarity: Float
)