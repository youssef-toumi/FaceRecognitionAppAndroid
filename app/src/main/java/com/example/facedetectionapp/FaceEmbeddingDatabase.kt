package com.example.facedetectionapp

import android.content.Context
import android.util.Log

class FaceEmbeddingDatabase(context: Context) {

    private val knownFaces = mutableMapOf<String, FloatArray>()
    private val storage = EmbeddingStorage(context)
    private val TAG = "FaceEmbeddingDatabase"

    init {
        // Load existing embeddings from persistent storage
        knownFaces.putAll(storage.loadEmbeddings())
        Log.d(TAG, "📦 Initialized with ${knownFaces.size} known faces")
    }

    fun addEmbedding(name: String, embedding: FloatArray) {
        knownFaces[name] = embedding.clone()
        storage.saveEmbeddings(knownFaces) // persist immediately
        Log.d(TAG, "✅ Added & saved embedding for '$name'. Total: ${knownFaces.size}")
    }

    fun removeEmbedding(name: String) {
        knownFaces.remove(name)
        storage.saveEmbeddings(knownFaces)
    }

    fun clear() {
        knownFaces.clear()
        storage.clear()
    }

    fun recognize(embedding: FloatArray, threshold: Float = 0.5f): Pair<String, Float> {
        var bestName = "Unknown"
        var bestSim = 0f

        for ((name, known) in knownFaces) {
            val sim = cosineSimilarity(embedding, known)
            Log.d(TAG, "   🔍 Compare with '$name' → similarity = ${"%.4f".format(sim)}")
            if (sim > bestSim) {
                bestSim = sim
                bestName = if (sim > threshold) name else "Unknown"
            }
        }

        Log.d(TAG, "🏆 Best match: '$bestName' (${"%.4f".format(bestSim)})")
        return bestName to bestSim
    }

    private fun cosineSimilarity(e1: FloatArray, e2: FloatArray): Float {
        var dot = 0f
        for (i in e1.indices) dot += e1[i] * e2[i]
        return dot
    }
}