package com.example.facedetectionapp

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject

class EmbeddingStorage(private val context: Context) {

    private val prefs = context.getSharedPreferences("face_embeddings", Context.MODE_PRIVATE)
    private val TAG = "EmbeddingStorage"

    fun saveEmbeddings(embeddings: Map<String, FloatArray>) {
        val json = JSONArray()
        for ((name, vector) in embeddings) {
            val entry = JSONObject()
            entry.put("name", name)
            val array = JSONArray()
            vector.forEach { array.put(it.toDouble()) }
            entry.put("embedding", array)
            json.put(entry)
        }
        prefs.edit().putString("embeddings", json.toString()).apply()
        Log.d(TAG, "✅ Saved ${embeddings.size} embeddings")
    }

    fun loadEmbeddings(): Map<String, FloatArray> {
        val jsonString = prefs.getString("embeddings", null) ?: return emptyMap()
        val map = mutableMapOf<String, FloatArray>()
        try {
            val jsonArray = JSONArray(jsonString)
            for (i in 0 until jsonArray.length()) {
                val entry = jsonArray.getJSONObject(i)
                val name = entry.getString("name")
                val array = entry.getJSONArray("embedding")
                val floatArray = FloatArray(array.length()) { array.getDouble(it).toFloat() }
                map[name] = floatArray
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load embeddings", e)
        }
        Log.d(TAG, "📂 Loaded ${map.size} embeddings")
        return map
    }

    fun clear() {
        prefs.edit().remove("embeddings").apply()
        Log.d(TAG, "🗑️ Cleared all embeddings")
    }
}