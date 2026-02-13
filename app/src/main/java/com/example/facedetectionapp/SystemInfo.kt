package com.example.facedetectionapp

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import android.os.Debug
import android.util.Log
import java.io.BufferedReader
import java.io.FileReader
import java.io.RandomAccessFile

object SystemInfo {

    private const val TAG = "SystemInfo"

    fun logDeviceInfo(context: Context) {
        Log.d(TAG, "========== DEVICE INFORMATION ==========")
        Log.d(TAG, "Device Model: ${Build.MODEL}")
        Log.d(TAG, "Manufacturer: ${Build.MANUFACTURER}")
        Log.d(TAG, "Android Version: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
        Log.d(TAG, "Kernel: ${System.getProperty("os.version")}")
        Log.d(TAG, "CPU ABI: ${Build.SUPPORTED_ABIS.joinToString()}")

        val cores = Runtime.getRuntime().availableProcessors()
        Log.d(TAG, "CPU Cores: $cores")

        val maxFreq = readCpuMaxFreq()
        Log.d(TAG, "Max CPU Frequency: ${if (maxFreq > 0) "${maxFreq / 1000} MHz" else "Not available (emulated?)"}")

        val curFreq = readCpuCurFreq()
        Log.d(TAG, "Current CPU Frequency (core 0): ${if (curFreq > 0) "${curFreq / 1000} MHz" else "Not available"}")

        val memInfo = ActivityManager.MemoryInfo()
        (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memInfo)
        val totalRAM = memInfo.totalMem / (1024 * 1024)
        val availRAM = memInfo.availMem / (1024 * 1024)
        Log.d(TAG, "Total RAM: ${totalRAM} MB")
        Log.d(TAG, "Available RAM: ${availRAM} MB")

        val runtime = Runtime.getRuntime()
        val maxHeap = runtime.maxMemory() / (1024 * 1024)
        val totalHeap = runtime.totalMemory() / (1024 * 1024)
        val freeHeap = runtime.freeMemory() / (1024 * 1024)
        Log.d(TAG, "Heap: Max=${maxHeap} MB, Total=${totalHeap} MB, Free=${freeHeap} MB")

        Log.d(TAG, "=========================================")
    }

    private fun readCpuMaxFreq(): Long {
        return try {
            BufferedReader(FileReader("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")).use {
                it.readLine().toLong()
            }
        } catch (e: Exception) {
            -1L
        }
    }

    /** Read current frequency of CPU core 0 */
    fun readCpuCurFreq(): Long {
        // Try scaling_cur_freq first (common), then cpuinfo_cur_freq
        val paths = listOf(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"
        )
        for (path in paths) {
            try {
                return BufferedReader(FileReader(path)).use { it.readLine().toLong() }
            } catch (e: Exception) {
                // continue to next path
            }
        }
        return -1L
    }

    fun logMemoryUsage(tag: String = "Memory") {
        val runtime = Runtime.getRuntime()
        val total = runtime.totalMemory()
        val free = runtime.freeMemory()
        val used = total - free
        Log.d(tag, "Heap: Used=${used / 1024 / 1024} MB, Free=${free / 1024 / 1024} MB, Total=${total / 1024 / 1024} MB")

        val nativeHeap = Debug.getNativeHeapAllocatedSize() / (1024 * 1024)
        Log.d(tag, "Native Heap Allocated: ${nativeHeap} MB")
    }

    fun logCpuLoad() {
        try {
            val stat = RandomAccessFile("/proc/stat", "r")
            val line = stat.readLine()
            stat.close()
            val parts = line.split("\\s+".toRegex())
            if (parts[0] == "cpu") {
                val user = parts[1].toLong()
                val nice = parts[2].toLong()
                val system = parts[3].toLong()
                val idle = parts[4].toLong()
                val iowait = if (parts.size > 5) parts[5].toLong() else 0
                val irq = if (parts.size > 6) parts[6].toLong() else 0
                val softirq = if (parts.size > 7) parts[7].toLong() else 0
                val steal = if (parts.size > 8) parts[8].toLong() else 0
                val guest = if (parts.size > 9) parts[9].toLong() else 0

                val total = user + nice + system + idle + iowait + irq + softirq + steal + guest
                val busy = total - idle

                Log.d(TAG, "CPU Load: busy=$busy, total=$total, ratio=${busy * 100 / total}%")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read CPU load", e)
        }
    }

    /** Log current CPU frequency (core 0) */
    fun logCpuFrequency() {
        val freq = readCpuCurFreq()
        if (freq > 0) {
            Log.d(TAG, "Current CPU Frequency (core 0): ${freq / 1000} MHz")
        } else {
            Log.d(TAG, "Current CPU Frequency: not available")
        }
    }
}