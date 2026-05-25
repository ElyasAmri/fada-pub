package com.fada.ultrasound.llm

import android.content.Context
import android.os.Build
import android.system.Os
import android.system.OsConstants
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import java.io.File

class LlmRuntimeFactory {
    fun enforceSupportedRuntime() {
        val pageSize = runCatching { Os.sysconf(OsConstants._SC_PAGESIZE).toInt() }.getOrDefault(4096)
        val primaryAbi = Build.SUPPORTED_ABIS.firstOrNull().orEmpty()
        val isX86_64 = primaryAbi.contains("x86_64", ignoreCase = true)
        val is16Kb = pageSize >= 16_384

        if (isX86_64 && is16Kb) {
            throw IllegalStateException(
                "This emulator image is not supported for on-device model runtime " +
                    "(x86_64 with 16 KB page size). Use an ARM64 image/device."
            )
        }
    }

    fun create(context: Context, modelFile: File): Engine {
        val gpuResult = createInitializedEngine(context, modelFile, useGpu = true)
        gpuResult.getOrNull()?.let { return it }

        val cpuResult = createInitializedEngine(context, modelFile, useGpu = false)
        cpuResult.getOrNull()?.let { return it }

        throw IllegalStateException("Failed to initialize model runtime").apply {
            gpuResult.exceptionOrNull()?.let { addSuppressed(it) }
            cpuResult.exceptionOrNull()?.let { addSuppressed(it) }
        }
    }

    private fun createInitializedEngine(
        context: Context,
        modelFile: File,
        useGpu: Boolean
    ): Result<Engine> {
        return try {
            val backend = if (useGpu) Backend.GPU() else Backend.CPU()
            val engine = Engine(
                EngineConfig(
                    modelPath = modelFile.absolutePath,
                    backend = backend,
                    visionBackend = backend,
                    cacheDir = context.cacheDir.absolutePath
                )
            )
            engine.initialize()
            Result.success(engine)
        } catch (error: RuntimeException) {
            Result.failure(error)
        }
    }
}
