package com.fada.ultrasound.llm

import android.content.Context
import android.os.PowerManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ModelDownloadState(
    val modelId: String,
    val isBusy: Boolean,
    val status: String? = null,
    val error: String? = null
)

object ModelDownloadCoordinator {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val _states = MutableStateFlow<Map<String, ModelDownloadState>>(emptyMap())
    val states: StateFlow<Map<String, ModelDownloadState>> = _states.asStateFlow()

    fun startDownload(context: Context, model: LlmModelOption) {
        val existing = _states.value[model.id]
        if (existing?.isBusy == true) {
            return
        }

        val appContext = context.applicationContext
        val wakeLock = appContext.modelDownloadWakeLock()
        _states.update {
            it + (model.id to ModelDownloadState(modelId = model.id, isBusy = true, status = "Queued"))
        }
        scope.launch {
            try {
                wakeLock?.acquire(MAX_DOWNLOAD_WAKE_MS)
                ModelDownloadManager(appContext).getOrDownloadModel(model) { status ->
                    _states.update {
                        it + (model.id to ModelDownloadState(modelId = model.id, isBusy = true, status = status))
                    }
                }
                _states.update {
                    it + (model.id to ModelDownloadState(modelId = model.id, isBusy = false, status = "Stored"))
                }
            } catch (e: IllegalStateException) {
                markFailed(model.id, e.message ?: "Model download failed")
            } catch (e: RuntimeException) {
                markFailed(model.id, e.message ?: "Model download failed")
            } finally {
                if (wakeLock?.isHeld == true) {
                    wakeLock.release()
                }
            }
        }
    }

    private fun markFailed(modelId: String, message: String) {
        _states.update {
            it + (modelId to ModelDownloadState(modelId = modelId, isBusy = false, status = message, error = message))
        }
    }

    private fun Context.modelDownloadWakeLock(): PowerManager.WakeLock? {
        val powerManager = getSystemService(Context.POWER_SERVICE) as? PowerManager ?: return null
        return powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "FADA:ModelDownload"
        ).apply {
            setReferenceCounted(false)
        }
    }

    private const val MAX_DOWNLOAD_WAKE_MS = 6L * 60L * 60L * 1_000L
}
