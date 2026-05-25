package com.fada.ultrasound.llm

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Coordinates model storage, runtime lifecycle, and response streaming.
 */
object LlmResponseGenerator : LlmResponseClient {
    private val mutex = Mutex()
    private val contentFactory = LlmContentFactory()
    private val runtimeFactory = LlmRuntimeFactory()
    private val runtimes = mutableMapOf<String, LlmModelRuntime>()

    override suspend fun prepare(
        context: Context,
        model: LlmModelOption,
        onStatus: (String) -> Unit
    ) {
        runtimeFactory.enforceSupportedRuntime()
        mutex.withLock {
            getOrCreateRuntime(context, model, onStatus)
        }
    }

    override suspend fun generate(
        context: Context,
        model: LlmModelOption,
        conversationId: String,
        history: List<LlmChatTurn>,
        image: Bitmap?,
        imageFileName: String?,
        prompt: String,
        systemPrompt: String,
        onStatus: (String) -> Unit,
        onPartialResponse: (String) -> Unit
    ): String {
        runtimeFactory.enforceSupportedRuntime()
        val normalizedPrompt = prompt.ifBlank { "Continue." }
        contentFactory.createCurrentContents(
            context = context,
            image = image,
            imageFileName = imageFileName,
            prompt = normalizedPrompt
        ).use { preparedContents ->
            return mutex.withLock {
                getOrCreateRuntime(context, model, onStatus).generate(
                    conversationId = conversationId,
                    history = history,
                    systemPrompt = systemPrompt,
                    contents = preparedContents.contents,
                    onPartialResponse = onPartialResponse
                )
            }
        }
    }

    override fun release() {
        runBlocking {
            mutex.withLock {
                runtimes.values.forEach { runtime -> runtime.close() }
                runtimes.clear()
            }
        }
    }

    override fun releaseModel(modelId: String) {
        runBlocking {
            mutex.withLock {
                runtimes.remove(modelId)?.close()
            }
        }
    }

    private fun getOrCreateRuntime(
        context: Context,
        model: LlmModelOption,
        onStatus: (String) -> Unit
    ): LlmModelRuntime {
        runtimes[model.id]?.let { return it }

        closeOtherRuntimes(model.id)
        val modelFile = ModelDownloadManager(context).getOrDownloadModel(
            option = model,
            onStatus = onStatus
        )
        onStatus("Loading ${model.displayName}...")

        return LlmModelRuntime(
            engine = runtimeFactory.create(context, modelFile),
            contentFactory = contentFactory
        ).also { runtime ->
            runtimes[model.id] = runtime
        }
    }

    private fun closeOtherRuntimes(activeModelId: String) {
        val staleIds = runtimes.keys.filter { it != activeModelId }
        staleIds.forEach { modelId ->
            runtimes.remove(modelId)?.close()
        }
    }
}
