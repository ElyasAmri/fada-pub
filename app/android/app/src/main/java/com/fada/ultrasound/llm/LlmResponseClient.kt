package com.fada.ultrasound.llm

import android.content.Context
import android.graphics.Bitmap

data class LlmChatTurn(
    val role: LlmChatRole,
    val content: String,
    val imageFileName: String? = null
)

enum class LlmChatRole {
    User,
    Assistant
}

interface LlmResponseClient {
    suspend fun prepare(
        context: Context,
        model: LlmModelOption,
        onStatus: (String) -> Unit = {}
    )

    suspend fun generate(
        context: Context,
        model: LlmModelOption,
        conversationId: String,
        history: List<LlmChatTurn>,
        image: Bitmap?,
        imageFileName: String?,
        prompt: String,
        systemPrompt: String,
        onStatus: (String) -> Unit = {},
        onPartialResponse: (String) -> Unit = {}
    ): String

    fun releaseModel(modelId: String)

    fun release()
}
