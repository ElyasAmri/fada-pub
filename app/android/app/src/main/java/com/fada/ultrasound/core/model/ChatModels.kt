package com.fada.ultrasound.core.model

enum class ChatRole {
    User,
    Assistant
}

data class ChatMessage(
    val id: String,
    val role: ChatRole,
    val content: String,
    val modelName: String? = null,
    val latencyMs: Long? = null,
    val hasImage: Boolean = false,
    val imageFileName: String? = null,
    val imagePath: String? = null,
    val isStreaming: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)

data class ChatConversation(
    val id: String,
    val title: String,
    val messages: List<ChatMessage>,
    val createdAt: Long,
    val updatedAt: Long
)
