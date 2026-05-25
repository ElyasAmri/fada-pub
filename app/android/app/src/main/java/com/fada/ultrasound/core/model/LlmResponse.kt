package com.fada.ultrasound.core.model

data class LlmResponse(
    val modelId: String,
    val modelName: String,
    val content: String,
    val latencyMs: Long
)
