package com.fada.ultrasound.core.model

import com.fada.ultrasound.llm.LlmModelOption

data class ModelStorageInfo(
    val model: LlmModelOption,
    val isStored: Boolean,
    val sizeBytes: Long,
    val filePath: String,
    val status: String? = null,
    val isBusy: Boolean = false
)
