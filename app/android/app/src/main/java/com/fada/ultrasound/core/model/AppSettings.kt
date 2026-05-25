package com.fada.ultrasound.core.model

data class AppSettings(
    val keepImageAfterSend: Boolean = true,
    val useDefaultSystemPrompt: Boolean = true,
    val customSystemPrompt: String = ""
) {
    val effectiveSystemPrompt: String
        get() = buildString {
            if (useDefaultSystemPrompt) {
                append(DEFAULT_SYSTEM_PROMPT)
            }
            if (customSystemPrompt.isNotBlank()) {
                if (isNotBlank()) {
                    append("\n\n")
                }
                append(customSystemPrompt.trim())
            }
        }

    companion object {
        const val DEFAULT_SYSTEM_PROMPT =
            "You are a helpful multimodal assistant. Answer the user's prompt directly. " +
                "If an ultrasound image is provided, describe visible anatomy, likely view, and uncertainty. " +
                "Do not provide clinical diagnosis."
    }
}
