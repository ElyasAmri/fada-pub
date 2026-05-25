package com.fada.ultrasound.llm

/**
 * Supported on-device Gemma variants in LiteRT-LM format.
 */
data class LlmModelOption(
    val id: String,
    val displayName: String,
    val version: String,
    val downloadUrl: String,
    val expectedSha256: String? = null,
    val localFileName: String
)

object LlmModels {
    val default = LlmModelOption(
        id = "gemma-4-e2b",
        displayName = "Gemma 4 E2B",
        version = "1",
        downloadUrl = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
        localFileName = "gemma-4-E2B-it.litertlm"
    )

    val options = listOf(
        default,
        LlmModelOption(
            id = "gemma-4-e4b",
            displayName = "Gemma 4 E4B",
            version = "1",
            downloadUrl = "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
            localFileName = "gemma-4-E4B-it.litertlm"
        ),
        LlmModelOption(
            id = "gemma-4-e2b-fada",
            displayName = "Gemma 4 E2B FADA",
            version = "20260501-gemma4-hybrid",
            downloadUrl = "https://huggingface.co/elyasamri/gemma-4-e2b-fada-litertlm/resolve/main/gemma-4-E2B-fada-ft.litertlm",
            expectedSha256 = "1e47115687efe6afd36566ad0576790e221f070b9be8cc23f3ae013a34363624",
            localFileName = "gemma-4-E2B-fada-ft.litertlm"
        )
    )
}

