package com.fada.ultrasound.llm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LlmModelsTest {

    @Test
    fun defaultModelIsIncludedInOptions() {
        assertTrue(LlmModels.options.any { it.id == LlmModels.default.id })
    }

    @Test
    fun allModelIdsAreUnique() {
        val ids = LlmModels.options.map { it.id }
        assertEquals(ids.size, ids.toSet().size)
    }

    @Test
    fun allModelsUseLitertlmFileName() {
        assertTrue(LlmModels.options.all { it.localFileName.endsWith(".litertlm") })
    }

    @Test
    fun allModelsHaveHttpsDownloadUrls() {
        assertTrue(LlmModels.options.all { it.downloadUrl.startsWith("https://") })
    }
}

