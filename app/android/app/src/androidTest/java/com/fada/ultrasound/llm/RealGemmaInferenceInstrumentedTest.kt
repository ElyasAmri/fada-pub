package com.fada.ultrasound.llm

import android.content.Context
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class RealGemmaInferenceInstrumentedTest {

    @Test
    fun downloadsGemmaAndGeneratesResponse() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val sampleImage = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        sampleImage.eraseColor(0xFF000000.toInt())

        val response = LlmResponseGenerator.generate(
            context = context,
            model = LlmModels.default,
            conversationId = "test-1",
            history = emptyList(),
            image = sampleImage,
            imageFileName = "sample.png",
            prompt = "Describe this image.",
            systemPrompt = "Answer briefly."
        )

        assertTrue(response.isNotBlank())
    }

    @Test
    fun secondRunUsesCachedModelWithoutDownload() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val sampleImage = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        sampleImage.eraseColor(0xFF101010.toInt())

        // First run (may download if missing)
        LlmResponseGenerator.generate(
            context = context,
            model = LlmModels.default,
            conversationId = "test-2",
            history = emptyList(),
            image = sampleImage,
            imageFileName = "sample.png",
            prompt = "Describe this image.",
            systemPrompt = "Answer briefly."
        )

        val statuses = mutableListOf<String>()
        val secondResponse = LlmResponseGenerator.generate(
            context = context,
            model = LlmModels.default,
            conversationId = "test-3",
            history = emptyList(),
            image = sampleImage,
            imageFileName = "sample.png",
            prompt = "Describe this image.",
            systemPrompt = "Answer briefly.",
            onStatus = { statuses.add(it) }
        )

        assertTrue(secondResponse.isNotBlank())
        assertFalse(statuses.any { it.startsWith("Downloading") })
    }
}

