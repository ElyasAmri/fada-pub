package com.fada.ultrasound.viewmodel

import android.app.Application
import android.graphics.Bitmap
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.fada.ultrasound.llm.LlmChatTurn
import com.fada.ultrasound.llm.LlmModelOption
import com.fada.ultrasound.llm.LlmResponseClient
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class InferenceViewModelInstrumentedTest {

    @Test
    fun generateResponseForSampleImage_returnsResponseReady() {
        val application = ApplicationProvider.getApplicationContext<Application>()
        val fakeClient = object : LlmResponseClient {
            override suspend fun prepare(
                context: android.content.Context,
                model: LlmModelOption,
                onStatus: (String) -> Unit
            ) = Unit

            override suspend fun generate(
                context: android.content.Context,
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
                onStatus("Fake model running...")
                requireNotNull(image)
                val response = "mock-response:${model.id}:${image.width}x${image.height}"
                onPartialResponse(response)
                return response
            }

            override fun release() = Unit

            override fun releaseModel(modelId: String) = Unit
        }
        val viewModel = InferenceViewModel(application, fakeClient)
        val sampleImage = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)

        viewModel.selectModel("gemma-4-e2b")
        viewModel.setCapturedImage(sampleImage)
        viewModel.generateResponseForCurrentImage()

        val finalState = waitForTerminalState(viewModel)
        assertTrue(finalState is InferenceUiState.ResponseReady)

        val response = viewModel.llmResponse.value
        requireNotNull(response)
        assertEquals("gemma-4-e2b", response.modelId)
        assertTrue(response.content.contains("mock-response:gemma-4-e2b:64x64"))
    }

    @Test
    fun generateResponseWithoutImage_returnsResponseReady() {
        val application = ApplicationProvider.getApplicationContext<Application>()
        val fakeClient = object : LlmResponseClient {
            override suspend fun prepare(
                context: android.content.Context,
                model: LlmModelOption,
                onStatus: (String) -> Unit
            ) = Unit

            override suspend fun generate(
                context: android.content.Context,
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
                val response = "text-only:$prompt"
                onPartialResponse(response)
                return response
            }

            override fun release() = Unit

            override fun releaseModel(modelId: String) = Unit
        }
        val viewModel = InferenceViewModel(application, fakeClient)

        viewModel.generateResponseForCurrentImage()

        val state = waitForTerminalState(viewModel)
        assertTrue(state is InferenceUiState.ResponseReady)
    }

    private fun waitForTerminalState(
        viewModel: InferenceViewModel,
        timeoutMs: Long = 5_000L
    ): InferenceUiState {
        val start = System.currentTimeMillis()
        while (System.currentTimeMillis() - start < timeoutMs) {
            val state = viewModel.uiState.value
            if (state is InferenceUiState.ResponseReady || state is InferenceUiState.Error) {
                return state
            }
            Thread.sleep(50)
        }
        return viewModel.uiState.value
    }
}

