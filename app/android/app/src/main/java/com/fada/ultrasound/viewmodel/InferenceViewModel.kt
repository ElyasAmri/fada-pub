package com.fada.ultrasound.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.fada.ultrasound.core.data.ConversationStore
import com.fada.ultrasound.core.model.AppSettings
import com.fada.ultrasound.core.model.ChatConversation
import com.fada.ultrasound.core.model.ChatMessage
import com.fada.ultrasound.core.model.ChatRole
import com.fada.ultrasound.core.model.LlmResponse
import com.fada.ultrasound.core.model.ModelStorageInfo
import com.fada.ultrasound.llm.LlmChatRole
import com.fada.ultrasound.llm.LlmChatTurn
import com.fada.ultrasound.llm.LlmModelOption
import com.fada.ultrasound.llm.LlmModels
import com.fada.ultrasound.llm.ModelDownloadCoordinator
import com.fada.ultrasound.llm.ModelDownloadManager
import com.fada.ultrasound.llm.ModelDownloadState
import com.fada.ultrasound.llm.LlmResponseClient
import com.fada.ultrasound.llm.LlmResponseGenerator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.UUID

class InferenceViewModel(
    application: Application,
    private val responseClient: LlmResponseClient = LlmResponseGenerator
) : AndroidViewModel(application) {

    private val conversationStore = ConversationStore(application)
    private val storedState = conversationStore.load()
    private val initialConversations = storedState
        ?.conversations
        ?.takeIf { it.isNotEmpty() }
        ?: listOf(createConversation())

    constructor(application: Application) : this(
        application = application,
        responseClient = LlmResponseGenerator
    )

    // UI state
    private val _uiState = MutableStateFlow<InferenceUiState>(InferenceUiState.Idle)
    val uiState: StateFlow<InferenceUiState> = _uiState.asStateFlow()

    // Selected image
    private val _selectedImage = MutableStateFlow<Bitmap?>(null)
    val selectedImage: StateFlow<Bitmap?> = _selectedImage.asStateFlow()

    private val _selectedImageFileName = MutableStateFlow<String?>(null)
    val selectedImageFileName: StateFlow<String?> = _selectedImageFileName.asStateFlow()

    private val _selectedModel = MutableStateFlow(
        LlmModels.options.firstOrNull { it.id == storedState?.selectedModelId } ?: LlmModels.default
    )
    val selectedModel: StateFlow<LlmModelOption> = _selectedModel.asStateFlow()

    private val _modelOptions = MutableStateFlow(LlmModels.options)
    val modelOptions: StateFlow<List<LlmModelOption>> = _modelOptions.asStateFlow()

    private val _llmResponse = MutableStateFlow<LlmResponse?>(null)
    val llmResponse: StateFlow<LlmResponse?> = _llmResponse.asStateFlow()

    private val _conversations = MutableStateFlow(initialConversations)
    val conversations: StateFlow<List<ChatConversation>> = _conversations.asStateFlow()

    private val _currentConversationId = MutableStateFlow(
        storedState?.currentConversationId
            ?.takeIf { storedId -> initialConversations.any { it.id == storedId } }
            ?: initialConversations.first().id
    )
    val currentConversationId: StateFlow<String> = _currentConversationId.asStateFlow()

    private val _modelStorage = MutableStateFlow<List<ModelStorageInfo>>(emptyList())
    val modelStorage: StateFlow<List<ModelStorageInfo>> = _modelStorage.asStateFlow()

    private val _settings = MutableStateFlow(storedState?.settings ?: AppSettings())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private var downloadStates: Map<String, ModelDownloadState> = ModelDownloadCoordinator.states.value

    init {
        refreshModelStorage()
        viewModelScope.launch {
            ModelDownloadCoordinator.states.collect { states ->
                downloadStates = states
                refreshModelStorage()
            }
        }
        persistState()
    }

    /**
     * Load an image from URI (gallery picker).
     */
    fun loadImageFromUri(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = InferenceUiState.Loading("Loading image...")

            try {
                val inputStream: InputStream? = getApplication<Application>()
                    .contentResolver
                    .openInputStream(uri)

                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap != null) {
                    withContext(Dispatchers.Main) {
                        replaceSelectedImage(bitmap, resolveDisplayName(uri))
                        _llmResponse.value = null
                        _uiState.value = InferenceUiState.ImageLoaded
                    }
                } else {
                    _uiState.value = InferenceUiState.Error("Failed to decode image")
                }
            } catch (e: Exception) {
                _uiState.value = InferenceUiState.Error("Failed to load image: ${e.message}")
            }
        }
    }

    /**
     * Set a captured image from camera.
     */
    fun setCapturedImage(bitmap: Bitmap) {
        replaceSelectedImage(bitmap, "Camera image")
        _llmResponse.value = null
        _uiState.value = InferenceUiState.ImageLoaded
    }

    /**
     * Select the current multimodal model.
     */
    fun selectModel(modelId: String) {
        val model = _modelOptions.value.firstOrNull { it.id == modelId } ?: return
        val previousModelId = _selectedModel.value.id
        _selectedModel.value = model
        if (previousModelId != model.id) {
            viewModelScope.launch(Dispatchers.IO) {
                responseClient.releaseModel(previousModelId)
            }
            _uiState.value = InferenceUiState.Idle
        }
        persistState()
    }

    fun createNewConversation() {
        val conversation = createConversation()
        _conversations.value = listOf(conversation) + _conversations.value
        _currentConversationId.value = conversation.id
        persistState()
    }

    fun selectConversation(conversationId: String) {
        if (_conversations.value.any { it.id == conversationId }) {
            _currentConversationId.value = conversationId
            persistState()
        }
    }

    fun deleteConversation(conversationId: String) {
        deleteConversations(setOf(conversationId))
    }

    fun deleteConversations(conversationIds: Set<String>) {
        if (conversationIds.isEmpty()) return
        val remaining = _conversations.value.filterNot { it.id in conversationIds }
        _conversations.value = remaining.ifEmpty { listOf(createConversation()) }
        if (_currentConversationId.value in conversationIds) {
            _currentConversationId.value = _conversations.value.first().id
        }
        persistState()
    }

    fun clearCurrentConversation() {
        val currentId = _currentConversationId.value
        _conversations.value = _conversations.value.map { conversation ->
            if (conversation.id == currentId) {
                conversation.copy(
                    title = "New conversation",
                    messages = emptyList(),
                    updatedAt = System.currentTimeMillis()
                )
            } else {
                conversation
            }
        }
        _llmResponse.value = null
        _uiState.value = InferenceUiState.Idle
        persistState()
    }

    fun clearAllConversations() {
        val conversation = createConversation()
        _conversations.value = listOf(conversation)
        _currentConversationId.value = conversation.id
        _llmResponse.value = null
        _uiState.value = InferenceUiState.Idle
        persistState()
    }

    fun updateKeepImageAfterSend(keepImageAfterSend: Boolean) {
        _settings.value = _settings.value.copy(keepImageAfterSend = keepImageAfterSend)
        persistState()
    }

    fun updateUseDefaultSystemPrompt(enabled: Boolean) {
        _settings.value = _settings.value.copy(useDefaultSystemPrompt = enabled)
        persistState()
    }

    fun updateCustomSystemPrompt(systemPrompt: String) {
        _settings.value = _settings.value.copy(customSystemPrompt = systemPrompt)
        persistState()
    }

    /**
     * Generate a multimodal response on the selected image.
     */
    fun generateResponseForCurrentImage() {
        sendChatMessage(DEFAULT_IMAGE_PROMPT)
    }

    /**
     * Send a chat prompt with the selected image as visual context.
     */
    fun sendChatMessage(prompt: String) {
        val normalizedPrompt = prompt.trim()
        if (normalizedPrompt.isBlank()) {
            _uiState.value = InferenceUiState.Error("Enter a prompt")
            return
        }

        val bitmap = _selectedImage.value

        val currentId = _currentConversationId.value
        val imageFileName = _selectedImageFileName.value
        val history = currentConversationHistory(currentId)
        val selected = _selectedModel.value
        if (!ModelDownloadManager(getApplication()).getStoredModelInfo(selected).isStored) {
            _uiState.value = InferenceUiState.Error(
                "Download ${selected.displayName} from Models before chatting."
            )
            return
        }

        val userMessageId = UUID.randomUUID().toString()
        val chatImagePath = bitmap?.let { saveChatImage(it, userMessageId) }
        appendMessage(
            conversationId = currentId,
            message = ChatMessage(
                id = userMessageId,
                role = ChatRole.User,
                content = normalizedPrompt,
                hasImage = bitmap != null,
                imageFileName = if (bitmap != null) imageFileName ?: "Attached image" else null,
                imagePath = chatImagePath
            )
        )
        if (bitmap != null) {
            clearSelectedImageReference()
        }

        val assistantMessageId = UUID.randomUUID().toString()

        viewModelScope.launch(Dispatchers.Default) {
            _uiState.value = InferenceUiState.GeneratingResponse
            val startTime = System.currentTimeMillis()
            var assistantMessageCreated = false
            try {
                val output = responseClient.generate(
                    context = getApplication(),
                    model = selected,
                    conversationId = currentId,
                    history = history,
                    image = bitmap,
                    imageFileName = imageFileName,
                    prompt = normalizedPrompt,
                    systemPrompt = _settings.value.effectiveSystemPrompt,
                    onStatus = { status ->
                        _uiState.value = InferenceUiState.Loading(status)
                    },
                    onPartialResponse = { partial ->
                        if (!assistantMessageCreated) {
                            appendMessage(
                                conversationId = currentId,
                                message = ChatMessage(
                                    id = assistantMessageId,
                                    role = ChatRole.Assistant,
                                    content = partial,
                                    modelName = selected.displayName,
                                    isStreaming = true
                                )
                            )
                            assistantMessageCreated = true
                        } else {
                            updateMessageContent(
                                conversationId = currentId,
                                messageId = assistantMessageId,
                                content = partial,
                                isStreaming = true
                            )
                        }
                    }
                )
                val elapsed = System.currentTimeMillis() - startTime

                withContext(Dispatchers.Main) {
                    val response = LlmResponse(
                        modelId = selected.id,
                        modelName = selected.displayName,
                        content = output,
                        latencyMs = elapsed
                    )
                    _llmResponse.value = response
                    if (assistantMessageCreated) {
                        updateMessageContent(
                            conversationId = currentId,
                            messageId = assistantMessageId,
                            content = output,
                            latencyMs = elapsed,
                            isStreaming = false
                        )
                    } else {
                        appendMessage(
                            conversationId = currentId,
                            message = ChatMessage(
                                id = assistantMessageId,
                                role = ChatRole.Assistant,
                                content = output,
                                modelName = selected.displayName,
                                latencyMs = elapsed,
                                isStreaming = false
                            )
                        )
                    }
                    _uiState.value = InferenceUiState.ResponseReady(response)
                }
            } catch (e: IllegalStateException) {
                val message = e.message ?: "Model runtime failed"
                removeMessages(currentId, setOf(userMessageId, assistantMessageId))
                _uiState.value = InferenceUiState.Error(message)
            } catch (e: RuntimeException) {
                val message = e.message ?: "Inference failed"
                removeMessages(currentId, setOf(userMessageId, assistantMessageId))
                _uiState.value = InferenceUiState.Error(message)
            }
        }
    }

    fun refreshModelStorage() {
        val manager = ModelDownloadManager(getApplication())
        _modelStorage.value = _modelOptions.value.map { option ->
            val info = manager.getStoredModelInfo(option)
            val downloadState = downloadStates[option.id]
            ModelStorageInfo(
                model = option,
                isStored = info.isStored,
                sizeBytes = info.sizeBytes,
                filePath = info.file.absolutePath,
                status = downloadState?.status,
                isBusy = downloadState?.isBusy == true
            )
        }
    }

    fun downloadModel(modelId: String) {
        val model = _modelOptions.value.firstOrNull { it.id == modelId } ?: return
        ModelDownloadCoordinator.startDownload(getApplication(), model)
    }

    fun deleteStoredModel(modelId: String) {
        val model = _modelOptions.value.firstOrNull { it.id == modelId } ?: return
        viewModelScope.launch(Dispatchers.IO) {
            responseClient.releaseModel(modelId)
            ModelDownloadManager(getApplication()).invalidateModel(model)
            withContext(Dispatchers.Main) {
                refreshModelStorage()
            }
        }
    }

    fun deleteStoredModels(modelIds: Set<String>) {
        modelIds.forEach { deleteStoredModel(it) }
    }

    fun clearUnusedModelFiles() {
        if (downloadStates.values.any { it.isBusy }) {
            _uiState.value = InferenceUiState.Error("Wait for model downloads to finish before clearing old files")
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            val result = ModelDownloadManager(getApplication()).clearUnusedModelFiles(_modelOptions.value)
            withContext(Dispatchers.Main) {
                refreshModelStorage()
                _uiState.value = InferenceUiState.Loading(
                    "Cleared ${result.deletedFiles} old model file${if (result.deletedFiles == 1) "" else "s"}"
                )
            }
        }
    }

    /**
     * Clear the current selection and results.
     */
    fun clearSelection() {
        replaceSelectedImage(null, null)
        _llmResponse.value = null
        _uiState.value = InferenceUiState.Idle
    }

    override fun onCleared() {
        super.onCleared()
        replaceSelectedImage(null, null)
        responseClient.release()
    }

    private fun replaceSelectedImage(bitmap: Bitmap?, fileName: String?) {
        val current = _selectedImage.value
        if (current !== bitmap) {
            current?.recycle()
        }
        _selectedImage.value = bitmap
        _selectedImageFileName.value = fileName
    }

    private fun clearSelectedImageReference() {
        _selectedImage.value = null
        _selectedImageFileName.value = null
    }

    private fun saveChatImage(bitmap: Bitmap, messageId: String): String? {
        return runCatching {
            val directory = File(getApplication<Application>().filesDir, "chat_images").apply {
                mkdirs()
            }
            val file = File(directory, "$messageId.jpg")
            FileOutputStream(file).use { output ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, output)
            }
            file.absolutePath
        }.getOrNull()
    }

    private fun resolveDisplayName(uri: Uri): String {
        val resolver = getApplication<Application>().contentResolver
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (nameIndex >= 0 && cursor.moveToFirst()) {
                val name = cursor.getString(nameIndex)
                if (!name.isNullOrBlank()) {
                    return name
                }
            }
        }
        return uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.isNotBlank() } ?: "Selected image"
    }

    private fun appendMessage(conversationId: String, message: ChatMessage) {
        _conversations.value = _conversations.value.map { conversation ->
            if (conversation.id == conversationId) {
                val messages = conversation.messages + message
                conversation.copy(
                    title = conversationTitle(messages),
                    messages = messages,
                    updatedAt = System.currentTimeMillis()
                )
            } else {
                conversation
            }
        }.sortedByDescending { it.updatedAt }
        persistState()
    }

    private fun updateMessageContent(
        conversationId: String,
        messageId: String,
        content: String,
        latencyMs: Long? = null,
        isStreaming: Boolean? = null
    ) {
        _conversations.value = _conversations.value.map { conversation ->
            if (conversation.id == conversationId) {
                conversation.copy(
                    messages = conversation.messages.map { message ->
                        if (message.id == messageId) {
                            message.copy(
                                content = content,
                                latencyMs = latencyMs ?: message.latencyMs,
                                isStreaming = isStreaming ?: message.isStreaming
                            )
                        } else {
                            message
                        }
                    },
                    updatedAt = System.currentTimeMillis()
                )
            } else {
                conversation
            }
        }.sortedByDescending { it.updatedAt }
        persistState()
    }

    private fun removeMessages(conversationId: String, messageIds: Set<String>) {
        _conversations.value = _conversations.value.map { conversation ->
            if (conversation.id == conversationId) {
                val messages = conversation.messages.filterNot { it.id in messageIds }
                conversation.copy(
                    title = conversationTitle(messages),
                    messages = messages,
                    updatedAt = System.currentTimeMillis()
                )
            } else {
                conversation
            }
        }.sortedByDescending { it.updatedAt }
        persistState()
    }

    private fun markModelBusy(modelId: String, status: String) {
        _modelStorage.value = _modelStorage.value.map { info ->
            if (info.model.id == modelId) {
                info.copy(isBusy = true, status = status)
            } else {
                info
            }
        }
    }

    private fun markModelIdleWithError(modelId: String, message: String) {
        _modelStorage.value = _modelStorage.value.map { info ->
            if (info.model.id == modelId) {
                info.copy(isBusy = false, status = message)
            } else {
                info
            }
        }
        _uiState.value = InferenceUiState.Error(message)
    }

    private fun conversationTitle(messages: List<ChatMessage>): String {
        val firstUserMessage = messages.firstOrNull { it.role == ChatRole.User }?.content
        return firstUserMessage
            ?.take(42)
            ?.ifBlank { null }
            ?: "New conversation"
    }

    private fun createConversation(): ChatConversation {
        return ChatConversation(
            id = UUID.randomUUID().toString(),
            title = "New conversation",
            messages = emptyList(),
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis()
        )
    }

    private fun currentConversationHistory(conversationId: String): List<LlmChatTurn> {
        return _conversations.value
            .firstOrNull { it.id == conversationId }
            ?.messages
            .orEmpty()
            .filter { it.content.isNotBlank() && !it.isStreaming }
            .map { message ->
                LlmChatTurn(
                    role = when (message.role) {
                        ChatRole.User -> LlmChatRole.User
                        ChatRole.Assistant -> LlmChatRole.Assistant
                    },
                    content = message.content,
                    imageFileName = message.imageFileName
                )
            }
    }

    private fun persistState() {
        conversationStore.save(
            conversations = _conversations.value,
            currentConversationId = _currentConversationId.value,
            selectedModelId = _selectedModel.value.id,
            settings = _settings.value
        )
    }

    companion object {
        private const val DEFAULT_IMAGE_PROMPT =
            "Describe the visible anatomy, likely image view, and important uncertainty."
    }
}

/**
 * UI state for inference operations.
 */
sealed class InferenceUiState {
    data object Idle : InferenceUiState()
    data class Loading(val message: String) : InferenceUiState()
    data object ImageLoaded : InferenceUiState()
    data object GeneratingResponse : InferenceUiState()
    data class ResponseReady(val response: LlmResponse) : InferenceUiState()
    data class Error(val message: String) : InferenceUiState()
}
