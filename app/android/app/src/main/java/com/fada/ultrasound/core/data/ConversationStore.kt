package com.fada.ultrasound.core.data

import android.content.Context
import com.fada.ultrasound.core.model.AppSettings
import com.fada.ultrasound.core.model.ChatConversation
import com.fada.ultrasound.core.model.ChatMessage
import com.fada.ultrasound.core.model.ChatRole
import org.json.JSONArray
import org.json.JSONObject

data class StoredConversationState(
    val conversations: List<ChatConversation>,
    val currentConversationId: String?,
    val selectedModelId: String?,
    val settings: AppSettings
)

class ConversationStore(context: Context) {
    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun load(): StoredConversationState? {
        val raw = prefs.getString(KEY_STATE, null) ?: return null
        val root = JSONObject(raw)
        val conversations = root.getJSONArray("conversations").let { array ->
            List(array.length()) { index ->
                val item = array.getJSONObject(index)
                ChatConversation(
                    id = item.getString("id"),
                    title = item.getString("title"),
                    messages = item.getJSONArray("messages").let { messages ->
                        List(messages.length()) { messageIndex ->
                            val message = messages.getJSONObject(messageIndex)
                            ChatMessage(
                                id = message.getString("id"),
                                role = ChatRole.valueOf(message.getString("role")),
                                content = message.getString("content"),
                                modelName = message.optString("modelName").takeIf { it.isNotBlank() },
                                 latencyMs = if (message.has("latencyMs")) message.getLong("latencyMs") else null,
                                 hasImage = message.optBoolean("hasImage", false),
                                 imageFileName = message.optString("imageFileName").takeIf { it.isNotBlank() },
                                 imagePath = message.optString("imagePath").takeIf { it.isNotBlank() },
                                 isStreaming = false,
                                 createdAt = message.optLong("createdAt", System.currentTimeMillis())
                             )
                        }
                    },
                    createdAt = item.optLong("createdAt", System.currentTimeMillis()),
                    updatedAt = item.optLong("updatedAt", System.currentTimeMillis())
                )
            }
        }

        return StoredConversationState(
            conversations = conversations,
            currentConversationId = root.optString("currentConversationId").takeIf { it.isNotBlank() },
            selectedModelId = root.optString("selectedModelId").takeIf { it.isNotBlank() },
            settings = root.optJSONObject("settings")?.let { settings ->
                val legacyPrompt = settings.optString("systemPrompt").takeIf { it.isNotBlank() }
                AppSettings(
                    keepImageAfterSend = settings.optBoolean("keepImageAfterSend", true),
                    useDefaultSystemPrompt = settings.optBoolean(
                        "useDefaultSystemPrompt",
                        legacyPrompt == null || legacyPrompt == AppSettings.DEFAULT_SYSTEM_PROMPT
                    ),
                    customSystemPrompt = settings.optString("customSystemPrompt").takeIf { it.isNotBlank() }
                        ?: legacyPrompt?.takeIf { it != AppSettings.DEFAULT_SYSTEM_PROMPT }
                        ?: ""
                )
            } ?: AppSettings()
        )
    }

    fun save(
        conversations: List<ChatConversation>,
        currentConversationId: String,
        selectedModelId: String,
        settings: AppSettings
    ) {
        val root = JSONObject()
            .put("currentConversationId", currentConversationId)
            .put("selectedModelId", selectedModelId)
            .put(
                "settings",
                JSONObject()
                    .put("keepImageAfterSend", settings.keepImageAfterSend)
                    .put("useDefaultSystemPrompt", settings.useDefaultSystemPrompt)
                    .put("customSystemPrompt", settings.customSystemPrompt)
            )
            .put(
                "conversations",
                JSONArray().apply {
                    conversations.forEach { conversation ->
                        put(
                            JSONObject()
                                .put("id", conversation.id)
                                .put("title", conversation.title)
                                .put("createdAt", conversation.createdAt)
                                .put("updatedAt", conversation.updatedAt)
                                .put(
                                    "messages",
                                    JSONArray().apply {
                                        conversation.messages.forEach { message ->
                                            put(
                                                JSONObject()
                                                    .put("id", message.id)
                                                    .put("role", message.role.name)
                                                    .put("content", message.content)
                                                     .put("modelName", message.modelName ?: "")
                                                     .put("hasImage", message.hasImage)
                                                     .put("imageFileName", message.imageFileName ?: "")
                                                     .put("imagePath", message.imagePath ?: "")
                                                     .put("createdAt", message.createdAt)
                                                     .apply {
                                                        message.latencyMs?.let { put("latencyMs", it) }
                                                    }
                                            )
                                        }
                                    }
                                )
                        )
                    }
                }
            )

        prefs.edit().putString(KEY_STATE, root.toString()).apply()
    }

    companion object {
        private const val PREFS_NAME = "fada_conversations"
        private const val KEY_STATE = "state"
    }
}
