package com.fada.ultrasound.llm

import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import kotlinx.coroutines.flow.collect

class LlmModelRuntime(
    private val engine: Engine,
    private val contentFactory: LlmContentFactory
) : AutoCloseable {
    private data class ConversationHandle(
        val conversationId: String,
        val systemPrompt: String,
        val conversation: Conversation
    )

    private var activeConversation: ConversationHandle? = null

    suspend fun generate(
        conversationId: String,
        history: List<LlmChatTurn>,
        systemPrompt: String,
        contents: Contents,
        onPartialResponse: (String) -> Unit
    ): String {
        val conversation = getOrCreateConversation(conversationId, history, systemPrompt)
        var accumulated = ""
        conversation.sendMessageAsync(contents).collect { message ->
            accumulated = StreamingTextAccumulator.append(accumulated, message.toString())
            onPartialResponse(accumulated)
        }
        return accumulated
    }

    private fun getOrCreateConversation(
        conversationId: String,
        history: List<LlmChatTurn>,
        systemPrompt: String
    ): Conversation {
        activeConversation?.let { handle ->
            if (handle.conversationId == conversationId && handle.systemPrompt == systemPrompt) {
                return handle.conversation
            }
            handle.conversation.close()
            activeConversation = null
        }
        val config = ConversationConfig(
            systemInstruction = Contents.of(systemPrompt),
            initialMessages = contentFactory.historyToInitialMessages(history)
        )
        return engine.createConversation(config).also {
            activeConversation = ConversationHandle(conversationId, systemPrompt, it)
        }
    }

    override fun close() {
        activeConversation?.conversation?.close()
        activeConversation = null
        engine.close()
    }
}
