package com.fada.ultrasound.ui

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Forum
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.fada.ultrasound.core.model.ChatConversation
import com.fada.ultrasound.viewmodel.InferenceViewModel
import java.text.DateFormat
import java.util.Date

@Composable
fun ConversationsScreen(
    viewModel: InferenceViewModel,
    selectedConversationIds: Set<String>,
    onToggleConversationSelection: (String) -> Unit,
    onOpenConversation: () -> Unit
) {
    val conversations by viewModel.conversations.collectAsState()
    val currentConversationId by viewModel.currentConversationId.collectAsState()
    var searchQuery by remember { mutableStateOf("") }
    val visibleConversations = remember(conversations) {
        conversations.filter { it.messages.isNotEmpty() }
    }
    val filteredConversations = remember(visibleConversations, searchQuery) {
        val query = searchQuery.trim()
        if (query.isBlank()) {
            visibleConversations
        } else {
            visibleConversations.filter { conversation ->
                conversation.title.contains(query, ignoreCase = true) ||
                    conversation.messages.any { it.content.contains(query, ignoreCase = true) }
            }
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        TextField(
            value = searchQuery,
            onValueChange = { searchQuery = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp)
                .testTag(FadaTestTags.THREADS_SEARCH),
            singleLine = true,
            shape = CircleShape,
            colors = TextFieldDefaults.colors(
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                disabledIndicatorColor = Color.Transparent,
                focusedContainerColor = MaterialTheme.colorScheme.surfaceVariant,
                unfocusedContainerColor = MaterialTheme.colorScheme.surfaceVariant
            ),
            leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
            placeholder = { Text("Search threads") }
        )

        if (filteredConversations.isEmpty()) {
            Text(
                text = if (searchQuery.isBlank()) {
                    "Create a new conversation with the + button."
                } else {
                    "No matching threads"
                },
                modifier = Modifier.padding(24.dp),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .testTag(FadaTestTags.THREADS_LIST)
            ) {
                items(
                    items = filteredConversations,
                    key = { it.id }
                ) { conversation ->
                    ConversationRow(
                        conversation = conversation,
                        isCurrent = conversation.id == currentConversationId,
                        isSelectionMode = selectedConversationIds.isNotEmpty(),
                        isSelected = conversation.id in selectedConversationIds,
                        onSelect = {
                            if (selectedConversationIds.isNotEmpty()) {
                                onToggleConversationSelection(conversation.id)
                            } else {
                                viewModel.selectConversation(conversation.id)
                                onOpenConversation()
                            }
                        },
                        onLongSelect = {
                            onToggleConversationSelection(conversation.id)
                        }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ConversationRow(
    conversation: ChatConversation,
    isCurrent: Boolean,
    isSelectionMode: Boolean,
    isSelected: Boolean,
    onSelect: () -> Unit,
    onLongSelect: () -> Unit
) {
    val lastMessage = conversation.messages.lastOrNull()?.content ?: "No messages yet"
    val createdAt = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT)
        .format(Date(conversation.createdAt))
    val backgroundColor = when {
        isSelected -> MaterialTheme.colorScheme.primaryContainer
        isCurrent && !isSelectionMode -> MaterialTheme.colorScheme.secondaryContainer
        else -> MaterialTheme.colorScheme.surface
    }
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(backgroundColor)
            .testTag(FadaTestTags.THREAD_ROW_PREFIX + conversation.id)
            .combinedClickable(
                onClick = onSelect,
                onLongClick = onLongSelect
            )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.Forum,
                contentDescription = null,
                modifier = Modifier.padding(end = 12.dp),
                tint = if (isSelected) {
                    MaterialTheme.colorScheme.onPrimaryContainer
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                }
            )
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = conversation.title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = lastMessage.take(110),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = "$createdAt - ${conversation.messages.size} message${if (conversation.messages.size == 1) "" else "s"}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
        HorizontalDivider(
            color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.45f)
        )
    }
}
