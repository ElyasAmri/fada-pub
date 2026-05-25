package com.fada.ultrasound.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.fada.ultrasound.viewmodel.InferenceViewModel

@Composable
fun SettingsScreen(
    viewModel: InferenceViewModel,
    onNavigateToModels: () -> Unit,
    onNavigateToSystemPrompt: () -> Unit
) {
    val settings by viewModel.settings.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val conversations by viewModel.conversations.collectAsState()
    val modelStorage by viewModel.modelStorage.collectAsState()
    val storedModels = modelStorage.count { it.isStored }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
    ) {
        SectionHeader("Chat")
        SettingsRow(
            icon = { Icon(Icons.Default.Image, contentDescription = null) },
            title = "Keep image after send",
            subtitle = "Leave the current image attached for follow-up prompts",
            trailing = {
                Switch(
                    checked = settings.keepImageAfterSend,
                    onCheckedChange = viewModel::updateKeepImageAfterSend
                )
            }
        )
        SettingsRow(
            icon = { Icon(Icons.Default.Info, contentDescription = null) },
            title = "System prompt",
            subtitle = systemPromptSummary(settings.useDefaultSystemPrompt, settings.customSystemPrompt),
            onClick = onNavigateToSystemPrompt
        )

        SectionHeader("Models")
        SettingsRow(
            icon = { Icon(Icons.Default.Storage, contentDescription = null) },
            title = "Models",
            subtitle = "${selectedModel.displayName} - $storedModels stored",
            onClick = onNavigateToModels
        )

        SectionHeader("Data")
        SettingsRow(
            icon = { Icon(Icons.Default.Info, contentDescription = null) },
            title = "Conversations",
            subtitle = "${conversations.size} local thread${if (conversations.size == 1) "" else "s"}"
        )
        SettingsRow(
            icon = { Icon(Icons.Default.Delete, contentDescription = null) },
            title = "Clear current conversation",
            subtitle = "Remove messages in the active thread",
            onClick = { viewModel.clearCurrentConversation() }
        )
        SettingsRow(
            icon = { Icon(Icons.Default.Delete, contentDescription = null) },
            title = "Clear all conversations",
            subtitle = "Reset local chat history",
            onClick = { viewModel.clearAllConversations() }
        )

        Spacer(modifier = Modifier.height(24.dp))
    }
}

private fun systemPromptSummary(useDefault: Boolean, customPrompt: String): String {
    return when {
        useDefault && customPrompt.isNotBlank() -> "Default prompt plus custom instructions"
        useDefault -> "Default prompt enabled"
        customPrompt.isNotBlank() -> "Custom prompt only"
        else -> "No system prompt"
    }
}

@Composable
private fun SectionHeader(title: String) {
    Text(
        text = title,
        modifier = Modifier.padding(start = 24.dp, end = 24.dp, top = 24.dp, bottom = 8.dp),
        style = MaterialTheme.typography.labelLarge,
        color = MaterialTheme.colorScheme.primary,
        fontWeight = FontWeight.SemiBold
    )
}

@Composable
private fun SettingsRow(
    icon: @Composable () -> Unit,
    title: String,
    subtitle: String,
    trailing: @Composable (() -> Unit)? = null,
    onClick: (() -> Unit)? = null
) {
    Column {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier)
                .padding(horizontal = 24.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            Row(
                modifier = Modifier.size(28.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                icon()
            }
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyLarge
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            when {
                trailing != null -> trailing()
                onClick != null -> Icon(Icons.Default.ChevronRight, contentDescription = null)
            }
        }
        HorizontalDivider(modifier = Modifier.padding(start = 72.dp))
    }
}
