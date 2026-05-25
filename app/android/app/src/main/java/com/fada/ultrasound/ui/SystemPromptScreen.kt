package com.fada.ultrasound.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.fada.ultrasound.core.model.AppSettings
import com.fada.ultrasound.viewmodel.InferenceViewModel

@Composable
fun SystemPromptScreen(viewModel: InferenceViewModel) {
    val settings by viewModel.settings.collectAsState()
    var draftPrompt by remember(settings.customSystemPrompt) {
        mutableStateOf(settings.customSystemPrompt)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(vertical = 16.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Use default system prompt",
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "Adds FADA's baseline instruction before any custom prompt.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Switch(
                checked = settings.useDefaultSystemPrompt,
                onCheckedChange = viewModel::updateUseDefaultSystemPrompt
            )
        }

        HorizontalDivider()

        Column(
            modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Custom system prompt",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold
            )
            OutlinedTextField(
                value = draftPrompt,
                onValueChange = { draftPrompt = it },
                modifier = Modifier.fillMaxWidth(),
                minLines = 7,
                maxLines = 14,
                label = { Text("Additional instruction") },
                placeholder = { Text("Leave blank to use only the default prompt, or no prompt if default is disabled.") }
            )
            Text(
                text = activePromptSummary(settings.useDefaultSystemPrompt, settings.customSystemPrompt),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                TextButton(
                    onClick = {
                        viewModel.updateUseDefaultSystemPrompt(true)
                        viewModel.updateCustomSystemPrompt("")
                        draftPrompt = ""
                    }
                ) {
                    Text("Reset")
                }
                TextButton(
                    onClick = { viewModel.updateCustomSystemPrompt(draftPrompt) },
                    enabled = draftPrompt != settings.customSystemPrompt
                ) {
                    Text("Save")
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = AppSettings.DEFAULT_SYSTEM_PROMPT,
            modifier = Modifier.padding(horizontal = 24.dp),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun activePromptSummary(useDefault: Boolean, customPrompt: String): String {
    return when {
        useDefault && customPrompt.isNotBlank() -> "Active: default prompt plus custom instructions."
        useDefault -> "Active: default prompt."
        customPrompt.isNotBlank() -> "Active: custom prompt only."
        else -> "Active: no system prompt."
    }
}
