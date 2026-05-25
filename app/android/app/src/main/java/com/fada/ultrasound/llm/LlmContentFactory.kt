package com.fada.ultrasound.llm

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Message
import java.io.File
import java.io.FileOutputStream

class LlmContentFactory {
    fun createCurrentContents(
        context: Context,
        image: Bitmap?,
        imageFileName: String?,
        prompt: String
    ): PreparedContents {
        val imageFile = image?.toTempImageFile(context, imageFileName)
        val contents = if (imageFile != null) {
            Contents.of(
                Content.ImageFile(imageFile.absolutePath),
                Content.Text(prompt)
            )
        } else {
            Contents.of(prompt)
        }
        return PreparedContents(contents = contents, tempFile = imageFile)
    }

    fun historyToInitialMessages(history: List<LlmChatTurn>): List<Message> {
        return history
            .filter { it.content.isNotBlank() }
            .map { turn ->
                val text = buildString {
                    if (!turn.imageFileName.isNullOrBlank()) {
                        append("[Image attached: ")
                        append(turn.imageFileName)
                        append("]\n")
                    }
                    append(turn.content)
                }
                when (turn.role) {
                    LlmChatRole.User -> Message.user(text)
                    LlmChatRole.Assistant -> Message.model(text)
                }
            }
    }

    private fun Bitmap.toTempImageFile(context: Context, fileName: String?): File {
        val suffix = fileName
            ?.substringAfterLast('.', "")
            ?.takeIf { extension -> extension.isNotBlank() }
            ?.let { extension -> ".$extension" }
            ?: ".jpg"
        return File.createTempFile("ultrasound_input_", suffix, context.cacheDir).also { file ->
            FileOutputStream(file).use { out ->
                compress(Bitmap.CompressFormat.JPEG, 95, out)
            }
        }
    }
}

data class PreparedContents(
    val contents: Contents,
    private val tempFile: File?
) : AutoCloseable {
    override fun close() {
        tempFile?.delete()
    }
}
