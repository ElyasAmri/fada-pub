package com.fada.ultrasound.llm

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

data class StoredModelInfo(
    val file: File,
    val isStored: Boolean,
    val sizeBytes: Long
)

data class ModelCleanupResult(
    val deletedFiles: Int,
    val deletedBytes: Long
)

/**
 * Downloads and versions LiteRT-LM model files in phone storage under FADA/models.
 */
class ModelDownloadManager(private val context: Context) {

    fun getOrDownloadModel(
        option: LlmModelOption,
        onStatus: (String) -> Unit
    ): File {
        val targetFile = getModelFile(option)
        if (targetFile.exists()) {
            return targetFile
        }

        targetFile.parentFile?.mkdirs()
        val tempFile = File(targetFile.parentFile, "${targetFile.name}.download")

        onStatus("Downloading ${option.displayName}...")
        try {
            downloadFile(option.downloadUrl, tempFile, onStatus)
        } catch (e: IOException) {
            tempFile.delete()
            throw IllegalStateException(
                "Model download interrupted. Check the connection and try again.",
                e
            )
        }

        if (!option.expectedSha256.isNullOrBlank()) {
            onStatus("Verifying model integrity...")
            val downloadedSha = sha256(tempFile)
            if (!downloadedSha.equals(option.expectedSha256, ignoreCase = true)) {
                tempFile.delete()
                throw IllegalStateException(
                    "Checksum mismatch for ${option.displayName}. Expected ${option.expectedSha256}, got $downloadedSha"
                )
            }
        }

        if (targetFile.exists()) {
            targetFile.delete()
        }
        if (!tempFile.renameTo(targetFile)) {
            tempFile.delete()
            throw IllegalStateException("Failed to finalize model file: ${targetFile.absolutePath}")
        }

        return targetFile
    }

    fun invalidateModel(option: LlmModelOption) {
        getModelFile(option).delete()
        removeEmptyParents(getModelFile(option).parentFile)
    }

    fun clearUnusedModelFiles(activeOptions: List<LlmModelOption>): ModelCleanupResult {
        val root = getModelRoot()
        if (!root.exists()) {
            return ModelCleanupResult(deletedFiles = 0, deletedBytes = 0L)
        }

        val activeFiles = activeOptions
            .map { getModelFile(it).canonicalFile }
            .toSet()
        var deletedFiles = 0
        var deletedBytes = 0L

        root.walkBottomUp()
            .filter { it.isFile }
            .forEach { file ->
                val canonical = file.canonicalFile
                if (canonical !in activeFiles) {
                    val size = file.length()
                    if (file.delete()) {
                        deletedFiles += 1
                        deletedBytes += size
                    }
                }
            }

        root.walkBottomUp()
            .filter { it.isDirectory && it != root && it.listFiles()?.isEmpty() == true }
            .forEach { it.delete() }

        return ModelCleanupResult(deletedFiles = deletedFiles, deletedBytes = deletedBytes)
    }

    fun getStoredModelInfo(option: LlmModelOption): StoredModelInfo {
        val file = getModelFile(option)
        return StoredModelInfo(
            file = file,
            isStored = file.exists(),
            sizeBytes = if (file.exists()) file.length() else 0L
        )
    }

    fun getModelFile(option: LlmModelOption): File {
        return File(getModelRoot(), "${option.id}/${option.version}/${option.localFileName}")
    }

    private fun getModelRoot(): File {
        val externalBase = context.getExternalFilesDir(null)
        return if (externalBase != null) {
            File(externalBase, "FADA/models")
        } else {
            File(context.filesDir, "models")
        }
    }

    private fun removeEmptyParents(start: File?) {
        val root = getModelRoot().absoluteFile
        var current = start?.absoluteFile
        while (current != null && current != root && current.listFiles()?.isEmpty() == true) {
            val parent = current.parentFile
            current.delete()
            current = parent
        }
    }

    private fun downloadFile(
        fileUrl: String,
        destination: File,
        onStatus: (String) -> Unit
    ) {
        val connection = (URL(fileUrl).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 30_000
            readTimeout = 120_000
            instanceFollowRedirects = true
        }

        try {
            connection.connect()
            if (connection.responseCode !in 200..299) {
                throw IllegalStateException("Model download failed with HTTP ${connection.responseCode}")
            }

            val totalBytes = connection.contentLengthLong
            connection.inputStream.use { input ->
                FileOutputStream(destination).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var bytesCopied = 0L
                    var bytes = input.read(buffer)
                    while (bytes >= 0) {
                        output.write(buffer, 0, bytes)
                        bytesCopied += bytes

                        if (totalBytes > 0L) {
                            val percent = ((bytesCopied * 100) / totalBytes).toInt()
                            onStatus("Downloading ${percent}%")
                        }
                        bytes = input.read(buffer)
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            var bytes = input.read(buffer)
            while (bytes >= 0) {
                digest.update(buffer, 0, bytes)
                bytes = input.read(buffer)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}

