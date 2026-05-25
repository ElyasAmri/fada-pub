package com.fada.ultrasound.llm

internal object StreamingTextAccumulator {
    fun append(current: String, chunk: String): String {
        if (chunk.isEmpty()) return current
        if (current.isEmpty()) return chunk

        return if (chunk.startsWith(current)) {
            chunk
        } else {
            current + chunk
        }
    }
}
