package com.fada.ultrasound.llm

import org.junit.Assert.assertEquals
import org.junit.Test

class StreamingTextAccumulatorTest {
    @Test
    fun append_preservesWhitespaceOnlyChunks() {
        var output = "Anatomy:"
        output = StreamingTextAccumulator.append(output, " ")
        output = StreamingTextAccumulator.append(output, "**head**")
        output = StreamingTextAccumulator.append(output, "\n\n")
        output = StreamingTextAccumulator.append(output, "Likely view.")

        assertEquals("Anatomy: **head**\n\nLikely view.", output)
    }

    @Test
    fun append_acceptsCumulativeChunksWithoutDuplicating() {
        var output = StreamingTextAccumulator.append("", "The")
        output = StreamingTextAccumulator.append(output, "The fetal")
        output = StreamingTextAccumulator.append(output, "The fetal head")

        assertEquals("The fetal head", output)
    }

    @Test
    fun append_doesNotDropRepeatedSuffixChunks() {
        var output = StreamingTextAccumulator.append("", "**")
        output = StreamingTextAccumulator.append(output, "bold")
        output = StreamingTextAccumulator.append(output, "**")

        assertEquals("**bold**", output)
    }
}
