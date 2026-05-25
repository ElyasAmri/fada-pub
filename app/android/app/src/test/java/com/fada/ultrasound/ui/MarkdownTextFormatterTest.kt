package com.fada.ultrasound.ui

import androidx.compose.ui.text.font.FontWeight
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MarkdownTextFormatterTest {
    @Test
    fun parseBlocks_preservesParagraphSpacingAndLists() {
        val blocks = MarkdownTextFormatter.parseBlocks(
            "Intro line\nwith detail\n\n- **Finding:** normal\n- View: axial"
        )

        assertEquals(
            listOf(
                MarkdownBlock.Paragraph("Intro line\nwith detail"),
                MarkdownBlock.Blank,
                MarkdownBlock.ListItem("-", "**Finding:** normal"),
                MarkdownBlock.ListItem("-", "View: axial")
            ),
            blocks
        )
    }

    @Test
    fun buildAnnotatedText_parsesStrongEmphasis() {
        val annotated = MarkdownTextFormatter.buildAnnotatedText(
            text = "Likely **axial view** today",
            streaming = false
        )

        assertEquals("Likely axial view today", annotated.text)
        assertEquals(1, annotated.spanStyles.size)
        assertEquals(7, annotated.spanStyles.first().start)
        assertEquals(17, annotated.spanStyles.first().end)
        assertEquals(FontWeight.Bold, annotated.spanStyles.first().item.fontWeight)
    }

    @Test
    fun buildAnnotatedText_stylesUnclosedStrongMarkerWhileStreaming() {
        val annotated = MarkdownTextFormatter.buildAnnotatedText(
            text = "Likely **axial",
            streaming = true
        )

        assertEquals("Likely axial", annotated.text)
        assertTrue(annotated.spanStyles.any { it.item.fontWeight == FontWeight.Bold })
    }

    @Test
    fun buildAnnotatedText_keepsUnclosedStrongMarkerAfterStreamingCompletes() {
        val annotated = MarkdownTextFormatter.buildAnnotatedText(
            text = "Likely **axial",
            streaming = false
        )

        assertEquals("Likely **axial", annotated.text)
        assertEquals(0, annotated.spanStyles.size)
    }
}
