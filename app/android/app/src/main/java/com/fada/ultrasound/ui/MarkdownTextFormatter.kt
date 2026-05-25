package com.fada.ultrasound.ui

import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight

internal sealed interface MarkdownBlock {
    data class Heading(val level: Int, val text: String) : MarkdownBlock
    data class Paragraph(val text: String) : MarkdownBlock
    data class ListItem(val marker: String, val text: String) : MarkdownBlock
    data class CodeBlock(val text: String) : MarkdownBlock
    data object Blank : MarkdownBlock
}

internal object MarkdownTextFormatter {
    private val headingRegex = Regex("^(#{1,6})\\s+(.+)$")
    private val unorderedListRegex = Regex("^([-*])\\s+(.+)$")
    private val orderedListRegex = Regex("^(\\d+)[.)]\\s+(.+)$")

    fun parseBlocks(markdown: String): List<MarkdownBlock> {
        val blocks = mutableListOf<MarkdownBlock>()
        val paragraphLines = mutableListOf<String>()
        val codeLines = mutableListOf<String>()
        var inCodeBlock = false

        fun flushParagraph() {
            if (paragraphLines.isNotEmpty()) {
                blocks += MarkdownBlock.Paragraph(paragraphLines.joinToString("\n"))
                paragraphLines.clear()
            }
        }

        fun appendBlank() {
            if (blocks.isNotEmpty() && blocks.last() !is MarkdownBlock.Blank) {
                blocks += MarkdownBlock.Blank
            }
        }

        markdown
            .replace("\r\n", "\n")
            .replace('\r', '\n')
            .split('\n')
            .forEach { rawLine ->
                val line = rawLine.trimEnd()
                val trimmedStart = line.trimStart()

                if (trimmedStart.startsWith("```")) {
                    if (inCodeBlock) {
                        blocks += MarkdownBlock.CodeBlock(codeLines.joinToString("\n"))
                        codeLines.clear()
                        inCodeBlock = false
                    } else {
                        flushParagraph()
                        inCodeBlock = true
                    }
                    return@forEach
                }

                if (inCodeBlock) {
                    codeLines += rawLine
                    return@forEach
                }

                if (line.isBlank()) {
                    flushParagraph()
                    appendBlank()
                    return@forEach
                }

                headingRegex.matchEntire(trimmedStart)?.let { match ->
                    flushParagraph()
                    blocks += MarkdownBlock.Heading(
                        level = match.groupValues[1].length,
                        text = match.groupValues[2]
                    )
                    return@forEach
                }

                unorderedListRegex.matchEntire(trimmedStart)?.let { match ->
                    flushParagraph()
                    blocks += MarkdownBlock.ListItem(
                        marker = "-",
                        text = match.groupValues[2]
                    )
                    return@forEach
                }

                orderedListRegex.matchEntire(trimmedStart)?.let { match ->
                    flushParagraph()
                    blocks += MarkdownBlock.ListItem(
                        marker = "${match.groupValues[1]}.",
                        text = match.groupValues[2]
                    )
                    return@forEach
                }

                paragraphLines += line
            }

        if (inCodeBlock) {
            blocks += MarkdownBlock.CodeBlock(codeLines.joinToString("\n"))
        }
        flushParagraph()

        return blocks
            .dropWhile { it is MarkdownBlock.Blank }
            .dropLastWhile { it is MarkdownBlock.Blank }
    }

    fun buildAnnotatedText(
        text: String,
        streaming: Boolean,
        baseWeight: FontWeight = FontWeight.Normal
    ): AnnotatedString {
        return buildAnnotatedString {
            var index = 0
            while (index < text.length) {
                val marker = strongMarkerAt(text, index)
                if (marker == null) {
                    append(text[index])
                    index += 1
                    continue
                }

                val close = text.indexOf(marker, startIndex = index + marker.length)
                if (close >= 0) {
                    pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                    append(text.substring(index + marker.length, close))
                    pop()
                    index = close + marker.length
                } else if (streaming) {
                    pushStyle(SpanStyle(fontWeight = FontWeight.Bold))
                    append(text.substring(index + marker.length))
                    pop()
                    index = text.length
                } else {
                    append(marker)
                    index += marker.length
                }
            }

            if (baseWeight != FontWeight.Normal) {
                addStyle(SpanStyle(fontWeight = baseWeight), 0, length)
            }
        }
    }

    private fun strongMarkerAt(text: String, index: Int): String? {
        return when {
            text.startsWith("**", index) -> "**"
            text.startsWith("__", index) -> "__"
            else -> null
        }
    }
}
