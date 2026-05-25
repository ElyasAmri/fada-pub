package com.fada.ultrasound.inference

import com.fada.ultrasound.data.UltrasoundClasses

/**
 * Represents a single classification prediction.
 */
data class Prediction(
    val classIndex: Int,
    val className: String,
    val displayName: String,
    val confidence: Float,
    val description: String
)

data class ClassificationResult(
    val predictions: List<Prediction>,
    val inferenceTimeMs: Long,
    val modelName: String = "FADA Classifier"
) {
    /**
     * Get the top prediction.
     */
    val topPrediction: Prediction
        get() = predictions.first()

    /**
     * Get top-k predictions.
     */
    fun topK(k: Int): List<Prediction> {
        return predictions.take(k.coerceAtMost(predictions.size))
    }

    /**
     * Check if the top prediction is confident.
     */
    fun isConfident(threshold: Float = 0.7f): Boolean {
        return topPrediction.confidence >= threshold
    }

    /**
     * Get confidence level description.
     */
    fun getConfidenceLevel(): String {
        return when {
            topPrediction.confidence >= 0.85f -> "High confidence"
            topPrediction.confidence >= 0.70f -> "Good confidence"
            topPrediction.confidence >= 0.50f -> "Moderate confidence"
            else -> "Low confidence"
        }
    }

    companion object {
        /**
         * Create a ClassificationResult from raw model output.
         *
         * @param logits Raw model output (12 values)
         * @param inferenceTimeMs Time taken for inference
         * @return ClassificationResult with sorted predictions
         */
        fun fromLogits(logits: FloatArray, inferenceTimeMs: Long): ClassificationResult {
            // Apply softmax to convert logits to probabilities
            val maxLogit = logits.maxOrNull() ?: 0f
            val expLogits = logits.map { kotlin.math.exp(it - maxLogit) }
            val sumExp = expLogits.sum()
            val probabilities = expLogits.map { (it / sumExp).toFloat() }

            // Create predictions sorted by confidence (descending)
            val predictions = probabilities.mapIndexed { index, probability ->
                val className = UltrasoundClasses.CLASSES.getOrElse(index) { "Unknown" }
                Prediction(
                    classIndex = index,
                    className = className,
                    displayName = UltrasoundClasses.getDisplayName(className),
                    confidence = probability,
                    description = UltrasoundClasses.getDescription(className)
                )
            }.sortedByDescending { it.confidence }

            return ClassificationResult(
                predictions = predictions,
                inferenceTimeMs = inferenceTimeMs
            )
        }
    }
}
