package com.fada.ultrasound.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteClassifier(
    private val context: Context,
    private val modelFileName: String = "fada_classifier.tflite",
    private val useGpu: Boolean = true
) {
    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var isInitialized = false

    // Model configuration
    private val numClasses = 12
    private var inputShape: IntArray = intArrayOf(1, 3, 224, 224) // NCHW
    private var isNHWC = false

    companion object {
        private const val TAG = "TFLiteClassifier"
    }

    /**
     * Initialize the TFLite interpreter.
     * Must be called before running inference.
     */
    fun initialize(): Boolean {
        if (isInitialized) return true

        return try {
            val model = loadModelFile()
            val options = Interpreter.Options()

            // Try GPU acceleration if requested
            if (useGpu && CompatibilityList().isDelegateSupportedOnThisDevice) {
                try {
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    Log.i(TAG, "GPU acceleration enabled")
                } catch (e: Exception) {
                    Log.w(TAG, "GPU acceleration not available, using CPU", e)
                }
            }

            // Use multiple threads for CPU inference
            options.setNumThreads(4)

            interpreter = Interpreter(model, options)

            // Get input/output tensor info
            val inputTensor = interpreter!!.getInputTensor(0)
            inputShape = inputTensor.shape()
            Log.i(TAG, "Input shape: ${inputShape.contentToString()}")
            Log.i(TAG, "Input dtype: ${inputTensor.dataType()}")

            // Detect NCHW vs NHWC format
            // NCHW: [1, 3, 224, 224] - channels at index 1
            // NHWC: [1, 224, 224, 3] - channels at index 3
            isNHWC = inputShape.size == 4 && inputShape[3] == 3

            val outputTensor = interpreter!!.getOutputTensor(0)
            Log.i(TAG, "Output shape: ${outputTensor.shape().contentToString()}")

            isInitialized = true
            Log.i(TAG, "Model initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize model", e)
            false
        }
    }

    /**
     * Run classification on a bitmap image.
     *
     * @param bitmap Input image (any size, will be resized)
     * @return ClassificationResult with predictions and timing
     */
    fun classify(bitmap: Bitmap): ClassificationResult? {
        if (!isInitialized) {
            if (!initialize()) {
                Log.e(TAG, "Model not initialized")
                return null
            }
        }

        return try {
            val startTime = System.currentTimeMillis()

            // Preprocess image based on detected format
            val inputBuffer = if (isNHWC) {
                ImagePreprocessor.preprocessBitmapNHWC(bitmap)
            } else {
                ImagePreprocessor.preprocessBitmap(bitmap)
            }

            // Prepare output buffer
            val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            // Run inference
            interpreter!!.run(inputBuffer, outputBuffer)

            val inferenceTime = System.currentTimeMillis() - startTime

            // Extract logits from output buffer
            outputBuffer.rewind()
            val logits = FloatArray(numClasses)
            for (i in 0 until numClasses) {
                logits[i] = outputBuffer.float
            }

            Log.d(TAG, "Inference completed in ${inferenceTime}ms")

            ClassificationResult.fromLogits(logits, inferenceTime)
        } catch (e: Exception) {
            Log.e(TAG, "Classification failed", e)
            null
        }
    }

    /**
     * Run classification on a pre-processed ByteBuffer.
     */
    fun classifyBuffer(inputBuffer: ByteBuffer): ClassificationResult? {
        if (!isInitialized) {
            if (!initialize()) return null
        }

        return try {
            val startTime = System.currentTimeMillis()

            val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4)
            outputBuffer.order(ByteOrder.nativeOrder())

            inputBuffer.rewind()
            interpreter!!.run(inputBuffer, outputBuffer)

            val inferenceTime = System.currentTimeMillis() - startTime

            outputBuffer.rewind()
            val logits = FloatArray(numClasses)
            for (i in 0 until numClasses) {
                logits[i] = outputBuffer.float
            }

            ClassificationResult.fromLogits(logits, inferenceTime)
        } catch (e: Exception) {
            Log.e(TAG, "Classification failed", e)
            null
        }
    }

    /**
     * Load TFLite model from assets.
     */
    private fun loadModelFile(): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelFileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Check if GPU is available.
     */
    fun isGpuAvailable(): Boolean {
        return CompatibilityList().isDelegateSupportedOnThisDevice
    }

    /**
     * Get model information for display.
     */
    fun getModelInfo(): String {
        return buildString {
            appendLine("Model: $modelFileName")
            appendLine("Initialized: $isInitialized")
            appendLine("Input shape: ${inputShape.contentToString()}")
            appendLine("Format: ${if (isNHWC) "NHWC" else "NCHW"}")
            appendLine("GPU available: ${isGpuAvailable()}")
            appendLine("GPU enabled: ${gpuDelegate != null}")
        }
    }

    /**
     * Release resources.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        isInitialized = false
        Log.i(TAG, "Classifier closed")
    }
}
