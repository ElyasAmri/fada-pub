package com.fada.ultrasound.inference

import android.graphics.Bitmap
import android.graphics.Color
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ImagePreprocessor {

    // Model input dimensions
    const val INPUT_WIDTH = 224
    const val INPUT_HEIGHT = 224
    const val NUM_CHANNELS = 3

    // ImageNet normalization constants (must match training exactly)
    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f) // RGB
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)  // RGB

    /**
     * Preprocess a bitmap for TFLite inference.
     * Applies resize, RGB extraction, and ImageNet normalization.
     *
     * @param bitmap Input bitmap (any size)
     * @return ByteBuffer ready for TFLite (NCHW format, float32)
     */
    fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        // Resize to model input size
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)

        // Calculate buffer size: 1 batch * 3 channels * 224 * 224 * 4 bytes per float
        val bufferSize = 1 * NUM_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT * 4
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Extract pixels
        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        resized.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        // Convert to normalized float array in NCHW format (channels first)
        // TFLite expects: [batch, channels, height, width]

        // Red channel
        for (y in 0 until INPUT_HEIGHT) {
            for (x in 0 until INPUT_WIDTH) {
                val pixel = pixels[y * INPUT_WIDTH + x]
                val r = Color.red(pixel) / 255.0f
                val normalized = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                byteBuffer.putFloat(normalized)
            }
        }

        // Green channel
        for (y in 0 until INPUT_HEIGHT) {
            for (x in 0 until INPUT_WIDTH) {
                val pixel = pixels[y * INPUT_WIDTH + x]
                val g = Color.green(pixel) / 255.0f
                val normalized = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                byteBuffer.putFloat(normalized)
            }
        }

        // Blue channel
        for (y in 0 until INPUT_HEIGHT) {
            for (x in 0 until INPUT_WIDTH) {
                val pixel = pixels[y * INPUT_WIDTH + x]
                val b = Color.blue(pixel) / 255.0f
                val normalized = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
                byteBuffer.putFloat(normalized)
            }
        }

        byteBuffer.rewind()

        // Clean up resized bitmap if different from original
        if (resized != bitmap) {
            resized.recycle()
        }

        return byteBuffer
    }

    /**
     * Alternative preprocessing that returns a FloatArray (for debugging).
     */
    fun preprocessToFloatArray(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
        val floatArray = FloatArray(NUM_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT)

        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        resized.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        var index = 0

        // NCHW format: all R, then all G, then all B
        for (c in 0 until NUM_CHANNELS) {
            for (y in 0 until INPUT_HEIGHT) {
                for (x in 0 until INPUT_WIDTH) {
                    val pixel = pixels[y * INPUT_WIDTH + x]
                    val value = when (c) {
                        0 -> Color.red(pixel) / 255.0f
                        1 -> Color.green(pixel) / 255.0f
                        else -> Color.blue(pixel) / 255.0f
                    }
                    floatArray[index++] = (value - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
                }
            }
        }

        if (resized != bitmap) {
            resized.recycle()
        }

        return floatArray
    }

    /**
     * Preprocess for NHWC format (height, width, channels).
     * Use this if TFLite model expects NHWC input.
     */
    fun preprocessBitmapNHWC(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)

        val bufferSize = 1 * INPUT_WIDTH * INPUT_HEIGHT * NUM_CHANNELS * 4
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        resized.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        // NHWC format: for each pixel, store R, G, B
        for (y in 0 until INPUT_HEIGHT) {
            for (x in 0 until INPUT_WIDTH) {
                val pixel = pixels[y * INPUT_WIDTH + x]

                val r = (Color.red(pixel) / 255.0f - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
                val g = (Color.green(pixel) / 255.0f - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
                val b = (Color.blue(pixel) / 255.0f - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

                byteBuffer.putFloat(r)
                byteBuffer.putFloat(g)
                byteBuffer.putFloat(b)
            }
        }

        byteBuffer.rewind()

        if (resized != bitmap) {
            resized.recycle()
        }

        return byteBuffer
    }
}
