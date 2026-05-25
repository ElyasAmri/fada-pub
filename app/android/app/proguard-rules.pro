# FADA Android App ProGuard Rules

# TensorFlow Lite
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.**

# Keep model class names for debugging
-keepnames class com.fada.ultrasound.inference.**

# Compose
-dontwarn androidx.compose.**

# CameraX
-keep class androidx.camera.** { *; }
