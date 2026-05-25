# FADA Android App

Native Android app for image-to-LLM workflows in fetal ultrasound research.

## Features

- Chat-first image analysis workflow with camera capture and gallery picker
- Conversation selector for switching, creating, and deleting local chat threads
- On-device Gemma model selector and model management screen (Gemma 4 E2B/E4B)
- Runtime model download and deletion from app-private storage (`FADA/models/...`)
- Settings screen for chat behavior, model management access, and local data management
- Optional SHA-256 verification support in model metadata
- Material 3 Compose UI with a single top app bar

## Project Structure

```
app/android/
|-- app/src/main/java/com/fada/ultrasound/
|   |-- llm/
|   |   |-- LlmModels.kt
|   |   |-- ModelDownloadManager.kt
|   |   +-- LlmResponseGenerator.kt
|   |-- viewmodel/
|   |   +-- InferenceViewModel.kt
|   +-- ui/
|       |-- ChatScreen.kt
|       |-- ConversationsScreen.kt
|       |-- ModelsScreen.kt
|       |-- SettingsScreen.kt
|       |-- MainScreen.kt
|       |-- CameraScreen.kt
|       |-- ResultsScreen.kt
|       +-- Navigation.kt
```

## Build

1. Open `app/android/` in Android Studio.
2. Sync Gradle and run `app`.

Command line:

```bash
cd app/android
./gradlew assembleDebug
```

## Model Delivery

Models are not shipped inside the APK. The app downloads selected `.litertlm` files from the configured model URLs in `LlmModels.kt` and stores them under app-private storage.

For production, pin your own versioned URLs and set `expectedSha256` values for each model entry.

