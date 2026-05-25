package com.fada.ultrasound.ui

import androidx.compose.animation.AnimatedContentTransitionScope
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.fada.ultrasound.viewmodel.InferenceViewModel

/**
 * Navigation routes for the app.
 */
sealed class Screen(val route: String) {
    data object Chat : Screen("chat")
    data object Conversations : Screen("conversations")
    data object Models : Screen("models")
    data object Settings : Screen("settings")
    data object SystemPrompt : Screen("system_prompt")
    data object Camera : Screen("camera")
    data object Results : Screen("results")
}

/**
 * Main navigation host for the app.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FADANavHost() {
    val navController = rememberNavController()
    val viewModel: InferenceViewModel = viewModel()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    val conversations by viewModel.conversations.collectAsState()
    val currentConversationId by viewModel.currentConversationId.collectAsState()
    val currentConversation = conversations.firstOrNull { it.id == currentConversationId }
    var selectedConversationIds by remember { mutableStateOf(emptySet<String>()) }
    var selectedModelIds by remember { mutableStateOf(emptySet<String>()) }
    val showGlobalTopBar = currentRoute != Screen.Camera.route && currentRoute != Screen.Results.route
    val showSettingsAction = currentRoute != Screen.Settings.route &&
        currentRoute != Screen.Models.route &&
        currentRoute != Screen.SystemPrompt.route

    LaunchedEffect(currentRoute) {
        if (currentRoute != Screen.Conversations.route) {
            selectedConversationIds = emptySet()
        }
        if (currentRoute != Screen.Models.route) {
            selectedModelIds = emptySet()
        }
    }

    Scaffold(
        topBar = {
            if (showGlobalTopBar) {
                TopAppBar(
                    title = {
                        Text(
                            when (currentRoute) {
                                Screen.Conversations.route -> "FADA"
                                Screen.Chat.route -> currentConversation?.title ?: "New conversation"
                                Screen.Models.route -> "Models"
                                Screen.Settings.route -> "Settings"
                                Screen.SystemPrompt.route -> "System prompt"
                                else -> "FADA"
                            }
                        )
                    },
                    navigationIcon = {
                        if (currentRoute == Screen.Chat.route) {
                            IconButton(
                                onClick = {
                                    if (!navController.popBackStack(Screen.Conversations.route, inclusive = false)) {
                                        navController.navigate(Screen.Conversations.route) {
                                            launchSingleTop = true
                                        }
                                    }
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                    contentDescription = "Back to threads"
                                )
                            }
                        } else if (currentRoute != null && currentRoute != Screen.Conversations.route) {
                            IconButton(onClick = { navController.popBackStack() }) {
                                Icon(
                                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                                    contentDescription = "Back"
                                )
                            }
                        }
                    },
                    actions = {
                        if (currentRoute == Screen.Conversations.route) {
                            if (selectedConversationIds.isNotEmpty()) {
                                IconButton(
                                    onClick = {
                                        viewModel.deleteConversations(selectedConversationIds)
                                        selectedConversationIds = emptySet()
                                    }
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Delete,
                                        contentDescription = "Delete selected threads"
                                    )
                                }
                            } else {
                                IconButton(
                                    onClick = {
                                        viewModel.createNewConversation()
                                        navController.navigate(Screen.Chat.route) {
                                            launchSingleTop = true
                                        }
                                    }
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Add,
                                        contentDescription = "New chat"
                                    )
                                }
                            }
                        }
                        if (currentRoute == Screen.Models.route) {
                            if (selectedModelIds.isNotEmpty()) {
                                IconButton(
                                    onClick = {
                                        viewModel.deleteStoredModels(selectedModelIds)
                                        selectedModelIds = emptySet()
                                    }
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Delete,
                                        contentDescription = "Delete selected models"
                                    )
                                }
                            } else {
                                IconButton(
                                    onClick = { viewModel.clearUnusedModelFiles() }
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.DeleteSweep,
                                        contentDescription = "Clear old model files"
                                    )
                                }
                            }
                        }
                        if (showSettingsAction) {
                            IconButton(
                                onClick = {
                                    navController.navigate(Screen.Settings.route) {
                                        launchSingleTop = true
                                    }
                                }
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Settings,
                                    contentDescription = "Settings"
                                )
                            }
                        }
                    }
                )
            }
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = Screen.Conversations.route,
            modifier = Modifier.padding(paddingValues),
            enterTransition = {
                slideIntoContainer(
                    AnimatedContentTransitionScope.SlideDirection.Left,
                    animationSpec = tween(220)
                )
            },
            exitTransition = {
                slideOutOfContainer(
                    AnimatedContentTransitionScope.SlideDirection.Left,
                    animationSpec = tween(220)
                )
            },
            popEnterTransition = {
                slideIntoContainer(
                    AnimatedContentTransitionScope.SlideDirection.Right,
                    animationSpec = tween(220)
                )
            },
            popExitTransition = {
                slideOutOfContainer(
                    AnimatedContentTransitionScope.SlideDirection.Right,
                    animationSpec = tween(220)
                )
            }
        ) {
            composable(Screen.Chat.route) {
                ChatScreen(
                    viewModel = viewModel,
                    onNavigateToCamera = {
                        navController.navigate(Screen.Camera.route)
                    },
                    onNavigateToModels = {
                        navController.navigate(Screen.Models.route) {
                            launchSingleTop = true
                        }
                    }
                )
            }

            composable(Screen.Conversations.route) {
                ConversationsScreen(
                    viewModel = viewModel,
                    selectedConversationIds = selectedConversationIds,
                    onToggleConversationSelection = { conversationId ->
                        selectedConversationIds = selectedConversationIds.toggle(conversationId)
                    },
                    onOpenConversation = {
                        navController.navigate(Screen.Chat.route) {
                            launchSingleTop = true
                        }
                    }
                )
            }

            composable(Screen.Models.route) {
                ModelsScreen(
                    viewModel = viewModel,
                    selectedModelIds = selectedModelIds,
                    onToggleModelSelection = { modelId ->
                        selectedModelIds = selectedModelIds.toggle(modelId)
                    }
                )
            }

            composable(Screen.Settings.route) {
                SettingsScreen(
                    viewModel = viewModel,
                    onNavigateToModels = {
                        navController.navigate(Screen.Models.route) {
                            launchSingleTop = true
                        }
                    },
                    onNavigateToSystemPrompt = {
                        navController.navigate(Screen.SystemPrompt.route) {
                            launchSingleTop = true
                        }
                    }
                )
            }

            composable(Screen.SystemPrompt.route) {
                SystemPromptScreen(viewModel = viewModel)
            }

            composable(Screen.Camera.route) {
                CameraScreen(
                    onImageCaptured = { bitmap ->
                        viewModel.setCapturedImage(bitmap)
                        navController.popBackStack()
                    },
                    onNavigateBack = {
                        navController.popBackStack()
                    }
                )
            }

            composable(Screen.Results.route) {
                ResultsScreen(
                    viewModel = viewModel,
                    onNavigateBack = {
                        navController.popBackStack()
                    },
                    onNewImage = {
                        viewModel.clearSelection()
                        navController.popBackStack()
                    }
                )
            }
        }
    }
}

private fun Set<String>.toggle(value: String): Set<String> {
    return if (value in this) {
        this - value
    } else {
        this + value
    }
}
