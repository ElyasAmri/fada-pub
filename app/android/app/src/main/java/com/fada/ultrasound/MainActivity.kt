package com.fada.ultrasound

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.fada.ultrasound.ui.CrashBoundaryScreen
import com.fada.ultrasound.ui.FADANavHost
import com.fada.ultrasound.ui.theme.FADATheme

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val crashDetails = AppErrorBoundary.consumeLastCrash(this)

        setContent {
            var pendingCrash by remember { mutableStateOf(crashDetails) }
            FADATheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (pendingCrash.isNullOrBlank()) {
                        FADANavHost()
                    } else {
                        CrashBoundaryScreen(
                            crashDetails = pendingCrash ?: "Unknown error",
                            onContinue = {
                                pendingCrash = null
                            }
                        )
                    }
                }
            }
        }
    }
}
