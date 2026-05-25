package com.fada.ultrasound

import android.app.Application
import android.content.Context
import android.os.Process
import java.io.PrintWriter
import java.io.StringWriter
import kotlin.system.exitProcess

object AppErrorBoundary {
    private const val PREFS_NAME = "app_error_boundary"
    private const val KEY_LAST_CRASH = "last_crash"

    fun install(application: Application) {
        val current = Thread.getDefaultUncaughtExceptionHandler()
        if (current is BoundaryHandler) return

        Thread.setDefaultUncaughtExceptionHandler(
            BoundaryHandler(
                application = application,
                previous = current
            )
        )
    }

    fun consumeLastCrash(context: Context): String? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val value = prefs.getString(KEY_LAST_CRASH, null)
        if (!value.isNullOrBlank()) {
            prefs.edit().remove(KEY_LAST_CRASH).apply()
        }
        return value
    }

    private fun storeCrash(context: Context, throwable: Throwable) {
        val stacktrace = StringWriter().also { writer ->
            throwable.printStackTrace(PrintWriter(writer))
        }.toString()

        val summary = buildString {
            append(throwable::class.java.simpleName)
            val message = throwable.message
            if (!message.isNullOrBlank()) {
                append(": ")
                append(message)
            }
            append("\n\n")
            append(stacktrace)
        }

        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_LAST_CRASH, summary)
            .apply()
    }

    private class BoundaryHandler(
        private val application: Application,
        private val previous: Thread.UncaughtExceptionHandler?
    ) : Thread.UncaughtExceptionHandler {
        override fun uncaughtException(thread: Thread, throwable: Throwable) {
            storeCrash(application, throwable)
            if (previous != null) {
                previous.uncaughtException(thread, throwable)
                return
            }

            Process.killProcess(Process.myPid())
            exitProcess(10)
        }
    }
}

