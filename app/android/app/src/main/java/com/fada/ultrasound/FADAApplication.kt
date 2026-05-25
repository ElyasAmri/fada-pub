package com.fada.ultrasound

import android.app.Application

class FADAApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        AppErrorBoundary.install(this)
    }
}
