package com.fada.ultrasound

import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@LargeTest
@RunWith(AndroidJUnit4::class)
class ChatFlowE2eTest {

    private lateinit var device: UiDevice
    private lateinit var scenario: ActivityScenario<MainActivity>

    @Before
    fun launchApp() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        device = UiDevice.getInstance(instrumentation)
        device.pressHome()
        device.executeShellCommand("cmd statusbar collapse")

        scenario = ActivityScenario.launch(MainActivity::class.java)
        device.wait(Until.hasObject(By.pkg(PACKAGE_NAME).depth(0)), START_TIMEOUT_MS)
        device.findObject(By.text("Continue"))?.click()
    }

    @After
    fun closeApp() {
        scenario.close()
    }

    @Test
    fun threadsScreenShowsSearchAndNewConversationPrompt() {
        waitForObject(text("FADA"))
        assertNotNull(device.findObject(By.text("Search threads")))
        assertNotNull(device.findObject(By.desc("New chat")))
    }

    @Test
    fun newChatAcceptsTextPrompt() {
        waitForObject(desc("New chat")).click()
        val input = waitForObject(text("Message"))
        input.click()
        device.executeShellCommand("input text hi")

        waitForObject(text("hi"))
        assertNotNull(device.findObject(By.desc("Send")))
        assertNull(device.findObject(By.textContains("FAILED_PRECONDITION")))
    }

    private fun waitForObject(selector: BySelectorCompat, timeoutMs: Long = START_TIMEOUT_MS) =
        requireNotNull(device.wait(Until.findObject(selector.selector), timeoutMs)) {
            "Timed out waiting for ${selector.description}"
        }

    private fun text(text: String) = BySelectorCompat(
        selector = androidx.test.uiautomator.By.text(text),
        description = "text '$text'"
    )

    private fun textContains(text: String) = BySelectorCompat(
        selector = androidx.test.uiautomator.By.textContains(text),
        description = "text containing '$text'"
    )

    private fun desc(description: String) = BySelectorCompat(
        selector = androidx.test.uiautomator.By.desc(description),
        description = "content description '$description'"
    )

    private data class BySelectorCompat(
        val selector: androidx.test.uiautomator.BySelector,
        val description: String
    )

    private companion object {
        const val PACKAGE_NAME = "com.fada.ultrasound"
        const val START_TIMEOUT_MS = 10_000L
    }
}
