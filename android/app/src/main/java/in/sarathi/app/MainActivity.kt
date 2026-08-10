package `in`.sarathi.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.KeyEvent
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import `in`.sarathi.app.runtime.GuidanceService

/**
 * A deliberately tiny screen.
 *
 * The product is used with the display off, so this exists to grant camera
 * permission, start and stop the service, and switch language. Every control
 * is large, has a content description for TalkBack, and is reachable without
 * sight - a button whose only label is an icon does not exist to this user.
 *
 * Volume keys are captured so guidance can be started, stopped and queried
 * without unlocking or looking at the phone.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var status: TextView
    private var running = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildUi())
        requestPermissions()
    }

    private fun buildUi(): ViewGroup {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(48, 48, 48, 48)
        }
        status = TextView(this).apply {
            text = getString(R.string.status_idle)
            textSize = 28f
            gravity = Gravity.CENTER
            // Announced by TalkBack whenever it changes, so state is audible
            // rather than only visible.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                accessibilityLiveRegion = android.view.View.ACCESSIBILITY_LIVE_REGION_POLITE
            }
        }
        val toggle = Button(this).apply {
            text = getString(R.string.start_desc)
            textSize = 24f
            minHeight = 180
            contentDescription = getString(R.string.start_desc)
            setOnClickListener { toggle() }
        }
        val language = Button(this).apply {
            text = "English / हिन्दी"
            textSize = 22f
            minHeight = 150
            contentDescription = getString(R.string.lang_desc)
            setOnClickListener { switchLanguage() }
        }
        root.addView(status)
        root.addView(toggle)
        root.addView(language)
        return root
    }

    private fun requestPermissions() {
        val needed = buildList {
            if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) add(Manifest.permission.CAMERA)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
                ContextCompat.checkSelfPermission(
                    this@MainActivity, Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (needed.isNotEmpty()) ActivityCompat.requestPermissions(this, needed.toTypedArray(), 1)
    }

    private fun toggle() {
        val intent = Intent(this, GuidanceService::class.java)
        if (running) stopService(intent) else ContextCompat.startForegroundService(this, intent)
        running = !running
        status.text = getString(if (running) R.string.status_running else R.string.status_idle)
    }

    private fun switchLanguage() {
        val prefs = getSharedPreferences(GuidanceService.PREFS, Context.MODE_PRIVATE)
        val next = if (prefs.getString("lang", "en") == "en") "hi" else "en"
        prefs.edit().putString("lang", next).apply()
        if (running) { toggle(); toggle() }   // restart so the new voice loads
    }

    /**
     * Forward an on-demand request to the running service.
     *
     * Deliberately silent when guidance is not running: pressing volume-up on a
     * stopped app should not spin up a camera and a two-gigabyte model. Start
     * it first, with the button that starts it.
     */
    private fun send(action: String) {
        if (!running) return
        startService(Intent(this, GuidanceService::class.java).setAction(action))
    }

    /**
     * Hardware buttons, so the app is usable with the screen off and the phone
     * in a pocket.
     *
     *   volume-down          start / stop guidance
     *   volume-up, tapped    describe the scene
     *   volume-up, held      read any text
     *
     * Both on-demand features share one button because there are only two
     * buttons, and tap-versus-hold is a distinction that can be made by feel
     * without looking. The tap fires on key-*up*, not key-down: committing on
     * the way down would make every long press trigger a description first.
     */
    private var volumeUpHandled = false

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> { toggle(); true }
            KeyEvent.KEYCODE_VOLUME_UP -> {
                if (event != null && event.repeatCount == 0) {
                    volumeUpHandled = false
                    event.startTracking()   // required for onKeyLongPress to fire
                }
                true
            }
            else -> super.onKeyDown(keyCode, event)
        }
    }

    override fun onKeyLongPress(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode != KeyEvent.KEYCODE_VOLUME_UP) return super.onKeyLongPress(keyCode, event)
        volumeUpHandled = true
        send(GuidanceService.ACTION_READ)
        return true
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode != KeyEvent.KEYCODE_VOLUME_UP) return super.onKeyUp(keyCode, event)
        if (!volumeUpHandled) send(GuidanceService.ACTION_DESCRIBE)
        volumeUpHandled = false
        return true
    }
}
