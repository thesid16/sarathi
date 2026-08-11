package `in`.sarathi.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.util.TypedValue
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import `in`.sarathi.app.models.SharedData
import `in`.sarathi.app.runtime.GuidanceBus
import `in`.sarathi.app.runtime.GuidanceService
import `in`.sarathi.app.ui.DetectionOverlay

/**
 * The screen.
 *
 * An earlier version of this file was three buttons and a word, on the
 * reasoning that a blind user operates the app face-down and a display is
 * wasted effort. The reasoning is half right and the conclusion was wrong.
 * The blind user does not watch the screen — but a sighted helper setting the
 * phone up does, a developer chasing a missed kerb does, and anyone being
 * shown the project for the first time does. An app whose entire state lives
 * in `adb logcat` cannot be handed to someone.
 *
 * So there are two interfaces here, over the same pipeline:
 *
 * **By ear.** Volume-down starts and stops. Volume-up tapped describes the
 * scene, held reads text. Every control carries a content description, no
 * meaning is conveyed by colour alone, and the app is fully operable with the
 * display dark.
 *
 * **By eye.** The live camera with the detector's boxes drawn over it, what it
 * last said, and the numbers that explain why — rate, backend, thermal
 * headroom, and the raw maximum class score that distinguishes "nothing is
 * there" from "the input is broken".
 *
 * State is read from the service rather than tracked here. A local boolean
 * drifts the moment the service is stopped from its notification, and the
 * on-demand controls then refuse to fire because they are guarding on a flag
 * that disagrees with reality.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var preview: PreviewView
    private lateinit var overlay: DetectionOverlay
    private lateinit var statusPill: TextView
    private lateinit var spoken: TextView
    private lateinit var busyLine: TextView
    private lateinit var startButton: TextView
    private lateinit var describeButton: TextView
    private lateinit var readButton: TextView
    private lateinit var modelButton: TextView
    private lateinit var stats: TextView

    private val running: Boolean get() = GuidanceService.isRunning

    private val onSnapshot: (GuidanceBus.Snapshot) -> Unit = { snap -> render(snap) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildUi())
        requestPermissions()
        GuidanceBus.observe(onSnapshot)
    }

    override fun onResume() {
        super.onResume()
        GuidanceService.attachPreview(preview.surfaceProvider)
        render(GuidanceBus.snapshot)
    }

    override fun onPause() {
        // Released rather than left attached: an unwatched preview stream is
        // camera work nobody is looking at, on a device where the whole design
        // is about not doing work nobody needs.
        GuidanceService.attachPreview(null)
        super.onPause()
    }

    override fun onDestroy() {
        GuidanceBus.stopObserving(onSnapshot)
        super.onDestroy()
    }

    // -- layout --------------------------------------------------------------

    private fun dp(value: Int) = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics
    ).toInt()

    private fun buildUi(): ViewGroup {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(INK)
        }

        // -- the live feed, and the boxes over it --
        val viewport = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH, 0, 1f)
            setBackgroundColor(Color.BLACK)
        }
        preview = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(MATCH, MATCH)
            // FIT_CENTER, not FILL: the overlay maps detections onto the whole
            // frame, and a preview that crops would put the boxes somewhere the
            // user is not looking.
            scaleType = PreviewView.ScaleType.FIT_CENTER
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        overlay = DetectionOverlay(this).apply {
            layoutParams = FrameLayout.LayoutParams(MATCH, MATCH)
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        statusPill = TextView(this).apply {
            layoutParams = FrameLayout.LayoutParams(WRAP, WRAP).apply {
                gravity = Gravity.TOP or Gravity.START
                setMargins(dp(14), dp(14), 0, 0)
            }
            setPadding(dp(12), dp(6), dp(12), dp(6))
            textSize = 13f
            setTextColor(INK)
            background = pill(AMBER)
            // Announced by TalkBack when it changes, so the state is audible as
            // well as visible.
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        busyLine = TextView(this).apply {
            layoutParams = FrameLayout.LayoutParams(WRAP, WRAP).apply {
                gravity = Gravity.TOP or Gravity.END
                setMargins(0, dp(14), dp(14), 0)
            }
            setPadding(dp(12), dp(6), dp(12), dp(6))
            textSize = 13f
            setTextColor(INK)
            background = pill(CYAN)
            visibility = View.GONE
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        viewport.addView(preview)
        viewport.addView(overlay)
        viewport.addView(statusPill)
        viewport.addView(busyLine)
        root.addView(viewport)

        // -- what it said: the actual product output, given the most room --
        spoken = TextView(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH, WRAP)
            setPadding(dp(18), dp(14), dp(18), dp(12))
            textSize = 20f
            setTextColor(PAPER)
            text = getString(R.string.spoken_placeholder)
            accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
        }
        root.addView(spoken)

        stats = TextView(this).apply {
            layoutParams = LinearLayout.LayoutParams(MATCH, WRAP)
            setPadding(dp(18), 0, dp(18), dp(12))
            textSize = 12f
            typeface = android.graphics.Typeface.MONOSPACE
            setTextColor(MUTED)
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        root.addView(stats)

        // -- controls --
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(MATCH, WRAP)
            setPadding(dp(12), 0, dp(12), dp(8))
        }
        describeButton = actionButton(getString(R.string.describe_label), SLATE) { send(GuidanceService.ACTION_DESCRIBE) }
        readButton = actionButton(getString(R.string.read_label), SLATE) { send(GuidanceService.ACTION_READ) }
        row.addView(describeButton)
        row.addView(readButton)
        root.addView(row)

        val row2 = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(MATCH, WRAP)
            setPadding(dp(12), 0, dp(12), dp(16))
        }
        startButton = actionButton(getString(R.string.start_desc), AMBER, dark = true) { toggle() }
        modelButton = actionButton(getString(R.string.model_label), SLATE) { chooseModel() }
        row2.addView(startButton)
        row2.addView(modelButton)
        root.addView(row2)

        return root
    }

    private fun pill(colour: Int) = GradientDrawable().apply {
        setColor(colour)
        cornerRadius = dp(14).toFloat()
    }

    /**
     * A control sized for a thumb rather than a cursor.
     *
     * 64dp exceeds the 48dp accessibility minimum deliberately: this is used
     * while walking, one-handed, by someone who may not be looking at it.
     */
    private fun actionButton(
        label: String,
        colour: Int,
        dark: Boolean = false,
        onClick: () -> Unit,
    ): TextView = TextView(this).apply {
        layoutParams = LinearLayout.LayoutParams(0, dp(64), 1f).apply {
            setMargins(dp(6), dp(6), dp(6), dp(6))
        }
        text = label
        contentDescription = label
        gravity = Gravity.CENTER
        textSize = 16f
        setTextColor(if (dark) INK else PAPER)
        background = pill(colour)
        isClickable = true
        isFocusable = true
        setOnClickListener { onClick() }
    }

    // -- state ---------------------------------------------------------------

    private fun render(snap: GuidanceBus.Snapshot) {
        val live = running
        statusPill.text = if (live) getString(R.string.status_running) else getString(R.string.status_idle)
        statusPill.background = pill(if (live) AMBER else MUTED)
        startButton.text = getString(if (live) R.string.stop_desc else R.string.start_desc)
        startButton.contentDescription = startButton.text

        // Controls that cannot work say so by being disabled, rather than
        // accepting a press and doing nothing - which is exactly how the old
        // screen behaved and exactly why it seemed broken.
        setEnabled(describeButton, live && snap.vlmInstalled)
        setEnabled(readButton, live)
        setEnabled(modelButton, !live)
        modelButton.text = if (live) getString(R.string.model_locked) else getString(R.string.model_label)

        if (!snap.vlmInstalled) {
            describeButton.contentDescription = getString(R.string.describe_missing)
        }

        if (snap.busy.isNotEmpty()) {
            busyLine.text = snap.busy
            busyLine.visibility = View.VISIBLE
        } else {
            busyLine.visibility = View.GONE
        }

        if (snap.lastSpoken.isNotEmpty()) spoken.text = "“${snap.lastSpoken}”"

        if (live) {
            overlay.update(snap.detections, snap.frameWidth, snap.frameHeight)
            stats.text = buildString {
                append(snap.modelId).append("  ·  ").append(snap.backend)
                append("\n")
                append("%.1f Hz".format(snap.hz))
                append("   ").append(snap.inferenceMs).append(" ms")
                append("   skip ").append(snap.skipPercent).append("%")
                append("   ").append(snap.activity.lowercase())
                if (!snap.thermalHeadroom.isNaN()) {
                    append("   thermal %.2f".format(snap.thermalHeadroom))
                }
                append("\n")
                append(snap.detections.size).append(" detected")
                // The number that separates an empty scene from a broken one.
                append("   peak score %.2f".format(snap.maxScore))
            }
        } else {
            overlay.clear()
            stats.text = getString(R.string.stats_idle)
        }
    }

    private fun setEnabled(view: TextView, enabled: Boolean) {
        view.isEnabled = enabled
        view.alpha = if (enabled) 1f else 0.38f
    }

    // -- actions -------------------------------------------------------------

    private fun toggle() {
        val intent = Intent(this, GuidanceService::class.java)
        if (running) {
            stopService(intent)
        } else {
            ContextCompat.startForegroundService(this, intent)
        }
        // The service takes a moment to settle either way; re-read rather than
        // assume, and re-attach the preview once it exists.
        preview.postDelayed({
            GuidanceService.attachPreview(preview.surfaceProvider)
            render(GuidanceBus.snapshot)
        }, 700)
        render(GuidanceBus.snapshot)
    }

    /**
     * Pick which detector runs.
     *
     * Reads the manifests the app actually ships rather than a hard-coded
     * list, so a model dropped into `models/manifests/` appears here without a
     * code change — which is the entire promise of the manifest system, and it
     * was previously unreachable from the app at all.
     */
    private fun chooseModel() {
        if (running) return
        val manifests = runCatching {
            SharedData.listManifests(this).filter { it.task == "detection" && it.loadable }
        }.getOrDefault(emptyList())
        if (manifests.isEmpty()) {
            AlertDialog.Builder(this)
                .setTitle(R.string.model_label)
                .setMessage(R.string.model_none)
                .setPositiveButton(android.R.string.ok, null)
                .show()
            return
        }
        val prefs = getSharedPreferences(GuidanceService.PREFS, Context.MODE_PRIVATE)
        val current = prefs.getString("detector", "yolo11n-coco-320.yaml")
        val files = manifests.map { "${it.id}.yaml" }
        val labels = manifests.map { "${it.id}\n${it.license} · ${it.distribution}" }.toTypedArray()
        val checked = files.indexOf(current).coerceAtLeast(0)

        AlertDialog.Builder(this)
            .setTitle(R.string.model_label)
            .setSingleChoiceItems(labels, checked) { dialog, which ->
                prefs.edit().putString("detector", files[which]).apply()
                dialog.dismiss()
                render(GuidanceBus.snapshot)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun switchLanguage() {
        val prefs = getSharedPreferences(GuidanceService.PREFS, Context.MODE_PRIVATE)
        val next = if (prefs.getString("lang", "en") == "en") "hi" else "en"
        prefs.edit().putString("lang", next).apply()
        if (running) { toggle(); toggle() }   // restart so the new voice loads
    }

    private fun send(action: String) {
        if (!running) return
        startService(Intent(this, GuidanceService::class.java).setAction(action))
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

    // -- hardware keys, so none of the above is required ---------------------

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

    private companion object {
        const val MATCH = ViewGroup.LayoutParams.MATCH_PARENT
        const val WRAP = ViewGroup.LayoutParams.WRAP_CONTENT

        // Dark by default and only dark. This is held up in front of a camera,
        // often outdoors and often at night, and a white screen at night is a
        // torch pointed at whoever is helping. The amber is the one saturated
        // colour, reserved for the single question that matters at a glance:
        // is it running?
        val INK = Color.rgb(16, 19, 21)
        val PAPER = Color.rgb(231, 234, 230)
        val MUTED = Color.rgb(110, 122, 128)
        val AMBER = Color.rgb(229, 168, 60)
        val SLATE = Color.rgb(38, 44, 48)
        val CYAN = Color.rgb(120, 214, 200)
    }
}
