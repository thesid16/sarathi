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
    private lateinit var languageButton: TextView
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

        // Push the overlaid pills below the clock and the notification icons.
        // Without this they are drawn underneath the system status bar - the
        // amber "Guiding" badge landed directly on top of the time, which
        // looks like a broken app before anyone has pressed anything.
        androidx.core.view.ViewCompat.setOnApplyWindowInsetsListener(viewport) { _, insets ->
            val top = insets.getInsets(
                androidx.core.view.WindowInsetsCompat.Type.systemBars()
            ).top
            (statusPill.layoutParams as FrameLayout.LayoutParams)
                .setMargins(dp(14), top + dp(10), 0, 0)
            (busyLine.layoutParams as FrameLayout.LayoutParams)
                .setMargins(0, top + dp(10), dp(14), 0)
            statusPill.requestLayout()
            busyLine.requestLayout()
            insets
        }

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
        languageButton = actionButton(languageLabel(), SLATE) { switchLanguage() }
        languageButton.contentDescription = getString(R.string.lang_desc)
        row2.addView(startButton)
        row2.addView(modelButton)
        row2.addView(languageButton)
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
        // The bus posts to the main thread, so a snapshot published a moment
        // before onDestroy can be delivered a moment after it - and the
        // 700 ms re-attach below outlives a fast rotate-and-back.
        if (isFinishing || isDestroyed) return
        val live = running
        statusPill.text = if (live) getString(R.string.status_running) else getString(R.string.status_idle)
        statusPill.background = pill(if (live) AMBER else MUTED)
        startButton.text = getString(if (live) R.string.stop_desc else R.string.start_desc)
        startButton.contentDescription = startButton.text

        // Controls that cannot work say so by being disabled, rather than
        // accepting a press and doing nothing - which is exactly how the old
        // screen behaved and exactly why it seemed broken.
        // Enabled whenever they can do something. A disabled control is
        // indistinguishable from a broken one to someone who just pressed it,
        // and both on-demand tiers can start guidance themselves - so the only
        // genuine blocker is weights that are not installed.
        setEnabled(describeButton, vlmAvailable())
        setEnabled(readButton, true)
        setEnabled(modelButton, !live)
        modelButton.text = getString(R.string.model_label)

        describeButton.contentDescription =
            if (vlmAvailable()) getString(R.string.describe_label)
            else getString(R.string.describe_missing)

        if (snap.busy.isNotEmpty()) {
            busyLine.text = snap.busy
            busyLine.visibility = View.VISIBLE
        } else {
            busyLine.visibility = View.GONE
        }

        if (snap.lastSpoken.isNotEmpty()) {
            spoken.text = "“${snap.lastSpoken}”"
        } else {
            // Running with nothing said yet is the normal state on a scene with
            // no hazard in it, and the line below says why. Stopped is stopped.
            spoken.text = getString(
                if (live) R.string.listening else R.string.spoken_placeholder
            )
        }

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
                    // Named, not just numbered. A viewer seeing 1.0 Hz where
                    // the app claims 8 concludes it is broken; the honest
                    // explanation is that the platform is close to throttling
                    // and the governor got there first. Charging is the usual
                    // cause on a desk - a Pixel 8a on AC sits at 0.97 with the
                    // skin at only 34 C, because the CPU cluster is at 71.
                    if (snap.thermalHeadroom >= 0.60f) append(" (heat-limited)")
                }
                // Measured, not configured. Every ground-plane distance is
                // computed from this angle, so it belongs on screen next to
                // the distances it produces.
                append(
                    if (snap.pitchDeg.isNaN()) "   tilt —"
                    else "   tilt %.0f°".format(snap.pitchDeg)
                )
                append("\n")
                append(snap.detections.size).append(" detected")
                // The number that separates an empty scene from a broken one.
                append("   peak score %.2f".format(snap.maxScore))
                // And why it is not talking, which is otherwise unknowable.
                if (snap.quietReason.isNotEmpty()) {
                    append("\nquiet — ").append(snap.quietReason)
                }
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
        // The service takes a moment to settle either way, so re-read rather
        // than assume. The preview surface is handed over as soon as the
        // service object exists; CameraSource holds it until its own use case
        // is built, so there is no window to miss.
        preview.postDelayed({
            if (isFinishing || isDestroyed) return@postDelayed
            GuidanceService.attachPreview(preview.surfaceProvider)
            render(GuidanceBus.snapshot)
        }, 400)
        preview.postDelayed({
            if (isFinishing || isDestroyed) return@postDelayed
            GuidanceService.attachPreview(preview.surfaceProvider)
        }, 1_500)
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
        // Only models whose weights are actually present.
        //
        // Listing every detection manifest would offer yolox-nano, which has
        // no .tflite in this build - selecting it starts guidance with no
        // detector at all, and the app then runs silently while appearing
        // completely healthy. A control that can be pressed must do what it
        // says; one that cannot should not be on the screen.
        val manifests = runCatching {
            SharedData.listManifests(this)
                .filter { it.task == "detection" && it.loadable && weightsPresent(it) }
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
        val current = prefs.getString("detector", DEFAULT_DETECTOR)
        val files = manifests.map { "${it.id}.yaml" }
        val labels = manifests.map { "${it.id}\n${it.license} · ${it.distribution}" }.toTypedArray()
        // A stored value no longer in the list - a stale preference, or
        // sideloaded weights since deleted - would otherwise be coerced to
        // index 0, showing the first model as chosen while the service
        // faithfully loads the missing one and detects nothing. Repair the
        // preference instead of misreporting it.
        var checked = files.indexOf(current)
        if (checked < 0) {
            checked = 0
            prefs.edit().putString("detector", files[0]).apply()
        }

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

    /** Whether this manifest's Android weights exist, bundled or sideloaded. */
    private fun weightsPresent(manifest: `in`.sarathi.app.models.ModelManifest): Boolean {
        val file = manifest.fileFor("android") ?: return false
        val inAssets = runCatching {
            assets.list("models")?.contains(file) == true
        }.getOrDefault(false)
        return inAssets || java.io.File(java.io.File(filesDir, "models"), file).exists()
    }

    /**
     * Switch between English and Hindi.
     *
     * Restarts the service when running, because the phrase book, the TTS
     * voice and the OCR script are all chosen at startup - Devanagari is a
     * separate recogniser, not a flag on the Latin one, so a live swap would
     * leave the app reading Hindi signs with an English model and finding
     * nothing.
     */
    /**
     * What the language button offers, derived from what is stored.
     *
     * Built from the preference rather than hardcoded, because a label that
     * only updates on click is wrong on every launch after the first: prefs
     * say Hindi, the button still reads "हिन्दी" - offering to switch to the
     * language already in use - and pressing it switches back to English.
     */
    private fun languageLabel(): String {
        val current = getSharedPreferences(GuidanceService.PREFS, Context.MODE_PRIVATE)
            .getString("lang", "en")
        return if (current == "hi") "English" else getString(R.string.lang_short)
    }

    private fun switchLanguage() {
        val prefs = getSharedPreferences(GuidanceService.PREFS, Context.MODE_PRIVATE)
        val next = if (prefs.getString("lang", "en") == "en") "hi" else "en"
        prefs.edit().putString("lang", next).apply()
        languageButton.text = languageLabel()
        if (running) {
            stopService(Intent(this, GuidanceService::class.java))
            preview.postDelayed({
                if (isFinishing || isDestroyed) return@postDelayed
                ContextCompat.startForegroundService(
                    this, Intent(this, GuidanceService::class.java)
                )
                preview.postDelayed({
                    if (isFinishing || isDestroyed) return@postDelayed
                    GuidanceService.attachPreview(preview.surfaceProvider)
                }, 700)
            }, 400)
        }
    }

    /**
     * Ask the service to do something, starting it first if it is not running.
     *
     * Refusing to act while stopped was technically defensible and practically
     * indistinguishable from a broken button: the user presses "Describe
     * scene", nothing happens, and there is no way to learn why. Pressing it
     * plainly means "describe the scene", and turning the camera on is part of
     * doing that.
     */
    private fun send(action: String) {
        val intent = Intent(this, GuidanceService::class.java).setAction(action)
        if (running) {
            startService(intent)
            return
        }
        toggle()
        // The camera and the model need a moment; the request is forwarded once
        // the service exists rather than dropped.
        preview.postDelayed({
            if (isFinishing || isDestroyed) return@postDelayed
            startService(intent)
        }, 1_200)
    }

    /** Whether the scene-description weights are on this device. */
    private fun vlmAvailable(): Boolean = runCatching {
        val manifest = SharedData.manifest(this, "gemma-4-e2b-vlm.yaml")
        val file = manifest.fileFor("android") ?: return false
        java.io.File(java.io.File(filesDir, "models"), file).exists()
    }.getOrDefault(false)

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

        const val DEFAULT_DETECTOR = "yolo11n-coco-320.yaml"
    }
}
