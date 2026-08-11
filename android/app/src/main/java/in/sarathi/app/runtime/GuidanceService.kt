package `in`.sarathi.app.runtime

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import `in`.sarathi.app.MainActivity
import `in`.sarathi.app.R
import `in`.sarathi.app.camera.CameraSource
import `in`.sarathi.app.guidance.SaliencyEngine
import `in`.sarathi.app.guidance.Tracker
import `in`.sarathi.app.guidance.Voice
import `in`.sarathi.app.models.PhraseBook
import `in`.sarathi.app.models.SharedData
import `in`.sarathi.app.perception.CameraModel
import `in`.sarathi.app.perception.Geometry
import `in`.sarathi.app.perception.Tilt
import `in`.sarathi.app.perception.LiteRtDetector
import `in`.sarathi.app.ocr.TextReader
import `in`.sarathi.app.vlm.SceneDescriber
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * Guidance as a foreground service, so it survives the screen going off.
 *
 * For this user the screen is pure waste: on a mid-range phone an always-on
 * display can consume more power than the inference does, and they cannot read
 * it anyway. The app is designed to be operated entirely with the volume and
 * headset buttons, screen dark - which is why this is a service with an
 * activity attached rather than an activity that happens to use the camera.
 */
class GuidanceService : LifecycleService() {

    private lateinit var camera: CameraSource
    private lateinit var scheduler: Scheduler
    private lateinit var tracker: Tracker
    private lateinit var saliency: SaliencyEngine
    private lateinit var voice: Voice
    private lateinit var phrases: PhraseBook

    private var detector: LiteRtDetector? = null
    private var describer: SceneDescriber? = null
    private var reader: TextReader? = null

    /**
     * A pending "what is in front of me?" press.
     *
     * A flag rather than a queue, and read before the scheduler's gate rather
     * than after. Standing still, the gate drops almost every frame and the
     * keepalive floor is 0.2 Hz - so a request served after the gate could sit
     * for five seconds, which to someone who just pressed a button is a broken
     * device. Read first, it is served by the next camera frame.
     */
    private val describeRequested = AtomicBoolean(false)
    private val readRequested = AtomicBoolean(false)
    private var cameraModel: CameraModel? = null
    private lateinit var tilt: Tilt
    private var priors = emptyMap<String, `in`.sarathi.app.models.SizePrior>()

    // Field diagnostics. This runs on a phone in a pocket with the screen off,
    // so "it feels wrong" is the only bug report a user can give. A periodic
    // line in logcat is the difference between diagnosing that and guessing.
    private var lastReportAt = 0L
    private var detectionCount = 0L
    private var utteranceCount = 0L
    private var inferenceMsTotal = 0L
    private var lastMaxScore = 0f

    override fun onCreate() {
        super.onCreate()
        instance = this
        createChannel()
        startForeground(NOTIFICATION_ID, buildNotification())

        val lang = getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("lang", "en")!!
        phrases = SharedData.phraseBook(this, lang)
        priors = SharedData.sizePriors(this)
        val hazards = SharedData.taxonomy(this)
        val bridge = SharedData.labelBridge(this)

        tilt = Tilt(this).also { it.start() }
        scheduler = Scheduler(power = getSystemService(POWER_SERVICE) as? PowerManager)
        tracker = Tracker()
        saliency = SaliencyEngine()
        voice = Voice(this) { ready ->
            if (ready) {
                voice.setLanguage(lang)
                voice.say(phrases.systemPhrase("starting"), urgent = false)
            }
        }

        // Falls back when the stored choice cannot be honoured. The picker
        // only offers models whose weights exist, but a preference outlives
        // the build that wrote it: an app updated with a model removed would
        // otherwise start, report itself healthy, and detect nothing.
        val prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val stored = prefs.getString("detector", DETECTOR_MANIFEST) ?: DETECTOR_MANIFEST
        val detectorManifest = if (manifestUsable(stored)) stored else {
            Log.w(TAG, "stored detector $stored is unusable; falling back to $DETECTOR_MANIFEST")
            prefs.edit().putString("detector", DETECTOR_MANIFEST).apply()
            DETECTOR_MANIFEST
        }
        detector = runCatching {
            val manifest = SharedData.manifest(this, detectorManifest)
            val labels = manifest.output?.labels?.let { SharedData.labels(this, "$it.txt") }
                ?: emptyList()
            LiteRtDetector.create(this, manifest, labels, hazards, bridge)
        }.getOrElse {
            Log.w(TAG, "detector unavailable: ${it.message}")
            null
        }
        Log.i(TAG, if (detector != null) "detector loaded: $detectorManifest"
                   else "running WITHOUT a detector - camera and speech only")
        GuidanceBus.publish {
            it.copy(
                running = true,
                modelId = detectorManifest.removeSuffix(".yaml"),
                backend = detector?.backend ?: "none",
            )
        }

        // Constructed always, loaded never - the engine is built on the first
        // press and released again when idle. The manifest is read here only so
        // the app can answer "is it installed?" without touching LiteRT-LM.
        describer = runCatching {
            SceneDescriber(this, SharedData.manifest(this, VLM_MANIFEST))
        }.getOrElse {
            Log.w(TAG, "describer unavailable: ${it.message}")
            null
        }
        val vlmReady = describer?.isInstalled() == true
        Log.i(TAG, "scene description: " + if (vlmReady) "weights present" else "not installed")
        GuidanceBus.publish { it.copy(vlmInstalled = vlmReady) }

        // The recogniser is script-specific and the language can change, so it
        // is built here alongside the phrase book rather than lazily.
        reader = runCatching { TextReader(lang) }.getOrNull()
        reader?.let { ocr ->
            // Off the main thread: the first call makes Play Services deliver
            // the recognition model, which is a network fetch on a device that
            // has never used it.
            thread(name = "sarathi-ocr-selftest", isDaemon = true) {
                Log.i(TAG, ocr.selfTest(this))
            }
        }
        val surveyMarker = java.io.File(filesDir, "survey")
        if (surveyMarker.exists()) {
            surveyMarker.delete()
            runCatching {
                LiteRtDetector.runSurvey(this, SharedData.manifest(this, DETECTOR_MANIFEST))
            }.onFailure { Log.w(TAG, "survey failed: $it") }
        }
        detector?.let {
            Log.i(TAG, "backend: ${it.backend}")
            Log.i(TAG, it.selfTest(this))
        }

        camera = CameraSource(this)
        camera.start(this) { frame -> onFrame(frame) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        when (intent?.action) {
            ACTION_DESCRIBE -> requestDescription()
            ACTION_READ -> requestReading()
        }
        return START_STICKY
    }

    /** Whether a manifest exists, is loadable, and has weights on this device. */
    private fun manifestUsable(fileName: String): Boolean = runCatching {
        val manifest = SharedData.manifest(this, fileName)
        if (!manifest.loadable) return false
        val weights = manifest.fileFor("android") ?: return false
        val inAssets = assets.list("models")?.contains(weights) == true
        inAssets || java.io.File(java.io.File(filesDir, "models"), weights).exists()
    }.getOrDefault(false)

    private fun requestReading() {
        if (reader == null) {
            voice.say(phrases.systemPhrase("no_text"), urgent = false)
            return
        }
        voice.say(phrases.systemPhrase("reading"), urgent = false)
        GuidanceBus.publish { it.copy(busy = "Reading text…") }
        readRequested.set(true)
    }

    private fun serveReading(bitmap: android.graphics.Bitmap) {
        val ocr = reader ?: return
        thread(name = "sarathi-ocr", isDaemon = true) {
            val text = ocr.read(bitmap)
            bitmap.recycle()
            GuidanceBus.publish {
                it.copy(busy = "", lastSpoken = text ?: "(no text found)",
                    lastSpokenAtMs = System.currentTimeMillis())
            }
            voice.say(text ?: phrases.systemPhrase("no_text"), urgent = false)
            Log.i(TAG, "ocr ${ocr.lastReadMs}ms")
        }
    }

    private fun requestDescription() {
        val vlm = describer
        if (vlm == null || !vlm.isInstalled()) {
            voice.say(phrases.systemPhrase("vlm_missing"), urgent = false)
            return
        }
        // Spoken before any work starts, and picked to match the actual wait.
        // Silence from a device you cannot look at is indistinguishable from a
        // device that has died - but so is a cheerful "Looking" followed by
        // forty-nine seconds of nothing. Measured on a Pixel 8a: 49 s the very
        // first time while GPU kernels compile, 4.4 s on every launch after,
        // and none at all while the engine is still resident.
        voice.say(
            phrases.systemPhrase(if (vlm.isLoaded()) "describing" else "describing_first_run"),
            urgent = false,
        )
        GuidanceBus.publish {
            it.copy(busy = if (vlm.isLoaded()) "Describing…" else "Loading scene model…")
        }
        describeRequested.set(true)
    }

    private fun onFrame(frame: CameraSource.Frame) {
        val now = System.currentTimeMillis()
        if (describeRequested.compareAndSet(true, false)) {
            frame.toBitmap()?.let { serveDescription(it) }
        }
        if (readRequested.compareAndSet(true, false)) {
            frame.toBitmap()?.let { serveReading(it) }
        }
        describer?.trimIfIdle(now)
        val age = now - frame.timestampMs
        val decision = scheduler.decide(frame.luma, frame.width, frame.height, age, now)
        if (!decision.run) { report(now); return }

        val model = detector ?: return
        val bitmap = frame.toBitmap() ?: return
        // Rebuilt whenever the frame dimensions change, not cached once.
        //
        // The bitmap is rotated upright, so on a phone turned from portrait to
        // landscape it goes from 480x640 to 640x480. A camera model built from
        // the first frame and kept would then hold the wrong principal point
        // and the wrong horizon row for every frame after the turn - and the
        // ground-plane distances derived from that horizon would be quietly
        // wrong rather than absent.
        // Rebuilt when the frame size changes OR the phone is tilted, because
        // pitch is not a correction to the distance - it is most of it. Two
        // degrees is about the smallest change worth recomputing for; a walking
        // gait swings several, and the sensor is already smoothed.
        val pitch = if (tilt.live) tilt.pitchDeg else 0.0
        val camModel = cameraModel
            ?.takeIf {
                it.width == bitmap.width && it.height == bitmap.height &&
                    kotlin.math.abs(it.pitchDeg - pitch) < 2.0
            }
            ?: CameraModel(bitmap.width, bitmap.height, pitchDeg = pitch).also {
                cameraModel = it
            }

        val detections = model.detect(bitmap)
        val frameW = bitmap.width
        val frameH = bitmap.height
        detectionCount += detections.size
        inferenceMsTotal += model.lastInferenceMs
        lastMaxScore = model.lastMaxScore
        for (det in detections) {
            val estimate = Geometry.estimate(det.label, det.box, camModel, priors, bitmap.height)
            det.distanceM = estimate.metres
            det.distanceSource = estimate.source
            det.bearingDeg = camModel.bearingDeg(((det.box[0] + det.box[2]) / 2.0))
        }
        bitmap.recycle()

        GuidanceBus.publish {
            it.copy(
                detections = detections,
                frameWidth = frameW,
                frameHeight = frameH,
                inferenceMs = model.lastInferenceMs,
                maxScore = model.lastMaxScore,
            )
        }

        val tracks = tracker.update(detections, now)
        val chosen = saliency.select(tracks, now)
        GuidanceBus.publish { it.copy(quietReason = saliency.lastReason) }
        if (chosen == null) return
        val text = phrases.utterance(
            label = chosen.track.label,
            bearingDeg = chosen.track.bearingDeg,
            metres = chosen.track.distanceM,
            urgent = chosen.urgent,
        )
        val spoken = voice.say(text, urgent = chosen.urgent, earcon = chosen.urgent)
        utteranceCount++
        if (spoken) {
            GuidanceBus.publish { it.copy(lastSpoken = text, lastSpokenAtMs = now) }
        }
        Log.i(TAG, "say${if (spoken) "" else " [dropped]"}: \"$text\"  " +
            "(${chosen.track.label} ${chosen.track.distanceM?.let { "%.1f m".format(it) } ?: "?"} " +
            "${chosen.track.bearingDeg?.let { "%.0f deg".format(it) } ?: "?"} " +
            "score=%.2f${if (chosen.urgent) " URGENT" else ""})".format(chosen.score))
        report(now)
    }

    /**
     * Run the VLM off the camera thread, and let guidance carry on regardless.
     *
     * Description takes seconds. Blocking the frame loop on it would suspend
     * hazard detection for exactly as long - trading the safety feature for the
     * convenience one, which is the wrong way round.
     */
    private fun serveDescription(bitmap: android.graphics.Bitmap) {
        val vlm = describer ?: return
        thread(name = "sarathi-vlm", isDaemon = true) {
            val answer = vlm.describe(
                bitmap,
                phrases.systemPhrase("describe_prompt"),
                phrases.systemPhrase("describe_system"),
            )
            bitmap.recycle()
            GuidanceBus.publish {
                it.copy(busy = "", lastSpoken = answer ?: "(no description)",
                    lastSpokenAtMs = System.currentTimeMillis())
            }
            // Not urgent: a hazard warning arriving mid-description should
            // interrupt it, never the other way round.
            voice.say(answer ?: phrases.systemPhrase("no_description"), urgent = false)
            Log.i(TAG, "vlm backend=${vlm.backend} load=${vlm.loadMs}ms " +
                "describe=${vlm.lastDescribeMs}ms")
        }
    }

    /** One line every few seconds: what the pipeline is actually doing. */
    private fun report(now: Long) {
        if (now - lastReportAt < REPORT_INTERVAL_MS) return
        lastReportAt = now
        GuidanceBus.publish {
            it.copy(
                hz = scheduler.targetHz(scheduler.lastPressure),
                skipPercent = (scheduler.skipRate * 100).toInt(),
                activity = scheduler.activity.name,
                thermalHeadroom = scheduler.lastPressure.toFloat(),
                pitchDeg = if (tilt.live) tilt.pitchDeg else Double.NaN,
            )
        }
        val ran = scheduler.framesRan.coerceAtLeast(1)
        Log.i(TAG, "frames=${scheduler.framesConsidered} ran=${scheduler.framesRan} " +
            "(skip ${"%.0f".format(scheduler.skipRate * 100)}%) " +
            "activity=${scheduler.activity} " +
            "hz=${"%.1f".format(scheduler.targetHz(scheduler.lastPressure))} " +
            "thermal=${"%.2f".format(scheduler.lastPressure)} " +
            "detections=$detectionCount maxScore=${"%.3f".format(lastMaxScore)} " +
            "inference=${inferenceMsTotal / ran}ms " +
            "said=$utteranceCount dropped=${voice.droppedCount} " +
            "skips=${scheduler.skips}")
    }

    override fun onDestroy() {
        instance = null
        if (::tilt.isInitialized) tilt.stop()
        GuidanceBus.reset()
        describer?.close()
        reader?.close()
        camera.stop()
        detector?.close()
        voice.say(phrases.systemPhrase("stopping"), urgent = false)
        voice.shutdown()
        super.onDestroy()
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID, getString(R.string.notification_channel),
            NotificationManager.IMPORTANCE_LOW,
        ).apply { setShowBadge(false) }
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(open)
            .setOngoing(true)
            .setSilent(true)
            .build()
    }

    companion object {
        private const val TAG = "SarathiService"
        const val PREFS = "sarathi"
        private const val CHANNEL_ID = "guidance"
        private const val NOTIFICATION_ID = 1
        private const val DETECTOR_MANIFEST = "yolo11n-coco-320.yaml"
        private const val VLM_MANIFEST = "gemma-4-e2b-vlm.yaml"

        /**
         * The live service, or null.
         *
         * The screen asks this whether guidance is actually running, instead of
         * keeping its own boolean. A local flag in the activity drifts the
         * moment the service is stopped from its notification or killed by the
         * system, and then the on-demand buttons silently do nothing because
         * they guard on a flag that disagrees with reality.
         */
        @Volatile
        var instance: GuidanceService? = null
            private set

        val isRunning: Boolean get() = instance != null

        /** Attach or release a preview surface. No-op when nothing is running. */
        fun attachPreview(provider: androidx.camera.core.Preview.SurfaceProvider?) {
            instance?.let { if (it::camera.isInitialized) it.camera.setPreviewSurface(provider) }
        }

        /** Volume-up, short press. Forwarded from the activity. */
        const val ACTION_DESCRIBE = "in.sarathi.app.DESCRIBE"

        /** Volume-up, held. */
        const val ACTION_READ = "in.sarathi.app.READ"
        private const val REPORT_INTERVAL_MS = 5_000L
    }
}
