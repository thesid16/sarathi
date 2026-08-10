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
import `in`.sarathi.app.perception.LiteRtDetector

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
    private var cameraModel: CameraModel? = null
    private var priors = emptyMap<String, `in`.sarathi.app.models.SizePrior>()

    override fun onCreate() {
        super.onCreate()
        createChannel()
        startForeground(NOTIFICATION_ID, buildNotification())

        val lang = getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("lang", "en")!!
        phrases = SharedData.phraseBook(this, lang)
        priors = SharedData.sizePriors(this)
        val hazards = SharedData.taxonomy(this)
        val bridge = SharedData.labelBridge(this)

        scheduler = Scheduler(power = getSystemService(POWER_SERVICE) as? PowerManager)
        tracker = Tracker()
        saliency = SaliencyEngine()
        voice = Voice(this) { ready ->
            if (ready) {
                voice.setLanguage(lang)
                voice.say(phrases.systemPhrase("starting"), urgent = false)
            }
        }

        detector = runCatching {
            val manifest = SharedData.manifest(this, DETECTOR_MANIFEST)
            val labels = manifest.output?.labels?.let { SharedData.labels(this, "$it.txt") }
                ?: emptyList()
            LiteRtDetector.create(this, manifest, labels, hazards, bridge)
        }.getOrElse {
            Log.w(TAG, "detector unavailable: ${it.message}")
            null
        }

        camera = CameraSource(this)
        camera.start(this) { frame -> onFrame(frame) }
    }

    private fun onFrame(frame: CameraSource.Frame) {
        val now = System.currentTimeMillis()
        val age = now - frame.timestampMs
        val decision = scheduler.decide(frame.luma, frame.width, frame.height, age, now)
        if (!decision.run) return

        val model = detector ?: return
        val bitmap = frame.toBitmap() ?: return
        val camModel = cameraModel ?: CameraModel(bitmap.width, bitmap.height).also {
            cameraModel = it
        }

        val detections = model.detect(bitmap)
        for (det in detections) {
            val estimate = Geometry.estimate(det.label, det.box, camModel, priors, bitmap.height)
            det.distanceM = estimate.metres
            det.distanceSource = estimate.source
            det.bearingDeg = camModel.bearingDeg(((det.box[0] + det.box[2]) / 2.0))
        }
        bitmap.recycle()

        val tracks = tracker.update(detections, now)
        val chosen = saliency.select(tracks, now) ?: return
        val text = phrases.utterance(
            label = chosen.track.label,
            bearingDeg = chosen.track.bearingDeg,
            metres = chosen.track.distanceM,
            urgent = chosen.urgent,
        )
        voice.say(text, urgent = chosen.urgent, earcon = chosen.urgent)
    }

    override fun onDestroy() {
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
    }
}
