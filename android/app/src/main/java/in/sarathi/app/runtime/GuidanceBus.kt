package `in`.sarathi.app.runtime

import android.os.Handler
import android.os.Looper
import `in`.sarathi.app.guidance.Detection

/**
 * What the running pipeline is doing, published for anything that wants to show it.
 *
 * The product is meant to be used with the screen off, and for a long time that
 * was taken as licence to build no screen worth looking at. It is the wrong
 * conclusion. A blind user does not watch the display, but a developer
 * diagnosing a missed kerb does, and so does a sighted friend setting the phone
 * up, and so does anyone being shown the thing for the first time. An app whose
 * internal state is visible only in `adb logcat` is undemonstrable.
 *
 * So the service publishes here and the UI reads. One direction, no binder, no
 * lifecycle coupling: the pipeline runs identically whether or not anything is
 * listening, which keeps the screen genuinely optional rather than merely
 * unhelpful.
 */
object GuidanceBus {

    /** Everything the screen needs in one immutable snapshot. */
    data class Snapshot(
        val running: Boolean = false,
        val modelId: String = "—",
        val backend: String = "—",
        /** Detections from the most recent inference, in upright frame pixels. */
        val detections: List<Detection> = emptyList(),
        val frameWidth: Int = 0,
        val frameHeight: Int = 0,
        val inferenceMs: Long = 0,
        val hz: Double = 0.0,
        val skipPercent: Int = 0,
        val activity: String = "—",
        val thermalHeadroom: Float = Float.NaN,
        /** Highest raw class score, before thresholding. Separates an empty
         *  scene from a broken input, which is the whole reason it is here. */
        val maxScore: Float = 0f,
        val lastSpoken: String = "",
        val lastSpokenAtMs: Long = 0,
        /** Human-readable note about the on-demand tiers: loading, reading, idle. */
        val busy: String = "",
        val vlmInstalled: Boolean = false,
    )

    @Volatile
    var snapshot = Snapshot()
        private set

    private val listeners = mutableSetOf<(Snapshot) -> Unit>()
    private val main = Handler(Looper.getMainLooper())

    fun observe(listener: (Snapshot) -> Unit) {
        synchronized(listeners) { listeners += listener }
        listener(snapshot)
    }

    fun stopObserving(listener: (Snapshot) -> Unit) {
        synchronized(listeners) { listeners -= listener }
    }

    /**
     * Replace the snapshot. Called from the camera thread, delivered on main.
     *
     * `update` takes the previous snapshot so callers can change one field
     * without racing each other to reconstruct the rest - the detection loop,
     * the speech layer and the VLM thread all publish here from different
     * threads and at very different rates.
     */
    fun publish(update: (Snapshot) -> Snapshot) {
        val next = synchronized(this) { update(snapshot).also { snapshot = it } }
        val current = synchronized(listeners) { listeners.toList() }
        if (current.isEmpty()) return
        main.post { current.forEach { it(next) } }
    }

    fun reset() = publish { Snapshot(vlmInstalled = it.vlmInstalled) }
}
