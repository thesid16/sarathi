package `in`.sarathi.app.runtime

import android.os.Build
import android.os.PowerManager
import kotlin.math.abs

/**
 * Deciding when NOT to run the model.
 *
 * Everything else makes inference better; this makes it happen less often,
 * which is the only reason the app can run for a day. Measured on the desktop
 * prototype: a stationary user needs inference on 3% of frames, a walking one
 * on 18%.
 *
 * Gates run cheapest-first so a frame that will be skipped is skipped for the
 * least possible cost: staleness, then motion at about 0.1 ms, then rate, then
 * thermal.
 */
class Scheduler(
    private val config: Config = Config(),
    private val power: PowerManager? = null,
) {
    data class Config(
        val maxInferenceHz: Double = 8.0,
        val idleInferenceHz: Double = 1.0,
        /**
         * Floor rate that runs regardless of any gate. A blind user cannot
         * distinguish "nothing to report" from "stopped working", so no gate
         * is permitted to silence the system completely.
         */
        val keepaliveHz: Double = 0.2,
        val maxFrameAgeMs: Long = 250,
        val motionEnabled: Boolean = true,
        val motionThreshold: Double = 0.012,
        val settleMs: Long = 2000,
        val thermalEnabled: Boolean = true,
        val thermalSoft: Double = 0.30,
        val thermalHard: Double = 0.70,
        val depthHz: Double = 2.0,
    )

    enum class Skip { RAN, STALE, STATIC, RATE, THERMAL }
    enum class Activity { IDLE, MOVING }

    data class Decision(
        val run: Boolean,
        val reason: Skip,
        val activity: Activity,
        val targetHz: Double,
        val keepalive: Boolean = false,
    )

    private val gate = MotionGate(config.motionThreshold)
    private var lastInference = 0L
    private var lastDepth = 0L
    private var lastMotionAt = 0L
    var activity: Activity = Activity.MOVING
        private set

    var framesConsidered = 0L; private set
    var framesRan = 0L; private set
    val skips = mutableMapOf<Skip, Long>()

    /**
     * Thermal headroom, 0 cool to 1 about to throttle.
     *
     * Android exposes this directly from API 30. Degrading before the platform
     * throttles keeps the announcement rate falling smoothly instead of the OS
     * yanking it - a step change in how often the system speaks is audible and
     * unsettling; a ramp is not.
     */
    private fun thermalPressure(): Double {
        val pm = power
        if (!config.thermalEnabled || pm == null) return 0.0
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return 0.0
        val headroom = runCatching { pm.getThermalHeadroom(10) }.getOrNull() ?: return 0.0
        if (headroom.isNaN()) return 0.0
        return headroom.toDouble().coerceIn(0.0, 1.0)
    }

    fun targetHz(pressure: Double): Double {
        val base = if (activity == Activity.MOVING) config.maxInferenceHz else config.idleInferenceHz
        if (!config.thermalEnabled || pressure <= config.thermalSoft) return base
        if (pressure >= config.thermalHard) return config.idleInferenceHz
        val span = (config.thermalHard - config.thermalSoft).coerceAtLeast(1e-6)
        val fraction = (pressure - config.thermalSoft) / span
        return base - (base - config.idleInferenceHz) * fraction
    }

    /**
     * @param luma downscaled greyscale plane, for the motion gate
     * @param frameAgeMs how stale this frame already is
     */
    fun decide(luma: ByteArray, width: Int, height: Int, frameAgeMs: Long, nowMs: Long): Decision {
        framesConsidered++

        if (frameAgeMs > config.maxFrameAgeMs) return skip(Skip.STALE, 0.0)

        var moved = true
        if (config.motionEnabled) {
            moved = gate.changed(luma, width, height)
            if (moved) lastMotionAt = nowMs
            activity = if (nowMs - lastMotionAt < config.settleMs) Activity.MOVING else Activity.IDLE
        }

        val pressure = thermalPressure()
        val hz = targetHz(pressure)
        val since = nowMs - lastInference

        // Keepalive is checked before the gates so nothing can suppress it.
        if (config.keepaliveHz > 0 && since >= (1000.0 / config.keepaliveHz)) {
            lastInference = nowMs
            framesRan++
            return Decision(true, Skip.RAN, activity, hz, keepalive = true)
        }

        // 5% tolerance: frames never land on exact intervals, and comparing
        // strictly costs about a fifth of the target rate - a camera at twice
        // the target ends up delivering two thirds of it.
        if (hz <= 0 || since < (1000.0 / hz) * 0.95) {
            val reason = if (config.thermalEnabled && pressure > config.thermalSoft)
                Skip.THERMAL else Skip.RATE
            return skip(reason, hz)
        }

        if (config.motionEnabled && !moved && activity == Activity.IDLE) return skip(Skip.STATIC, hz)

        lastInference = nowMs
        framesRan++
        return Decision(true, Skip.RAN, activity, hz)
    }

    /** Whether the Tier 2 depth pass has earned its cost. */
    fun shouldRunDepth(nowMs: Long): Boolean {
        if (config.depthHz <= 0) return false
        if (activity == Activity.IDLE) return false   // steps do not appear while standing still
        if (nowMs - lastDepth < 1000.0 / config.depthHz) return false
        lastDepth = nowMs
        return true
    }

    private fun skip(reason: Skip, hz: Double): Decision {
        skips[reason] = (skips[reason] ?: 0) + 1
        return Decision(false, reason, activity, hz)
    }

    val skipRate: Double
        get() = if (framesConsidered == 0L) 0.0 else 1.0 - framesRan.toDouble() / framesConsidered
}

/**
 * Frame-difference gate on a small greyscale downscale.
 *
 * Deliberately crude: the question is not "what moved" but "is this worth
 * looking at properly", and anything more sophisticated would cost a
 * meaningful fraction of the inference it exists to avoid.
 */
class MotionGate(private val threshold: Double, private val size: Int = 64) {
    private var previous: FloatArray? = null
    var lastScore: Double = 1.0
        private set

    fun changed(luma: ByteArray, width: Int, height: Int): Boolean {
        val current = FloatArray(size * size)
        val sx = width.toFloat() / size
        val sy = height.toFloat() / size
        for (y in 0 until size) {
            val srcY = (y * sy).toInt().coerceIn(0, height - 1)
            for (x in 0 until size) {
                val srcX = (x * sx).toInt().coerceIn(0, width - 1)
                val index = srcY * width + srcX
                current[y * size + x] =
                    (luma.getOrElse(index) { 0 }.toInt() and 0xFF) / 255f
            }
        }
        val prev = previous
        previous = current
        if (prev == null) {
            // Nothing to compare against. Treat the first frame as changed so
            // the pipeline always produces one result at startup rather than
            // sitting silent until something moves.
            lastScore = 1.0
            return true
        }
        var sum = 0.0
        for (i in current.indices) sum += abs(current[i] - prev[i])
        lastScore = sum / current.size
        return lastScore >= threshold
    }

    fun reset() { previous = null; lastScore = 1.0 }
}
