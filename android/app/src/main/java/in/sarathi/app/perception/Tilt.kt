package `in`.sarathi.app.perception

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.asin
import kotlin.math.sqrt

/**
 * How far below horizontal the camera is actually pointing, from gravity.
 *
 * This is the single biggest source of wrong distances, and it was previously
 * a constant in a config file. The ground-plane estimate is
 * `height / tan(depression)`, and depression is the sum of the camera's pitch
 * and the angle of the pixel below centre - so the pitch is not a minor
 * correction, it *is* the measurement. On a Pixel 8a the same box bottom gives
 * 4.4 m at 0 degrees and 1.2 m at 30 degrees. Assuming a fixed value means
 * every distance is wrong by whatever the user's actual posture differs from
 * the guess, which on a handheld phone is most of the range.
 *
 * Gravity gives it directly, at no meaningful power cost, and it is the one
 * quantity here that needs no calibration: the accelerometer is measuring a
 * physical constant.
 *
 * The device frame has +Z out of the screen, so the back camera looks along
 * -Z. The angle of that axis below horizontal is
 *
 *     pitch = asin(-g_z / |g|)
 *
 * which is 0 when the phone is held upright with the camera looking at the
 * horizon, and 90 when it is lying flat on a table with the camera pointing at
 * the floor.
 */
class Tilt(context: Context) : SensorEventListener {

    private val sensors = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val sensor: Sensor? =
        sensors?.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sensors?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    /**
     * Downward pitch in degrees, smoothed.
     *
     * Starts at the old fixed default so a device with no usable sensor
     * behaves exactly as before rather than worse.
     */
    @Volatile
    var pitchDeg: Double = 0.0
        private set

    /** Whether a real reading has arrived. Distances are guesses until it has. */
    @Volatile
    var live: Boolean = false
        private set

    fun start() {
        val target = sensor ?: return
        // SENSOR_DELAY_UI is ~60 ms, far faster than the 1-8 Hz the detector
        // runs at, and cheap. Gravity is a low-rate signal anyway.
        sensors?.registerListener(this, target, SensorManager.SENSOR_DELAY_UI)
    }

    fun stop() {
        sensors?.unregisterListener(this)
        live = false
    }

    override fun onSensorChanged(event: SensorEvent) {
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]
        val magnitude = sqrt((x * x + y * y + z * z).toDouble())
        if (magnitude < 1e-3) return

        val raw = Math.toDegrees(asin((z / magnitude).coerceIn(-1.0, 1.0)))

        // Low-pass, because a walking gait swings the phone several degrees per
        // step. Un-smoothed, announced distances would breathe in and out at
        // stride frequency, which sounds like the system changing its mind.
        pitchDeg = if (live) pitchDeg * (1 - SMOOTHING) + raw * SMOOTHING else raw
        live = true
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private companion object {
        const val SMOOTHING = 0.15
    }
}
