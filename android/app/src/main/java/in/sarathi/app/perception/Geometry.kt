package `in`.sarathi.app.perception

import `in`.sarathi.app.models.SizePrior
import kotlin.math.atan
import kotlin.math.atan2
import kotlin.math.tan

/**
 * Distance and bearing from one camera, without running a depth network.
 *
 * A port of the prototype's geometry, and the numbers are pinned by hand-worked
 * tests there rather than being reimplemented by eye here.
 */
class CameraModel(
    val width: Int,
    val height: Int,
    val hfovDeg: Double = 66.0,
    val mountHeightM: Double = 1.20,
    val pitchDeg: Double = 0.0,
) {
    val fx: Double get() = (width / 2.0) / tan(Math.toRadians(hfovDeg) / 2.0)
    // Square pixels: deriving fy from a vertical FOV would double-count the
    // aspect ratio and skew every distance.
    val fy: Double get() = fx
    val cx: Double get() = width / 2.0
    val cy: Double get() = height / 2.0

    val horizonY: Double get() = cy - fy * tan(Math.toRadians(pitchDeg))

    /**
     * The closest floor the camera can actually see, set by the bottom edge.
     * Further out than people expect - a near-level chest camera cannot see
     * the ground at its own feet.
     */
    val nearestVisibleGroundM: Double? get() = groundDistance(height - 1.0)

    fun bearingDeg(x: Double): Double = Math.toDegrees(atan2(x - cx, fx))

    fun groundDistance(yBottom: Double): Double? {
        val alpha = atan2(yBottom - cy, fy)
        val depression = Math.toRadians(pitchDeg) + alpha
        if (depression <= MIN_DEPRESSION) return null   // at or above the horizon
        return mountHeightM / tan(depression)
    }

    fun distanceFromHeight(pixelHeight: Double, realHeightM: Double): Double? {
        if (pixelHeight <= 1.0 || realHeightM <= 0.0) return null
        return realHeightM * fy / pixelHeight
    }

    companion object {
        private val MIN_DEPRESSION = Math.toRadians(0.6)
        const val MIN_DISTANCE_M = 0.3
        const val MAX_DISTANCE_M = 60.0
    }
}

data class DistanceEstimate(
    val metres: Double?,
    /** "ground" | "size" | "fused" | "bounded" | "none" */
    val source: String,
    val uncertainty: Double = 1.0,
)

object Geometry {

    private fun sane(value: Double?): Double? =
        if (value == null || !value.isFinite() ||
            value < CameraModel.MIN_DISTANCE_M || value > CameraModel.MAX_DISTANCE_M) null
        else value

    /**
     * Two estimators that fail in different situations, which is the point.
     *
     * The ground plane needs the box bottom to be where the object meets the
     * floor; the size prior needs the whole object visible. Fusion prefers
     * whichever is currently trustworthy and records which, so a wrong distance
     * is traceable to its source rather than blamed on "the model".
     */
    /**
     * How far below the horizon a box bottom must sit before its ground-plane
     * distance is trusted, as a fraction of frame height.
     *
     * Found by looking at a picture rather than a log: a wall clock in the
     * self-test frame measured **54.3 m** while a car in the same frame
     * measured 1.2 m. The count was right, the label was right, and only the
     * number was nonsense - which on a phone used with the screen off would
     * have been spoken as fact.
     */
    const val CONTACT_MARGIN_FRACTION = 0.04f

    fun estimate(
        label: String,
        box: FloatArray,          // x1, y1, x2, y2 in frame pixels
        camera: CameraModel,
        priors: Map<String, SizePrior>,
        frameHeight: Int,
    ): DistanceEstimate {
        val prior = priors[label] ?: SizePrior(null, 0.30, true)
        val boxH = (box[3] - box[1]).toDouble().coerceAtLeast(0.0)

        // A box on the frame boundary is cut off: the object continues below
        // the image and its real floor contact is unknown.
        val truncated = box[3] >= frameHeight - 1.5f

        // The contact row must sit far enough below the horizon for the
        // arithmetic to mean anything. `h / tan(depression)` runs to infinity
        // as the bottom edge approaches the horizon, so a few pixels there are
        // worth tens of metres and no detector places a box that precisely.
        //
        // This must stay identical to `_contact_is_usable` in
        // prototype/sarathi/perception/distance.py. The two implementations
        // reading the same manifests is the whole design; the phone quietly
        // computing a different distance from the laptop would defeat it.
        val contactUsable =
            box[3] >= camera.horizonY + frameHeight * CONTACT_MARGIN_FRACTION
        var ground = if (prior.grounded && !truncated && contactUsable) {
            camera.groundDistance(box[3].toDouble())
        } else null
        var size = prior.heightM?.let { camera.distanceFromHeight(boxH, it) }
        ground = sane(ground)
        size = sane(size)

        // A truncated box has its base below the visible frame, so the object
        // must be NEARER than the closest ground point the camera can see. The
        // size prior violates that routinely on partly visible objects, and it
        // errs in the dangerous direction - pushing the very closest things
        // further away.
        if (truncated && prior.grounded && size != null) {
            val limit = sane(camera.groundDistance((frameHeight - 1).toDouble().coerceAtLeast(1.0)))
            if (limit != null && size > limit) {
                return DistanceEstimate(limit, "bounded", 0.6)
            }
        }

        if (ground != null && size != null) {
            val ratio = maxOf(ground, size) / minOf(ground, size).coerceAtLeast(1e-6)
            return if (ratio <= 1.35) DistanceEstimate((ground + size) / 2.0, "fused", 0.12)
            else DistanceEstimate(ground, "ground", 0.30)
        }
        if (ground != null) return DistanceEstimate(ground, "ground", 0.20)
        if (size != null) return DistanceEstimate(size, "size", maxOf(0.15, prior.spread))
        return DistanceEstimate(null, "none", 1.0)
    }

    /**
     * How far to the side of the walking line an object sits.
     *
     * The number that decides whether something is in the way. A pole thirty
     * degrees off at eight metres is four metres to the side and irrelevant;
     * the same thirty degrees at one metre is about to be walked into.
     */
    fun lateralOffsetM(distanceM: Double, bearingDeg: Double): Double =
        distanceM * tan(Math.toRadians(bearingDeg))

    fun clockPosition(bearingDeg: Double): Int {
        var hour = 12 + Math.round(bearingDeg / 30.0).toInt()
        if (hour > 12) hour -= 12
        if (hour < 1) hour += 12
        return hour
    }
}
