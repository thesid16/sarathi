package `in`.sarathi.app.guidance

import `in`.sarathi.app.models.Hazard
import kotlin.math.max
import kotlin.math.min

data class Detection(
    var label: String,
    val score: Float,
    val classId: Int,
    val box: FloatArray,            // x1, y1, x2, y2 in frame pixels
    var distanceM: Double? = null,
    var distanceSource: String? = null,
    var bearingDeg: Double? = null,
    var hazard: Hazard = Hazard.LOW,
) {
    override fun equals(other: Any?) = other is Detection && other.box.contentEquals(box) &&
        other.label == label && other.score == score
    override fun hashCode() = box.contentHashCode() * 31 + label.hashCode()
}

/**
 * One object followed across frames.
 *
 * Tracking exists here less for tracking than for two things it enables: a
 * track must be seen twice before anything is said about it, so single-frame
 * false positives never reach speech; and closing speed, because a car 8 m
 * away approaching at 6 m/s is not the same fact as a car parked 8 m away.
 */
class Track(
    val id: Int,
    var label: String,
    val classId: Int,
    var box: FloatArray,
    var hazard: Hazard,
    var distanceM: Double?,
    var bearingDeg: Double?,
    val firstSeenMs: Long,
) {
    var lastSeenMs: Long = firstSeenMs
    var hits: Int = 1
    var misses: Int = 0
    private val history = ArrayDeque<Pair<Long, Double>>()

    fun confirmed(minHits: Int) = hits >= minHits

    fun record(distance: Double?, atMs: Long) {
        if (distance == null) { this.distanceM = null; return }
        // Light smoothing: distance drives what gets spoken, and an unsmoothed
        // estimate makes the same object drift between "two metres" and "three
        // metres" on consecutive frames.
        distanceM = distanceM?.let { 0.6 * it + 0.4 * distance } ?: distance
        history.addLast(atMs to distanceM!!)
        while (history.size > 8) history.removeFirst()
    }

    /**
     * Metres per second of approach, positive means nearer.
     *
     * Least-squares over recent history rather than a two-point difference:
     * distance estimates are noisy and a single bad frame should not read as a
     * two-metre lunge.
     */
    val closingSpeedMps: Double?
        get() {
            if (history.size < 3) return null
            val ts = history.map { it.first / 1000.0 }
            val ds = history.map { it.second }
            if (ts.last() - ts.first() < 0.2) return null
            val meanT = ts.average(); val meanD = ds.average()
            var num = 0.0; var den = 0.0
            for (i in ts.indices) {
                num += (ts[i] - meanT) * (ds[i] - meanD)
                den += (ts[i] - meanT) * (ts[i] - meanT)
            }
            if (den <= 0.0) return null
            return -(num / den)
        }

    fun timeToContactS(): Double? {
        val speed = closingSpeedMps ?: return null
        val d = distanceM ?: return null
        if (speed <= 0.05) return null
        return d / speed
    }
}

/** Greedy per-class IoU tracker. No motion model: at 5-8 Hz objects move a few
 *  pixels between frames, and a Kalman filter would add state and failure modes
 *  to solve a problem this product does not have. */
class Tracker(
    private val iouThreshold: Float = 0.25f,
    private val minHits: Int = 2,
    private val maxMisses: Int = 5,
) {
    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    fun update(detections: List<Detection>, nowMs: Long): List<Track> {
        val unmatchedTracks = tracks.indices.toMutableSet()
        val unmatchedDets = detections.indices.toMutableSet()

        // Matching only within a class stops a person's box from inheriting
        // the identity of the chair it walked in front of.
        val pairs = mutableListOf<Triple<Float, Int, Int>>()
        tracks.forEachIndexed { ti, track ->
            detections.forEachIndexed { di, det ->
                if (det.classId == track.classId) {
                    val overlap = iou(track.box, det.box)
                    if (overlap >= iouThreshold) pairs += Triple(overlap, ti, di)
                }
            }
        }
        pairs.sortByDescending { it.first }

        for ((_, ti, di) in pairs) {
            if (ti !in unmatchedTracks || di !in unmatchedDets) continue
            val track = tracks[ti]; val det = detections[di]
            track.box = det.box; track.hazard = det.hazard
            track.bearingDeg = det.bearingDeg
            track.lastSeenMs = nowMs; track.hits++; track.misses = 0
            track.record(det.distanceM, nowMs)
            unmatchedTracks -= ti; unmatchedDets -= di
        }

        unmatchedTracks.forEach { tracks[it].misses++ }
        for (di in unmatchedDets.sorted()) {
            val det = detections[di]
            val track = Track(nextId++, det.label, det.classId, det.box, det.hazard,
                det.distanceM, det.bearingDeg, nowMs)
            det.distanceM?.let { track.record(it, nowMs) }
            tracks += track
        }
        tracks.removeAll { it.misses > maxMisses }
        return tracks.filter { it.misses == 0 && it.confirmed(minHits) }
    }

    fun reset() { tracks.clear(); nextId = 1 }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = iw * ih
        if (inter <= 0f) return 0f
        val areaA = max(0f, a[2] - a[0]) * max(0f, a[3] - a[1])
        val areaB = max(0f, b[2] - b[0]) * max(0f, b[3] - b[1])
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
    }
}
