package `in`.sarathi.app.guidance

import `in`.sarathi.app.models.Hazard
import `in`.sarathi.app.perception.Geometry
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.pow

/**
 * Deciding what is worth saying.
 *
 * The hardest part of the product and the one most likely to decide whether
 * anyone keeps using it. A detector at 5 Hz on a street produces hundreds of
 * true detections a minute; speaking them is not assistance, it is noise, and
 * the documented reason people abandon assistive vision tools is that they talk
 * too much. Restraint is the feature.
 */
class SaliencyEngine(private val config: Config = Config()) {

    data class Config(
        val maxDistanceM: Double = 6.0,
        val corridorHalfWidthM: Double = 0.7,
        val proximityHalfM: Double = 2.5,
        val hazardBonus: Double = 0.45,
        val closingBonus: Double = 0.35,
        val scoreFloor: Double = 0.55,
        /** Low-hazard classes are context. They answer "what is on the table?"
         *  when asked and are never announced unprompted - otherwise the
         *  walking loop narrates cutlery. */
        val announceLowHazard: Boolean = false,
        val repeatCooldownMs: Long = 8_000,
        val classCooldownMs: Long = 3_000,
        val minUtteranceGapMs: Long = 1_500,
        val urgentDistanceM: Double = 2.5,
        val urgentTtcS: Double = 2.0,
        val urgentCooldownMs: Long = 2_000,
        val unknownDistanceProximity: Double = 0.35,
    )

    data class Ranked(
        val track: Track,
        val score: Double,
        val urgent: Boolean,
        val inPath: Boolean,
    )

    private val lastSpokenTrack = mutableMapOf<Int, Long>()
    private val lastSpokenClass = mutableMapOf<String, Long>()
    private val lastDistanceSpoken = mutableMapOf<Int, Double>()
    private var lastUtteranceAt = Long.MIN_VALUE / 4

    /**
     * The strongest candidate considered on the last pass, and why it lost.
     *
     * Silence is this system's most common output and its most ambiguous one.
     * "Nothing here is worth saying" and "the pipeline is broken" produce
     * exactly the same thing - nothing - and after a demo where the app was in
     * fact broken, being unable to tell those apart is what made it
     * undemonstrable rather than merely quiet.
     *
     * So the engine now records its reasoning. Nothing depends on this; it is
     * read by the screen and by logs, and it costs one assignment per pass.
     */
    var lastReason: String = "no detections"
        private set

    fun score(track: Track): Ranked {
        val distance = track.distanceM
        val proximity: Double
        val path: Double
        var lateral: Double? = null

        if (distance == null) {
            proximity = config.unknownDistanceProximity
            path = (1.0 - abs(track.bearingDeg ?: 0.0) / 45.0).coerceIn(0.0, 1.0) * 0.8
        } else {
            proximity = 1.0 / (1.0 + (distance / config.proximityHalfM).pow(2))
            lateral = Geometry.lateralOffsetM(distance, track.bearingDeg ?: 0.0)
            path = exp(-((lateral / config.corridorHalfWidthM).pow(2)))
        }

        // Hazard levels are not linear in consequence: you bump into a sofa,
        // you fall down a staircase.
        val hazard = (track.hazard.level / Hazard.CRITICAL.level.toDouble()).pow(1.5)
        val closing = ((track.closingSpeedMps ?: 0.0) / 2.0).coerceIn(0.0, 1.0)

        // Proximity and path MULTIPLY: near AND in the way, not moderately
        // both. Averaging them put a chair squarely in the walking line below
        // the announcement floor while a cup on a table nearly cleared it.
        var score = (proximity * path + config.hazardBonus * hazard +
            config.closingBonus * closing).coerceIn(0.0, 1.0)

        val beyondRange = distance != null && distance > config.maxDistanceM
        val tooTrivial = track.hazard == Hazard.LOW && !config.announceLowHazard
        if (beyondRange || tooTrivial) score = 0.0

        val inPath = lateral != null && abs(lateral) <= config.corridorHalfWidthM
        return Ranked(track, score, urgency(track, inPath, distance), inPath)
    }

    private fun urgency(track: Track, inPath: Boolean, distance: Double?): Boolean {
        val ttc = track.timeToContactS()
        if (ttc != null && ttc <= config.urgentTtcS && inPath) return true
        return track.hazard == Hazard.CRITICAL && inPath &&
            distance != null && distance <= config.urgentDistanceM
    }

    /** The one thing worth saying right now, or nothing. */
    fun select(tracks: List<Track>, nowMs: Long): Ranked? {
        val ranked = tracks.map { score(it) }.sortedByDescending { it.score }
        if (ranked.isEmpty()) {
            lastReason = "no detections"
            return null
        }
        val best = ranked.first()
        lastReason = when {
            best.track.hazard == Hazard.LOW && !config.announceLowHazard ->
                "${best.track.label}: low hazard, context only"
            best.track.distanceM != null && best.track.distanceM!! > config.maxDistanceM ->
                // Feet, like every other distance on the screen. Two units for
                // one quantity is a way to be wrong twice.
                "${best.track.label}: %.0f ft, beyond %.0f ft".format(
                    best.track.distanceM!! * 3.28084, config.maxDistanceM * 3.28084
                )
            best.score < config.scoreFloor ->
                "${best.track.label}: %.2f below %.2f".format(best.score, config.scoreFloor)
            else -> "${best.track.label}: %.2f".format(best.score)
        }
        for (candidate in ranked) {
            if (candidate.score < config.scoreFloor && !candidate.urgent) break
            if (!passesCooldown(candidate, nowMs)) {
                lastReason = "${candidate.track.label}: recently said"
                continue
            }
            if (!candidate.urgent && nowMs - lastUtteranceAt < config.minUtteranceGapMs) {
                lastReason = "${candidate.track.label}: too soon after the last"
                continue
            }
            // Chosen, so there is nothing to explain. Left set, the screen
            // reads "quiet - chair: 0.84" while the app is saying "chair
            // ahead" - a diagnostic added to disambiguate silence, reporting
            // silence that did not happen.
            lastReason = ""
            markSpoken(candidate, nowMs)
            return candidate
        }
        return null
    }

    private fun passesCooldown(candidate: Ranked, nowMs: Long): Boolean {
        val escalated = escalated(candidate)
        lastSpokenTrack[candidate.track.id]?.let { last ->
            val window = if (candidate.urgent) config.urgentCooldownMs else config.repeatCooldownMs
            if (nowMs - last < window && !escalated) return false
        }
        lastSpokenClass[candidate.track.label]?.let { last ->
            if (nowMs - last < config.classCooldownMs && !candidate.urgent && !escalated) return false
        }
        return true
    }

    /**
     * Has this become materially more dangerous since we spoke?
     *
     * Without it the cooldown is actively harmful: a car announced at six
     * metres would stay silent while it closed to one.
     */
    private fun escalated(candidate: Ranked): Boolean {
        val previous = lastDistanceSpoken[candidate.track.id] ?: return false
        val now = candidate.track.distanceM ?: return false
        if (now <= previous * 0.5) return true
        val ttc = candidate.track.timeToContactS()
        return ttc != null && ttc <= config.urgentTtcS
    }

    private fun markSpoken(candidate: Ranked, nowMs: Long) {
        lastSpokenTrack[candidate.track.id] = nowMs
        lastSpokenClass[candidate.track.label] = nowMs
        candidate.track.distanceM?.let { lastDistanceSpoken[candidate.track.id] = it }
        lastUtteranceAt = nowMs
    }

    fun reset() {
        lastSpokenTrack.clear(); lastSpokenClass.clear(); lastDistanceSpoken.clear()
        lastUtteranceAt = Long.MIN_VALUE / 4
    }
}
