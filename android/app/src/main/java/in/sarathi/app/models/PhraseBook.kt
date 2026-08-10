package `in`.sarathi.app.models

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Spoken phrasing, read from the same `phrases/<lang>.yaml` the prototype uses.
 *
 * Wording is data, not code, so fixing a word that grates on the hundredth
 * hearing is a file edit rather than an app release - and so English and Hindi
 * differ only in their tables. Hindi is not a translation of the English one:
 * it defaults to relative directions rather than clock bearings, because
 * "एक बजे की दिशा में" is long and only natural to someone already trained in
 * the convention.
 */
class PhraseBook(
    val lang: String,
    private val templates: Map<String, String>,
    private val bearing: Map<String, Any?>,
    private val distance: Map<String, Any?>,
    private val objects: Map<String, String>,
    private val system: Map<String, String>,
    val bearingStyle: String,
    private val hedgeAbove: Double,
) {
    fun objectName(label: String): String = objects[label] ?: label.replace('_', ' ')

    fun systemPhrase(key: String): String = system[key] ?: key.replace('_', ' ')

    private fun aheadWord(): String = bearing["ahead"] as? String ?: "ahead"

    fun bearingPhrase(degrees: Double?, aheadBand: Double = 12.0): String {
        if (degrees == null || abs(degrees) <= aheadBand) return aheadWord()
        if (bearingStyle == "relative") {
            @Suppress("UNCHECKED_CAST")
            val table = bearing["relative"] as? Map<String, String> ?: return aheadWord()
            val key = if (degrees < 0) {
                if (degrees > -35) "slight_left" else "left"
            } else {
                if (degrees < 35) "slight_right" else "right"
            }
            return table[key] ?: aheadWord()
        }
        @Suppress("UNCHECKED_CAST")
        val table = bearing["clock"] as? Map<Any, String> ?: return aheadWord()
        var hour = 12 + (degrees / 30.0).roundToInt()
        if (hour > 12) hour -= 12
        if (hour < 1) hour += 12
        return table[hour] ?: table[hour.toString()] ?: aheadWord()
    }

    fun distancePhrase(metres: Double?, uncertainty: Double = 0.0): String {
        if (metres == null) return ""
        @Suppress("UNCHECKED_CAST")
        val steps = distance["steps"] as? List<Map<String, Any?>> ?: return ""
        var text = steps.lastOrNull()?.get("text") as? String ?: ""
        for (step in steps) {
            val max = (step["max"] as? Number)?.toDouble() ?: continue
            if (metres < max) {
                text = step["text"] as? String ?: text
                break
            }
        }
        if (uncertainty > hedgeAbove) {
            val hedge = distance["hedge"] as? String
            if (!hedge.isNullOrBlank()) return "$hedge $text"
        }
        return text
    }

    /**
     * Urgent phrasing drops the distance entirely. At that range the user needs
     * to stop, not to learn whether it is 1.5 m or 2 m, and every extra
     * syllable is delay before they can act.
     */
    fun utterance(label: String, bearingDeg: Double?, metres: Double?, urgent: Boolean,
                  uncertainty: Double = 0.0): String {
        val name = objectName(label)
        val where = bearingPhrase(bearingDeg)
        val isAhead = where == aheadWord()
        val text = if (urgent) {
            (templates["urgent"] ?: "{object} {bearing}")
                .replace("{object}", name).replace("{bearing}", where)
        } else {
            val dist = distancePhrase(metres, uncertainty)
            val key = when {
                isAhead && dist.isNotEmpty() -> "ahead_full"
                isAhead -> "ahead_no_distance"
                dist.isNotEmpty() -> "full"
                else -> "no_distance"
            }
            (templates[key] ?: "{object}, {bearing}, {distance}")
                .replace("{object}", name)
                .replace("{bearing}", where)
                .replace("{distance}", dist)
        }
        return tidy(text)
    }

    private fun tidy(text: String): String =
        text.split(Regex("\\s+")).filter { it.isNotBlank() }.joinToString(" ")
            .replace(" ,", ",").replace(",,", ",").trim().trim(',').trim()

    companion object {
        @Suppress("UNCHECKED_CAST")
        fun from(lang: String, root: Map<String, Any?>) = PhraseBook(
            lang = root["lang"] as? String ?: lang,
            templates = (root["templates"] as? Map<String, String>) ?: emptyMap(),
            bearing = (root["bearing"] as? Map<String, Any?>) ?: emptyMap(),
            distance = (root["distance"] as? Map<String, Any?>) ?: emptyMap(),
            objects = (root["objects"] as? Map<String, String>) ?: emptyMap(),
            system = (root["system"] as? Map<String, String>) ?: emptyMap(),
            bearingStyle = root["bearing_style"] as? String ?: "clock",
            hedgeAbove = (root["hedge_above_uncertainty"] as? Number)?.toDouble() ?: 0.25,
        )
    }
}
