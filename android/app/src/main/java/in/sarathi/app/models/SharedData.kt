package `in`.sarathi.app.models

import android.content.Context
import org.yaml.snakeyaml.Yaml
import java.io.InputStreamReader

/**
 * Readers for the files this app shares with the Python prototype.
 *
 * Model manifests, phrase tables, the taxonomy and the size priors are copied
 * into assets at build time from one place in the repository. Nothing here is
 * a Kotlin transcription of a Python structure: both sides parse the same
 * bytes.
 *
 * That is the point. A model benchmarked on a laptop only stays meaningful on
 * a phone if both agree on input size, colour order, normalisation, decoder and
 * label order - and the reliable way to make two implementations agree is to
 * give them one source rather than two copies and a code review.
 */
object SharedData {

    private val yaml = Yaml()

    private fun <T> read(context: Context, path: String, block: (Map<String, Any?>) -> T): T {
        context.assets.open(path).use { stream ->
            @Suppress("UNCHECKED_CAST")
            val root = yaml.load<Map<String, Any?>>(InputStreamReader(stream, Charsets.UTF_8))
                ?: emptyMap()
            return block(root)
        }
    }

    fun manifest(context: Context, fileName: String): ModelManifest =
        read(context, "manifests/$fileName") { ModelManifest.from(it) }

    fun listManifests(context: Context): List<ModelManifest> =
        (context.assets.list("manifests") ?: emptyArray())
            .filter { it.endsWith(".yaml") || it.endsWith(".yml") }
            .mapNotNull { runCatching { manifest(context, it) }.getOrNull() }

    fun phraseBook(context: Context, lang: String): PhraseBook =
        read(context, "phrases/$lang.yaml") { PhraseBook.from(lang, it) }

    fun labels(context: Context, name: String): List<String> =
        context.assets.open("labels/$name").bufferedReader().useLines { lines ->
            lines.map { it.trim() }
                .filter { it.isNotEmpty() && !it.startsWith("#") }
                .toList()
        }

    /** Hazard level and spoken names per class, from the shared taxonomy. */
    fun taxonomy(context: Context): Map<String, Hazard> =
        read(context, "taxonomy/sarathi77.yaml") { root ->
            @Suppress("UNCHECKED_CAST")
            val classes = root["classes"] as? List<Map<String, Any?>> ?: emptyList()
            classes.associate { entry ->
                (entry["name"] as String) to Hazard.parse(entry["hazard"] as? String)
            }
        }

    /** Real-world height priors, used when the ground plane cannot be trusted. */
    fun sizePriors(context: Context): Map<String, SizePrior> =
        read(context, "taxonomy/size_priors.yaml") { root ->
            @Suppress("UNCHECKED_CAST")
            val priors = root["priors"] as? Map<String, Map<String, Any?>> ?: emptyMap()
            priors.mapValues { (_, spec) ->
                SizePrior(
                    heightM = (spec["h_m"] as? Number)?.toDouble(),
                    spread = (spec["spread"] as? Number)?.toDouble() ?: 0.30,
                    grounded = spec["grounded"] as? Boolean ?: true,
                )
            }
        }

    /** Detector label -> taxonomy class, for running a stock COCO checkpoint. */
    fun labelBridge(context: Context): Map<String, String> =
        read(context, "taxonomy/coco_to_sarathi.yaml") { root ->
            @Suppress("UNCHECKED_CAST")
            (root["map"] as? Map<String, String>) ?: emptyMap()
        }
}

data class SizePrior(
    val heightM: Double?,
    val spread: Double = 0.30,
    /** False for things not resting on the floor: signs, traffic lights, taps. */
    val grounded: Boolean = true,
)

enum class Hazard(val level: Int) {
    LOW(0), MEDIUM(1), HIGH(2), CRITICAL(3);

    companion object {
        fun parse(value: String?): Hazard = when (value?.lowercase()) {
            "critical" -> CRITICAL
            "high" -> HIGH
            "medium" -> MEDIUM
            else -> LOW
        }
    }
}
