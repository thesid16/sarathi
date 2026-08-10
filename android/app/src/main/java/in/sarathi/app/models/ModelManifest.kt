package `in`.sarathi.app.models

/**
 * The same manifest schema the Python prototype reads.
 *
 * Only the fields the phone needs are parsed. Anything unrecognised is ignored
 * rather than rejected, so the prototype can add fields for its own use without
 * breaking the app - the two evolve at different speeds and coupling them
 * tightly would defeat the point.
 */
data class ModelManifest(
    val id: String,
    val task: String,
    val license: String,
    val distribution: String,
    val files: Map<String, String>,
    val runtime: Map<String, String>,
    val input: InputSpec?,
    val output: OutputSpec?,
    val delegates: List<String>,
    val vendoredWeights: Boolean,
) {
    /** False for models whose licence bars them from a release build. */
    val loadable: Boolean get() = distribution != "excluded"

    /** The weights file for this platform, resolved through `runtime`. */
    fun fileFor(platform: String = "android"): String? {
        val engine = runtime[platform] ?: platform
        val format = ENGINE_FORMATS[engine] ?: engine
        return files[format] ?: files[engine]
    }

    companion object {
        private val ENGINE_FORMATS = mapOf(
            "litert" to "tflite",
            "tflite" to "tflite",
            "onnxruntime" to "onnx",
            "onnx" to "onnx",
        )

        @Suppress("UNCHECKED_CAST")
        fun from(root: Map<String, Any?>): ModelManifest {
            val files = (root["files"] as? Map<String, Any?> ?: emptyMap()).mapValues { (_, v) ->
                when (v) {
                    is String -> v
                    is Map<*, *> -> v["path"] as? String ?: ""
                    else -> ""
                }
            }
            return ModelManifest(
                id = root["id"] as? String ?: error("manifest is missing `id`"),
                task = root["task"] as? String ?: error("manifest is missing `task`"),
                license = root["license"] as? String ?: "unknown",
                distribution = root["distribution"] as? String ?: "bundled",
                files = files,
                runtime = (root["runtime"] as? Map<String, String>) ?: emptyMap(),
                input = (root["input"] as? Map<String, Any?>)?.let { InputSpec.from(it) },
                output = (root["output"] as? Map<String, Any?>)?.let { OutputSpec.from(it) },
                delegates = (root["delegates"] as? List<String>) ?: emptyList(),
                vendoredWeights = root["vendored_weights"] as? Boolean ?: false,
            )
        }
    }
}

/**
 * How a camera frame becomes this model's input tensor.
 *
 * Every one of these is a way to get plausible-looking detections in the wrong
 * place, or none at all. YOLOX wants BGR with no 0-1 scaling and corner
 * padding; Ultralytics wants RGB scaled to 0-1 with centred padding. Feeding
 * either the other's preprocessing produces a model that runs perfectly and
 * detects almost nothing.
 */
data class InputSpec(
    val width: Int,
    val height: Int,
    val layout: String,
    val color: String,
    val dtype: String,
    val resize: String,
    val padMode: String,
    val padValue: Int,
    val scale: Float,
    val mean: FloatArray,
    val std: FloatArray,
) {
    companion object {
        private fun triple(value: Any?, default: Float): FloatArray = when (value) {
            is Number -> floatArrayOf(value.toFloat(), value.toFloat(), value.toFloat())
            is List<*> -> FloatArray(3) { i -> (value.getOrNull(i) as? Number)?.toFloat() ?: default }
            else -> floatArrayOf(default, default, default)
        }

        fun from(spec: Map<String, Any?>) = InputSpec(
            width = (spec["width"] as Number).toInt(),
            height = (spec["height"] as Number).toInt(),
            layout = (spec["layout"] as? String ?: "NCHW").uppercase(),
            color = (spec["color"] as? String ?: "RGB").uppercase(),
            dtype = (spec["dtype"] as? String ?: "float32").lowercase(),
            resize = (spec["resize"] as? String ?: "letterbox").lowercase(),
            padMode = (spec["pad_mode"] as? String ?: "center").lowercase(),
            padValue = (spec["pad_value"] as? Number)?.toInt() ?: 114,
            scale = (spec["scale"] as? Number)?.toFloat() ?: 1.0f,
            mean = triple(spec["mean"], 0f),
            std = triple(spec["std"], 1f),
        )
    }

    // Explicit equals/hashCode: FloatArray uses identity, and a data class
    // with an array field silently compares by reference.
    override fun equals(other: Any?): Boolean =
        other is InputSpec && width == other.width && height == other.height &&
            layout == other.layout && color == other.color && dtype == other.dtype &&
            resize == other.resize && padMode == other.padMode &&
            mean.contentEquals(other.mean) && std.contentEquals(other.std)

    override fun hashCode(): Int =
        (((width * 31 + height) * 31 + layout.hashCode()) * 31 + color.hashCode()) * 31 +
            mean.contentHashCode()
}

data class OutputSpec(
    val decoder: String,
    val labels: String?,
    val confThreshold: Float,
    val nmsIou: Float,
    val maxDetections: Int,
    val strides: List<Int>,
) {
    companion object {
        @Suppress("UNCHECKED_CAST")
        fun from(spec: Map<String, Any?>) = OutputSpec(
            decoder = spec["decoder"] as? String ?: "yolo11",
            labels = spec["labels"] as? String,
            confThreshold = (spec["conf_threshold"] as? Number)?.toFloat() ?: 0.35f,
            nmsIou = (spec["nms_iou"] as? Number)?.toFloat() ?: 0.5f,
            maxDetections = (spec["max_detections"] as? Number)?.toInt() ?: 50,
            strides = (spec["strides"] as? List<Number>)?.map { it.toInt() } ?: listOf(8, 16, 32),
        )
    }
}
