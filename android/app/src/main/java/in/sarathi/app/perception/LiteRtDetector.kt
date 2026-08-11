package `in`.sarathi.app.perception

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import `in`.sarathi.app.guidance.Detection
import `in`.sarathi.app.models.Hazard
import `in`.sarathi.app.models.InputSpec
import `in`.sarathi.app.models.ModelManifest
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Detection via LiteRT, driven entirely by the shared manifest.
 *
 * Nothing here is model-specific. Input size, colour order, normalisation,
 * padding, decoder and thresholds all come from the same YAML the prototype
 * reads, so a model benchmarked on a laptop behaves identically here rather
 * than approximately.
 */
class LiteRtDetector private constructor(
    private val interpreter: Interpreter,
    private val manifest: ModelManifest,
    private val labels: List<String>,
    private val hazards: Map<String, Hazard>,
    private val bridge: Map<String, String>,
    /** Which backend won the first-run benchmark. Reported in diagnostics. */
    val backend: String = "unknown",
    private val delegate: org.tensorflow.lite.Delegate? = null,
) {
    private val spec: InputSpec = manifest.inputFor("android")
        ?: error("${manifest.id}: no input spec")
    private val out = manifest.output ?: error("${manifest.id}: no output spec")
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * 3 * spec.width * spec.height).order(ByteOrder.nativeOrder())

    var lastInferenceMs: Long = 0; private set
    /**
     * Highest class score in the last raw head, before thresholding.
     *
     * Separates "the model ran and the scene genuinely has nothing in it" from
     * "preprocessing or decoding is wrong so nothing can ever score". Those
     * look identical from a detection count of zero, and on a device with no
     * screen there is no other way to tell them apart.
     */
    var lastMaxScore: Float = 0f; private set
    private var loggedShape = false

    fun detect(bitmap: Bitmap): List<Detection> {
        val transform = writeInput(bitmap)

        val shape = interpreter.getOutputTensor(0).shape()
        val elements = shape.fold(1) { acc, n -> acc * n }
        val output = ByteBuffer.allocateDirect(4 * elements).order(ByteOrder.nativeOrder())

        val started = System.nanoTime()
        interpreter.run(inputBuffer, output)
        lastInferenceMs = (System.nanoTime() - started) / 1_000_000

        output.rewind()
        val raw = FloatArray(elements)
        output.asFloatBuffer().get(raw)
        val boxes = decode(raw, shape)
        if (!loggedShape) {
            loggedShape = true
            Log.i(TAG, "graph in=${interpreter.getInputTensor(0).shape().toList()} " +
                "out=${shape.toList()} spec=${spec.width}x${spec.height} " +
                "${spec.layout}/${spec.color} scale=${spec.scale}")
        }
        return nms(boxes, transform, bitmap.width, bitmap.height)
    }

    private fun writeInput(bitmap: Bitmap): Transform = prepareInput(bitmap, spec, inputBuffer)

    internal data class Transform(val scale: Float, val padX: Float, val padY: Float)

    private data class Candidate(val box: FloatArray, val score: Float, val cls: Int)


    private fun decode(raw: FloatArray, shape: IntArray): List<Candidate> {
        // Ultralytics head: [1, 4+nc, anchors], centre-form xywh already in
        // input pixels, class scores post-sigmoid, no separate objectness -
        // treating row 4 as objectness is a common porting bug.
        val channels: Int
        val anchors: Int
        val channelsFirst: Boolean
        if (shape.size == 3) {
            val a = shape[1]; val b = shape[2]
            channelsFirst = a < b
            channels = if (channelsFirst) a else b
            anchors = if (channelsFirst) b else a
        } else return emptyList()

        fun at(channel: Int, anchor: Int): Float =
            if (channelsFirst) raw[channel * anchors + anchor] else raw[anchor * channels + channel]

        val result = ArrayList<Candidate>(64)
        val numClasses = channels - 4
        var seenMax = 0f
        for (i in 0 until anchors) {
            var best = 0; var bestScore = 0f
            for (c in 0 until numClasses) {
                val s = at(4 + c, i)
                if (s > bestScore) { bestScore = s; best = c }
            }
            if (bestScore > seenMax) seenMax = bestScore
            if (bestScore < out.confThreshold) continue
            val cx = at(0, i); val cy = at(1, i)
            val w = at(2, i); val h = at(3, i)
            result += Candidate(
                floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), bestScore, best
            )
        }
        lastMaxScore = seenMax
        return result
    }

    private fun nms(
        candidates: List<Candidate>, t: Transform, frameW: Int, frameH: Int,
    ): List<Detection> {
        // Per class: class-agnostic NMS would suppress a person standing in a
        // doorway, and both matter to someone navigating by ear.
        val kept = ArrayList<Candidate>()
        candidates.groupBy { it.cls }.forEach { (_, group) ->
            val sorted = group.sortedByDescending { it.score }.toMutableList()
            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                kept += best
                sorted.removeAll { iou(best.box, it.box) > out.nmsIou }
            }
        }
        return kept.sortedByDescending { it.score }.take(out.maxDetections).map { c ->
            val b = floatArrayOf(
                ((c.box[0] - t.padX) / t.scale).coerceIn(0f, frameW.toFloat()),
                ((c.box[1] - t.padY) / t.scale).coerceIn(0f, frameH.toFloat()),
                ((c.box[2] - t.padX) / t.scale).coerceIn(0f, frameW.toFloat()),
                ((c.box[3] - t.padY) / t.scale).coerceIn(0f, frameH.toFloat()),
            )
            val rawLabel = labels.getOrElse(c.cls) { c.cls.toString() }
            val label = bridge[rawLabel] ?: rawLabel
            Detection(label, c.score, c.cls, b, hazard = hazards[label] ?: Hazard.LOW)
        }
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = iw * ih
        if (inter <= 0f) return 0f
        val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (union > 0f) inter / union else 0f
    }

    /**
     * Run the model on a known image bundled with the app.
     *
     * A detection count of zero is ambiguous on a device with no preview: the
     * scene may genuinely be empty, or preprocessing may be broken, or the
     * quantized graph may have collapsed. Scoring a frame whose answer is
     * known settles it without anyone having to point the phone at anything.
     */
    fun selfTest(context: Context): String {
        val bitmap = runCatching {
            context.assets.open("models/selftest.jpg").use {
                android.graphics.BitmapFactory.decodeStream(it)
            }
        }.getOrNull() ?: return "self-test: no bundled image"
        val found = detect(bitmap)
        val summary = found.take(4).joinToString(", ") {
            "${it.label} ${"%.2f".format(it.score)}"
        }
        bitmap.recycle()
        return "self-test: maxScore=${"%.3f".format(lastMaxScore)} " +
            "detections=${found.size} [${summary}] in ${lastInferenceMs}ms"
    }

    fun close() {
        interpreter.close()
        (delegate as? org.tensorflow.lite.gpu.GpuDelegate)?.close()
    }

    companion object {
        private const val TAG = "SarathiDetector"

        /** Letterbox, colour-convert and normalise per the manifest. */
        internal fun prepareInput(bitmap: Bitmap, spec: InputSpec, inputBuffer: ByteBuffer): Transform {
            val ratio = min(spec.width / bitmap.width.toFloat(), spec.height / bitmap.height.toFloat())
            val newW = (bitmap.width * ratio).toInt()
            val newH = (bitmap.height * ratio).toInt()
            val padX = if (spec.padMode == "center") (spec.width - newW) / 2 else 0
            val padY = if (spec.padMode == "center") (spec.height - newH) / 2 else 0

            val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
            val pixels = IntArray(newW * newH)
            scaled.getPixels(pixels, 0, newW, 0, 0, newW, newH)

            inputBuffer.rewind()
            val pad = spec.padValue.toFloat()
            // NCHW: all of one channel, then the next. NHWC interleaves.
            val planar = spec.layout == "NCHW"
            val channels = 3

            fun value(px: Int, channel: Int): Float {
                val r = (px shr 16) and 0xFF
                val g = (px shr 8) and 0xFF
                val b = px and 0xFF
                val raw = if (spec.color == "RGB") {
                    when (channel) { 0 -> r; 1 -> g; else -> b }
                } else {
                    when (channel) { 0 -> b; 1 -> g; else -> r }
                }.toFloat()
                return (raw * spec.scale - spec.mean[channel]) / spec.std[channel]
            }

            if (planar) {
                for (c in 0 until channels) {
                    for (y in 0 until spec.height) {
                        val sy = y - padY
                        for (x in 0 until spec.width) {
                            val sx = x - padX
                            val v = if (sy in 0 until newH && sx in 0 until newW)
                                value(pixels[sy * newW + sx], c)
                            else (pad * spec.scale - spec.mean[c]) / spec.std[c]
                            inputBuffer.putFloat(v)
                        }
                    }
                }
            } else {
                for (y in 0 until spec.height) {
                    val sy = y - padY
                    for (x in 0 until spec.width) {
                        val sx = x - padX
                        val inside = sy in 0 until newH && sx in 0 until newW
                        for (c in 0 until channels) {
                            val v = if (inside) value(pixels[sy * newW + sx], c)
                            else (pad * spec.scale - spec.mean[c]) / spec.std[c]
                            inputBuffer.putFloat(v)
                        }
                    }
                }
            }
            inputBuffer.rewind()
            if (scaled != bitmap) scaled.recycle()
            return Transform(ratio, padX.toFloat(), padY.toFloat())
        }

        /**
         * Loads weights according to the manifest's own distribution policy.
         *
         * `bundled` models are memory-mapped straight out of the APK - the
         * packager is told not to compress .tflite, so the interpreter maps
         * the asset in place with no copy and no decompress-to-disk on first
         * run. `user_download` models live in filesDir, because they are
         * fetched after install: the large ones would make the app
         * uninstallable for users on 4 GB phones and metered connections if
         * they shipped inside it.
         *
         * @return null when the weights are absent. The app then runs with the
         * camera, scheduler and speech alive but no detections, which is a far
         * better failure than refusing to start - and makes it obvious that a
         * model pack has not been installed.
         */
        fun create(
            context: Context,
            manifest: ModelManifest,
            labels: List<String>,
            hazards: Map<String, Hazard>,
            bridge: Map<String, String>,
        ): LiteRtDetector? {
            if (!manifest.loadable) {
                Log.w(TAG, "${manifest.id} is excluded by licence policy; refusing to load")
                return null
            }
            val fileName = manifest.fileFor("android") ?: run {
                Log.w(TAG, "${manifest.id} has no android weights declared")
                return null
            }

            val buffer = if (manifest.distribution == "bundled") {
                mapFromAssets(context, "models/$fileName")
                    ?: mapFromFiles(context, fileName)   // sideloaded during development
            } else {
                mapFromFiles(context, fileName)
            }
            if (buffer == null) {
                Log.w(TAG, "weights not found for ${manifest.id}: $fileName " +
                    "(distribution=${manifest.distribution})")
                return null
            }

            // Benchmarked on first run rather than assumed, and cached after.
            // Which backend wins is genuinely device-specific, and a delegate
            // that runs but disagrees with the CPU is rejected rather than
            // selected - silently wrong output is indistinguishable from an
            // empty scene everywhere above this class.
            val choice = Delegates.choose(context, buffer, sampleInput(context, manifest))
            val (options, delegate) = Delegates.optionsFor(choice.name)
                ?: (Interpreter.Options().apply { numThreads = 4 } to null)
            Log.i(TAG, "backend ${choice.name} (${"%.1f".format(choice.medianMs)} ms" +
                (if (choice.note.isNotEmpty()) ", ${choice.note}" else "") + ")")
            return LiteRtDetector(
                Interpreter(buffer, options), manifest, labels, hazards, bridge,
                backend = choice.name, delegate = delegate,
            )
        }

        /**
         * The self-test image, preprocessed exactly as a live frame would be.
         *
         * The delegate benchmark compares each candidate's output against the
         * CPU's, and that comparison is only meaningful on input that makes the
         * model produce something. On a blank tensor every class score sits near
         * zero and any two backends agree trivially - including one that has
         * collapsed, which is precisely the failure the check exists to catch.
         */
        private fun sampleInput(context: Context, manifest: ModelManifest): ByteBuffer? {
            val spec = manifest.inputFor("android") ?: return null
            val bitmap = runCatching {
                context.assets.open("models/selftest.jpg").use {
                    android.graphics.BitmapFactory.decodeStream(it)
                }
            }.getOrNull() ?: return null
            val buffer = ByteBuffer
                .allocateDirect(4 * 3 * spec.width * spec.height)
                .order(ByteOrder.nativeOrder())
            prepareInput(bitmap, spec, buffer)
            bitmap.recycle()
            return buffer
        }

        /**
         * Benchmark every model variant present against every backend.
         *
         * Triggered by a marker file rather than a UI control: it is a
         * developer and field-diagnosis tool, and the app's whole interface is
         * meant to be usable without looking at it.
         *
         *   adb push yolo11n-320-fp32.tflite /data/local/tmp/
         *   adb shell run-as in.sarathi.app cp /data/local/tmp/... files/models/
         *   adb shell run-as in.sarathi.app touch files/survey
         */
        fun runSurvey(context: Context, manifest: ModelManifest) {
            val models = ArrayList<Pair<String, java.nio.MappedByteBuffer>>()
            // Every detector this build ships, not just the selected one. The
            // point of the survey is to compare them, and enumerating the
            // manifests means a model added later is benchmarked without
            // touching this code.
            runCatching {
                `in`.sarathi.app.models.SharedData.listManifests(context)
                    .filter { it.task == "detection" && it.loadable }
                    .sortedBy { it.id }
                    .forEach { m ->
                        val file = m.fileFor("android") ?: return@forEach
                        mapFromAssets(context, "models/$file")?.let { models += m.id to it }
                    }
            }
            java.io.File(context.filesDir, "models")
                .listFiles { f -> f.name.endsWith(".tflite") }
                ?.sortedBy { it.name }
                ?.forEach { f -> mapFromFiles(context, f.name)?.let { models += f.name to it } }
            if (models.isEmpty()) return
            Delegates.logSurvey(Delegates.survey(models, sampleInput(context, manifest)))
        }

        private fun mapFromAssets(context: Context, path: String): java.nio.MappedByteBuffer? =
            runCatching {
                context.assets.openFd(path).use { fd ->
                    java.io.FileInputStream(fd.fileDescriptor).channel.map(
                        FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength,
                    )
                }
            }.getOrNull()

        private fun mapFromFiles(context: Context, fileName: String): java.nio.MappedByteBuffer? {
            val file = java.io.File(context.filesDir, "models/$fileName")
            if (!file.exists()) return null
            return runCatching {
                java.io.FileInputStream(file).use { input ->
                    input.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
                }
            }.getOrNull()
        }
    }
}
