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
) {
    private val spec: InputSpec = manifest.inputFor("android")
        ?: error("${manifest.id}: no input spec")
    private val out = manifest.output ?: error("${manifest.id}: no output spec")
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * 3 * spec.width * spec.height).order(ByteOrder.nativeOrder())

    var lastInferenceMs: Long = 0; private set

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
        return nms(boxes, transform, bitmap.width, bitmap.height)
    }

    /** Letterbox, colour-convert and normalise per the manifest. */
    private fun writeInput(bitmap: Bitmap): Transform {
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

    private data class Transform(val scale: Float, val padX: Float, val padY: Float)

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
        for (i in 0 until anchors) {
            var best = 0; var bestScore = 0f
            for (c in 0 until numClasses) {
                val s = at(4 + c, i)
                if (s > bestScore) { bestScore = s; best = c }
            }
            if (bestScore < out.confThreshold) continue
            val cx = at(0, i); val cy = at(1, i)
            val w = at(2, i); val h = at(3, i)
            result += Candidate(
                floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), bestScore, best
            )
        }
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

    fun close() = interpreter.close()

    companion object {
        private const val TAG = "SarathiDetector"

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

            val options = Interpreter.Options().apply {
                numThreads = 4
                // Delegate selection is benchmarked on first run rather than
                // assumed. NNAPI is deprecated as of Android 15, and which
                // accelerator actually wins on a Tensor G3 is a measurement,
                // not a guess.
            }
            return LiteRtDetector(Interpreter(buffer, options), manifest, labels, hazards, bridge)
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
