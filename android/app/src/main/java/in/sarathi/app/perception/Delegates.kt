package `in`.sarathi.app.perception

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import kotlin.math.abs

/**
 * Picking an accelerator by measuring, not by assuming.
 *
 * Which backend is fastest is genuinely not predictable. The GPU wins on some
 * phones and loses on others; a model containing ops the delegate cannot take
 * gets partitioned, and the transfers between CPU and GPU then cost more than
 * the compute saves. NNAPI - the historical route to a Tensor TPU - is
 * deprecated as of Android 15, and whether anything still reaches the TPU on a
 * Tensor G3 is a question no datasheet answers.
 *
 * So the app benchmarks the candidates once and keeps the winner. A second at
 * first launch buys a decision that is right for the device in the user's
 * hand, rather than right for the device the developer happened to own.
 *
 * **Fast is not sufficient.** A delegate that runs and produces wrong numbers
 * is worse than one that fails outright, because nothing reports it - the app
 * simply stops detecting. Every candidate is therefore scored against the CPU
 * on the same real image, and rejected if it disagrees.
 *
 * What counts as disagreement needs care. The first version of this check used
 * a flat absolute tolerance across the whole output tensor, and rejected the
 * GPU on a Pixel 8a. That was a bug in the check, not a fault in the delegate:
 * a detector head mixes box coordinates in pixels (0-320) with class scores in
 * [0,1], and demanding 0.05 absolute on a coordinate of 300 asks for 0.017%
 * relative accuracy, which fp16 cannot supply by construction. The rule below
 * is the standard mixed form, `|a-b| <= atol + rtol*|b|`, which holds class
 * scores to a tight absolute bound and coordinates to a proportional one. The
 * worst deviation of each kind is logged either way, so the decision can be
 * audited rather than trusted.
 */
object Delegates {

    private const val TAG = "SarathiDelegate"
    private const val PREFS = "sarathi-delegate"
    private const val KEY_NAME = "chosen"
    private const val KEY_MS = "chosen_ms"

    /**
     * Absolute slack, which governs small values - class scores above all.
     * 0.03 is well inside the gap between a confident detection and the 0.35
     * announce threshold, so no delegate passing this can flip a detection on
     * or off that the CPU would not.
     */
    private const val ATOL = 0.03f

    /**
     * Proportional slack, which governs large values - box coordinates. 2% of
     * a 320 px input is ~6 px, which moves a box edge by about the width of
     * the letterbox seam and changes a geometric distance by a few percent.
     * Tolerable; a delegate needing more than this is not doing the same
     * arithmetic.
     */
    private const val RTOL = 0.02f

    data class Choice(val name: String, val medianMs: Double, val note: String = "")

    /** What a candidate's output looks like next to the CPU's. */
    private data class Agreement(
        val worstAbs: Float,
        val worstRel: Float,
        /** Reference value at the worst mixed-rule violation, for context. */
        val atValue: Float,
        val violations: Int,
        /** Where in the output the disagreements sit. */
        val firstBad: Int,
        val lastBad: Int,
        val total: Int,
    ) {
        val ok: Boolean get() = violations == 0
        override fun toString() =
            "worst abs ${"%.4f".format(worstAbs)}, worst rel ${"%.4f".format(worstRel)}" +
                if (violations > 0) {
                    // Which region of the tensor is wrong says far more than
                    // how wrong it is. A detector head is a stack of rows with
                    // different jobs; damage confined to one band names the
                    // guilty subgraph, while damage spread evenly means the
                    // whole delegation is unsound.
                    ", $violations/$total over tolerance (ref ${"%.3f".format(atValue)}, " +
                        "indices $firstBad-$lastBad)"
                } else ""
    }

    /** Build interpreter options for a named backend. Returns null if unavailable. */
    fun optionsFor(name: String): Pair<Interpreter.Options, Delegate?>? = when (name) {
        "xnnpack-4t" -> Interpreter.Options().apply { numThreads = 4 } to null
        // Fewer threads is sometimes faster on a big.LITTLE layout, and always
        // cooler - which matters more here than raw latency, because heat is
        // what forced the rate down to 1 Hz on the fp32 build.
        "xnnpack-2t" -> Interpreter.Options().apply { numThreads = 2 } to null
        // The GPU is tried in several configurations rather than one, because
        // "the GPU is wrong" is not a finding you can act on - it does not say
        // whether the fault is in the delegate's int8 dequantisation path, in
        // its reduced precision, or in the OpenCL backend specifically. Each
        // variant below turns off exactly one of those.
        "gpu" -> gpu(GpuDelegateFactory.Options())
        "gpu-f32" -> gpu(GpuDelegateFactory.Options().setPrecisionLossAllowed(false))
        "gpu-noquant" -> gpu(GpuDelegateFactory.Options().setQuantizedModelsAllowed(false))
        "gpu-gl" -> gpu(
            GpuDelegateFactory.Options()
                .setForceBackend(GpuDelegateFactory.Options.GpuBackend.OPENGL)
        )
        else -> null
    }

    private fun gpu(options: GpuDelegateFactory.Options): Pair<Interpreter.Options, Delegate?>? =
        runCatching {
            val delegate = GpuDelegate(options)
            Interpreter.Options().apply { addDelegate(delegate) } to delegate as Delegate?
        }.getOrNull()

    private val CANDIDATES = listOf("xnnpack-4t", "xnnpack-2t", "gpu")

    /** Everything the survey explores, including diagnostic-only variants. */
    private val ALL_BACKENDS =
        listOf("xnnpack-4t", "xnnpack-2t", "gpu", "gpu-f32", "gpu-noquant", "gpu-gl")

    /**
     * Benchmark every candidate and return the fastest that both runs and
     * agrees with the CPU. Cached, so this cost is paid once.
     *
     * @param sample a prepared input tensor. A real preprocessed image, not
     * zeros: a blank frame drives every class score to near zero, so the
     * agreement check would be comparing noise against noise and would pass
     * anything. The image that exercises the model is the one that should
     * decide whether a delegate reproduces it.
     */
    fun choose(
        context: Context,
        model: MappedByteBuffer,
        sample: ByteBuffer? = null,
        force: Boolean = false,
    ): Choice {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (!force) {
            val cached = prefs.getString(KEY_NAME, null)
            if (cached != null) {
                return Choice(cached, prefs.getFloat(KEY_MS, 0f).toDouble(), "cached")
            }
        }

        val reference = referenceOutput(model, sample)
        if (reference == null) Log.w(TAG, "no CPU reference; agreement cannot be checked")
        var best: Choice? = null

        for (name in CANDIDATES) {
            val measured = benchmark(model, name, sample, reference)
            if (measured == null) {
                Log.i(TAG, "$name: unavailable")
                continue
            }
            val (ms, agreement) = measured
            val line = "$name: ${"%.1f".format(ms)} ms" +
                if (agreement != null) " ($agreement)" else ""
            if (agreement != null && !agreement.ok) {
                // The dangerous case. Silently wrong output looks exactly like
                // an empty scene from every layer above this one.
                Log.w(TAG, "$line - DISAGREES with CPU, rejected")
                continue
            }
            Log.i(TAG, line)
            if (best == null || ms < best.medianMs) best = Choice(name, ms)
        }

        val chosen = best ?: Choice("xnnpack-4t", 0.0, "fallback - nothing benchmarked cleanly")
        prefs.edit()
            .putString(KEY_NAME, chosen.name)
            .putFloat(KEY_MS, chosen.medianMs.toFloat())
            .apply()
        Log.i(TAG, "chose ${chosen.name} at ${"%.1f".format(chosen.medianMs)} ms")
        return chosen
    }

    /** Single-threaded CPU output, treated as ground truth for agreement. */
    private fun referenceOutput(model: MappedByteBuffer, sample: ByteBuffer?): FloatArray? =
        runCatching {
            val interpreter = Interpreter(model, Interpreter.Options().apply { numThreads = 1 })
            try {
                val input = sample ?: zeroInput(interpreter)
                val output = outputBuffer(interpreter)
                input.rewind()
                interpreter.run(input, output)
                output.rewind()
                FloatArray(output.capacity() / 4).also { output.asFloatBuffer().get(it) }
            } finally {
                interpreter.close()
            }
        }.getOrNull()

    private fun benchmark(
        model: MappedByteBuffer, name: String, sample: ByteBuffer?, reference: FloatArray?,
    ): Pair<Double, Agreement?>? = runCatching {
        val (options, delegate) = optionsFor(name) ?: return null
        val interpreter = Interpreter(model, options)
        try {
            val input = sample ?: zeroInput(interpreter)
            val output = outputBuffer(interpreter)

            // Warm up: first inference includes lazy graph setup and, for the
            // GPU, shader compilation. Timing it would measure the wrong thing.
            repeat(3) { input.rewind(); output.rewind(); interpreter.run(input, output) }

            val times = DoubleArray(8)
            for (i in times.indices) {
                input.rewind(); output.rewind()
                val started = System.nanoTime()
                interpreter.run(input, output)
                times[i] = (System.nanoTime() - started) / 1_000_000.0
            }
            times.sort()
            val median = times[times.size / 2]

            output.rewind()
            val got = FloatArray(output.capacity() / 4)
            output.asFloatBuffer().get(got)
            median to compare(reference, got)
        } finally {
            interpreter.close()
            (delegate as? GpuDelegate)?.close()
        }
    }.getOrNull()

    private fun compare(reference: FloatArray?, got: FloatArray): Agreement? {
        if (reference == null || reference.size != got.size) return null
        var worstAbs = 0f
        var worstRel = 0f
        var violations = 0
        var atValue = 0f
        var worstExcess = 0f
        var firstBad = -1
        var lastBad = -1
        for (i in reference.indices) {
            val ref = reference[i]
            val d = abs(ref - got[i])
            if (d > worstAbs) worstAbs = d
            val rel = if (abs(ref) > 1e-6f) d / abs(ref) else 0f
            if (rel > worstRel) worstRel = rel
            val allowed = ATOL + RTOL * abs(ref)
            if (d > allowed) {
                violations++
                if (firstBad < 0) firstBad = i
                lastBad = i
                if (d - allowed > worstExcess) { worstExcess = d - allowed; atValue = ref }
            }
        }
        return Agreement(
            worstAbs, worstRel, atValue, violations, firstBad, lastBad, reference.size,
        )
    }

    // ------------------------------------------------------------------
    // Survey: the whole grid of model variants against the whole grid of
    // backends, measured on the device in the user's hand.
    //
    // This exists because the pairing is not separable. A quantization that is
    // fastest on the CPU can be the one the GPU cannot run, so choosing the
    // model and then choosing the backend picks a combination nobody measured.
    // Running the grid is a few seconds once, and it is the only way the answer
    // comes out right on a phone that nobody developing this owned.
    // ------------------------------------------------------------------

    data class SurveyRow(
        val model: String,
        val backend: String,
        val medianMs: Double,
        val detail: String,
        val ok: Boolean,
    )

    fun survey(models: List<Pair<String, MappedByteBuffer>>, sample: ByteBuffer?): List<SurveyRow> {
        val rows = ArrayList<SurveyRow>()
        val sampleFloats = sample?.let {
            it.rewind()
            FloatArray(it.capacity() / 4).also { out -> it.asFloatBuffer().get(out) }
        }
        for ((name, buffer) in models) {
            val reference = runCatching {
                val interpreter = Interpreter(buffer, Interpreter.Options().apply { numThreads = 1 })
                try { runOnce(interpreter, sampleFloats) } finally { interpreter.close() }
            }.onFailure { Log.w(TAG, "$name reference: ${it.message?.lines()?.first()}") }
                .getOrNull()
            for (backend in ALL_BACKENDS) {
                var why = "failed to run"
                val row = runCatching { measure(buffer, backend, sampleFloats, reference) }
                    .onFailure { why = it.message?.lines()?.first()?.take(160) ?: it.toString() }
                    .getOrNull()
                if (row == null) {
                    rows += SurveyRow(name, backend, Double.NaN, why, false)
                    continue
                }
                val (ms, agreement) = row
                rows += SurveyRow(
                    name, backend, ms,
                    agreement?.toString() ?: "no reference",
                    agreement?.ok ?: true,
                )
            }
        }
        return rows
    }

    fun logSurvey(rows: List<SurveyRow>) {
        Log.i(TAG, "--- model x backend survey ---")
        for (r in rows) {
            val ms = if (r.medianMs.isNaN()) "  -  " else "%5.1f".format(r.medianMs)
            Log.i(TAG, "%-34s %-11s $ms ms  %s %s".format(
                r.model, r.backend, if (r.ok) "OK  " else "WRONG", r.detail,
            ))
        }
        Log.i(TAG, "--- end survey ---")
    }

    private fun measure(
        model: MappedByteBuffer, backend: String, sample: FloatArray?, reference: FloatArray?,
    ): Pair<Double, Agreement?>? {
        val (options, delegate) = optionsFor(backend) ?: return null
        val interpreter = Interpreter(model, options)
        try {
            val input = feed(interpreter, sample)
            val output = drain(interpreter)
            repeat(3) { input.rewind(); output.rewind(); interpreter.run(input, output) }
            val times = DoubleArray(8)
            for (i in times.indices) {
                input.rewind(); output.rewind()
                val t = System.nanoTime()
                interpreter.run(input, output)
                times[i] = (System.nanoTime() - t) / 1_000_000.0
            }
            times.sort()
            val outTensor = interpreter.getOutputTensor(0)
            val got = readFloats(output, outTensor.numBytes() /
                outTensor.shape().fold(1) { a, b -> a * b })
            return times[times.size / 2] to compare(reference, got)
        } finally {
            interpreter.close()
            (delegate as? GpuDelegate)?.close()
        }
    }

    private fun runOnce(interpreter: Interpreter, sample: FloatArray?): FloatArray {
        val input = feed(interpreter, sample)
        val output = drain(interpreter)
        interpreter.run(input, output)
        val tensor = interpreter.getOutputTensor(0)
        return readFloats(output, tensor.numBytes() / tensor.shape().fold(1) { a, b -> a * b })
    }

    /**
     * Write the sample in whatever dtype this graph's input tensor wants.
     *
     * An fp16 export makes the *whole* graph fp16, input tensor included, so a
     * float32 buffer is not merely imprecise there - it is the wrong number of
     * bytes and the call fails outright. Sizing from the tensor rather than
     * from an assumption is what lets one survey compare variants that do not
     * share an interface.
     */
    private fun feed(interpreter: Interpreter, sample: FloatArray?): ByteBuffer {
        val tensor = interpreter.getInputTensor(0)
        val elements = tensor.shape().fold(1) { a, b -> a * b }
        val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        // Bytes per element rather than the DataType enum: this LiteRT's Java
        // enum has no FLOAT16 member at all, even though the runtime happily
        // executes fp16 graphs. The byte width is the fact that matters here
        // and it is always reported correctly.
        for (i in 0 until elements) {
            val v = sample?.getOrNull(i) ?: 0f
            when (tensor.numBytes() / elements) {
                4 -> buffer.putFloat(v)
                2 -> buffer.putShort(android.util.Half.toHalf(v))
                else -> buffer.put((v * 255f).toInt().coerceIn(0, 255).toByte())
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun drain(interpreter: Interpreter): ByteBuffer =
        ByteBuffer.allocateDirect(interpreter.getOutputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())

    private fun readFloats(buffer: ByteBuffer, bytesPerElement: Int): FloatArray {
        buffer.rewind()
        return if (bytesPerElement == 2) {
            FloatArray(buffer.capacity() / 2) { android.util.Half.toFloat(buffer.short) }
        } else {
            FloatArray(buffer.capacity() / 4).also { buffer.asFloatBuffer().get(it) }
        }
    }

    private fun zeroInput(interpreter: Interpreter): ByteBuffer {
        val shape = interpreter.getInputTensor(0).shape()
        val elements = shape.fold(1) { a, b -> a * b }
        return ByteBuffer.allocateDirect(4 * elements).order(ByteOrder.nativeOrder())
    }

    private fun outputBuffer(interpreter: Interpreter): ByteBuffer {
        val shape = interpreter.getOutputTensor(0).shape()
        val elements = shape.fold(1) { a, b -> a * b }
        return ByteBuffer.allocateDirect(4 * elements).order(ByteOrder.nativeOrder())
    }
}
