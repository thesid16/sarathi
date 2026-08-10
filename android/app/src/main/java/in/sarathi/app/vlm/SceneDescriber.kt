package `in`.sarathi.app.vlm

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import `in`.sarathi.app.models.ModelManifest
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * On-demand scene description with Gemma 4 E2B, via LiteRT-LM.
 *
 * **This is not on the safety path, and the separation is deliberate.**
 * Hazards come from the detector and the geometric estimator: bounded, ~30 ms,
 * measured against held-out data, and wrong in ways that show up as a missed
 * box. A language model is none of those things. It takes seconds, holds most
 * of a gigabyte, and when it is wrong it is wrong in fluent confident prose -
 * which is the worst possible failure mode for someone who cannot check it
 * against what they see.
 *
 * So it answers a question, and only when asked. The user presses volume-up and
 * gets a sentence about what is in front of them. Nothing waits on it, nothing
 * is warned by it, and if it never loads at all the app is exactly as safe as
 * it was before.
 *
 * Three properties matter more than quality here:
 *
 * **It must not be resident when unused.** 1.9 GB on disk and ~700 MB live is
 * unaffordable on the phones much of this audience owns, so the engine is
 * built on first request and released after [IDLE_UNLOAD_MS] of silence or
 * whenever Android asks for memory back.
 *
 * **It must not queue.** A second press while a description is in flight is
 * dropped, not stacked - the same rule the speech layer follows, for the same
 * reason: an answer about a scene the user has already walked out of is worse
 * than no answer.
 *
 * **It must say something immediately.** Engine initialisation alone can take
 * ten seconds. To a blind user a silent device is a broken device, so the
 * caller is told to speak an acknowledgement before the work starts.
 */
class SceneDescriber(
    private val context: Context,
    private val manifest: ModelManifest,
) {
    private var engine: Engine? = null
    private var lastUsedAt = 0L
    private val busy = AtomicBoolean(false)

    /** Set once the model has been loaded at least once, for diagnostics. */
    var loadMs: Long = 0; private set
    var lastDescribeMs: Long = 0; private set

    /** Whether the weights are present. Nothing else here works without them. */
    fun isInstalled(): Boolean = weightsFile()?.exists() == true

    /**
     * Whether the installed file can actually see.
     *
     * A .litertlm is a container of named sections, and not every published
     * build of the same model holds the same ones. The `-gpu` build of Gemma 4
     * E2B contains exactly one section, `tf_lite_artisan_text_decoder`, and no
     * vision encoder at all - which nothing in its model card says, and which
     * surfaces only when the conversation is created, eleven seconds into a
     * request the user is waiting on, as `NOT_FOUND: TF_LITE_VISION_ENCODER`.
     *
     * Section names sit in the first kilobyte, so this reads 4 KB and answers
     * the question before any of that happens.
     */
    fun hasVisionEncoder(): Boolean {
        val file = weightsFile() ?: return false
        if (!file.exists()) return false
        return runCatching {
            val head = ByteArray(4096)
            val read = file.inputStream().use { it.read(head) }
            if (read <= 0) return false
            String(head, 0, read, Charsets.ISO_8859_1).contains(VISION_SECTION)
        }.getOrDefault(false)
    }

    private fun weightsFile(): File? =
        manifest.fileFor("android")?.let { File(File(context.filesDir, "models"), it) }

    /**
     * Describe [bitmap] in one sentence, or return null.
     *
     * Null covers every reason this can decline - weights absent, a request
     * already running, the engine failing to build - because to the layer above
     * they are the same event: there is no description to speak. The reason is
     * logged for whoever is diagnosing, not surfaced as four different failures
     * the user would have to tell apart by ear.
     */
    fun describe(bitmap: Bitmap, prompt: String): String? {
        if (!busy.compareAndSet(false, true)) {
            Log.i(TAG, "already describing; request dropped")
            return null
        }
        try {
            // Stamped before the work, not after. Set only on success, a failed
            // attempt leaves this at zero, the very next frame sees an engine
            // idle since the epoch and unloads it - so every retry pays the
            // eleven-second load again. The log line "unloading after
            // 1786394478s idle" is what that looks like from outside.
            lastUsedAt = System.currentTimeMillis()
            val engine = engine() ?: return null
            val frame = File(context.cacheDir, "vlm-frame.jpg")
            frame.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 88, it) }

            val started = System.currentTimeMillis()
            val answer = engine.createConversation(
                ConversationConfig(
                    systemInstruction = Contents.of(Content.Text(SYSTEM_INSTRUCTION)),
                    // A hard ceiling, not a hint. The system instruction asks
                    // for one sentence and a model may still write a paragraph;
                    // this bounds both the wait and the length of speech that
                    // would sit in front of the next hazard warning.
                    maxOutputToken = MAX_OUTPUT_TOKENS,
                )
            ).use { conversation: Conversation ->
                val reply = conversation.sendMessage(
                    Contents.of(
                        Content.ImageFile(frame.absolutePath),
                        Content.Text(prompt),
                    )
                )
                reply.contents.contents
                    .filterIsInstance<Content.Text>()
                    .joinToString("") { it.text }
            }
            lastDescribeMs = System.currentTimeMillis() - started
            lastUsedAt = System.currentTimeMillis()
            frame.delete()
            Log.i(TAG, "described in ${lastDescribeMs}ms: ${answer.take(120)}")
            return tidy(answer)
        } catch (t: Throwable) {
            Log.w(TAG, "describe failed: $t")
            return null
        } finally {
            busy.set(false)
        }
    }

    /**
     * Trim the model's answer into something worth speaking.
     *
     * Instruction-tuned models like to open with "The image shows" or "In this
     * picture", which costs a blind user a second of speech to learn nothing.
     * Markdown decoration is worse than useless out loud. And the length cap is
     * a safety property, not tidiness: a paragraph read aloud occupies the
     * audio channel for long enough that a real hazard announcement would be
     * dropped behind it.
     */
    private fun tidy(raw: String): String? {
        var text = raw.trim().replace(Regex("[*_#`]"), "")
        for (opener in OPENERS) {
            if (text.startsWith(opener, ignoreCase = true)) {
                text = text.substring(opener.length).trimStart()
                text = text.replaceFirstChar { it.uppercase() }
                break
            }
        }
        text = text.lines().firstOrNull { it.isNotBlank() }?.trim() ?: return null
        if (text.isEmpty()) return null
        return if (text.length > MAX_CHARS) text.take(MAX_CHARS).substringBeforeLast(' ') + "…"
        else text
    }

    private fun engine(): Engine? {
        engine?.let { return it }
        val weights = weightsFile()
        if (weights == null || !weights.exists()) {
            Log.i(TAG, "${manifest.id} not installed (${weights?.path}); " +
                "distribution=${manifest.distribution}")
            return null
        }
        if (!hasVisionEncoder()) {
            Log.w(TAG, "${weights.name} has no $VISION_SECTION section - this is a " +
                "text-only build and cannot describe an image")
            return null
        }
        return try {
            val started = System.currentTimeMillis()
            // GPU first. The vendor's own figures put it ahead of CPU on both
            // speed and memory, and unlike the detector's GPU delegate this
            // path is the one LiteRT-LM is built around - so it is tried
            // rather than assumed broken by association.
            val built = build(weights, Backend.GPU()) ?: build(weights, Backend.CPU())
            loadMs = System.currentTimeMillis() - started
            lastUsedAt = System.currentTimeMillis()
            if (built != null) Log.i(TAG, "engine ready in ${loadMs}ms")
            engine = built
            built
        } catch (t: Throwable) {
            Log.w(TAG, "engine init failed: $t")
            null
        }
    }

    private fun build(weights: File, backend: Backend): Engine? = try {
        Engine(
            EngineConfig(
                modelPath = weights.absolutePath,
                backend = backend,
                visionBackend = backend,
                // One image per request, and the request is a single question
                // about right now. Declaring that keeps the KV cache and image
                // buffers sized for what this app actually does rather than for
                // a chat client's worst case.
                maxNumImages = 1,
                // Persisting compiled kernels is what turns a ten-second first
                // load into something bearable on the second press.
                cacheDir = context.cacheDir.absolutePath,
            )
        ).also { it.initialize() }
    } catch (t: Throwable) {
        Log.w(TAG, "backend ${backend::class.simpleName} unavailable: $t")
        null
    }

    /**
     * Release the engine if it has been idle. Called from the frame loop, which
     * already runs on a timer, rather than from a timer of its own.
     */
    fun trimIfIdle(now: Long = System.currentTimeMillis()) {
        val live = engine ?: return
        if (busy.get() || now - lastUsedAt < IDLE_UNLOAD_MS) return
        Log.i(TAG, "unloading after ${(now - lastUsedAt) / 1000}s idle")
        runCatching { live.close() }
        engine = null
    }

    fun close() {
        runCatching { engine?.close() }
        engine = null
    }

    companion object {
        private const val TAG = "SarathiVLM"

        /** Container section that has to be present for image input to work. */
        private const val VISION_SECTION = "tf_lite_vision_encoder"

        /** Idle time after which the weights are handed back to the system. */
        const val IDLE_UNLOAD_MS = 90_000L

        /** Roughly twelve seconds of speech. Past that it stops being an answer. */
        private const val MAX_CHARS = 220

        /** ~30 words of headroom, matching what the system instruction asks for. */
        private const val MAX_OUTPUT_TOKENS = 64

        private val OPENERS = listOf(
            "the image shows", "this image shows", "the picture shows",
            "in this image,", "in the image,", "the photo shows",
            "i see", "here is", "this is a picture of", "this appears to be",
        )

        /**
         * The system instruction does the heavy lifting. A general-purpose VLM
         * describes a photograph; this user is standing in a place and needs to
         * know what is there, which is a different question with a different
         * answer.
         */
        private const val SYSTEM_INSTRUCTION =
            "You describe scenes for a blind person who is standing where the " +
                "camera is pointing. Answer in one short sentence, under 30 words. " +
                "Lead with what is directly ahead and closest. Name things plainly. " +
                "Give positions as left, ahead or right. Do not mention the image, " +
                "the photo, or the camera. Do not guess at anything you cannot see. " +
                "If the view is too dark or blurred to read, say exactly that."
    }
}
