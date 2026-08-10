package `in`.sarathi.app.ocr

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.concurrent.atomic.AtomicBoolean

/**
 * On-demand text reading: point the camera at a sign and hear what it says.
 *
 * Like scene description, this is off the safety path and runs only when asked.
 * A door number, a bus route, a platform sign and a packet of medicine are all
 * things a blind person routinely needs and a bounding-box detector cannot
 * supply, but none of them is a hazard and none of them is urgent.
 *
 * ## Why ML Kit, on a project that is careful about licences
 *
 * This is the least clean dependency in Sarathi and it is worth being explicit
 * about rather than quiet.
 *
 * The artifacts used here are the **unbundled** Play Services variants, so no
 * proprietary model ships inside the APK - recognition runs in Google Play
 * Services, the same way [android.speech.tts.TextToSpeech] already does for
 * every word this app speaks. That is the basis for treating it the way
 * AGPL-3.0 treats a System Library: a component of the operating system the
 * work runs on, rather than part of the work.
 *
 * It is a defensible reading and not an airtight one, so the design does not
 * depend on it:
 *
 * - Text reading is **optional**. Where Play Services is absent or the model
 *   has not been delivered, [read] returns null and the rest of the app is
 *   unaffected. A de-Googled build simply has no OCR.
 * - The manifest still declares `rapidocr-mobile` with its Apache-2.0
 *   PaddleOCR-derived weights, which is the fully open path a fork can take by
 *   writing one adapter. The prototype already runs it.
 *
 * The alternative - porting DBNet detection plus a CRNN/CTC recogniser and a
 * character dictionary to LiteRT - is several hundred lines of numerical code
 * whose failure mode is confidently misread text. Shipping that half-validated
 * to a blind user would be worse than depending on a platform service, so the
 * trade was made knowingly in that direction.
 *
 * ## Scripts
 *
 * Two recognisers, chosen by the user's language. Devanagari is not a variant
 * of the Latin model - it is a separate one, and the Latin recogniser returns
 * nothing useful for a Hindi sign. Getting this wrong would look exactly like
 * "there is no text here", which is why the two are selected explicitly rather
 * than left to a default.
 */
class TextReader(language: String) {

    private val script = if (language == "hi") Script.DEVANAGARI else Script.LATIN
    private val busy = AtomicBoolean(false)

    private val recognizer: TextRecognizer? = runCatching {
        when (script) {
            Script.DEVANAGARI ->
                TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build())
            Script.LATIN ->
                TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        }
    }.onFailure { Log.w(TAG, "no recogniser for $script: $it") }.getOrNull()

    var lastReadMs: Long = 0; private set

    enum class Script { LATIN, DEVANAGARI }

    /**
     * Read every piece of text in [bitmap], in the order a person would.
     *
     * Returns null when there is nothing to read or nothing to read it with -
     * the same reason [SceneDescriber][`in`.sarathi.app.vlm.SceneDescriber]
     * collapses its failures: from the outside they are one event, "no text",
     * and a blind listener cannot act differently on the distinction anyway.
     *
     * Blocking, so call it off the camera thread.
     */
    fun read(bitmap: Bitmap): String? {
        val client = recognizer ?: return null
        if (!busy.compareAndSet(false, true)) {
            Log.i(TAG, "already reading; request dropped")
            return null
        }
        return try {
            val started = System.currentTimeMillis()
            val result = Tasks.await(client.process(InputImage.fromBitmap(bitmap, 0)))
            lastReadMs = System.currentTimeMillis() - started
            val text = order(result, bitmap.width)
            Log.i(TAG, "read ${result.textBlocks.size} blocks in ${lastReadMs}ms " +
                "($script): ${text?.take(120)}")
            text
        } catch (t: Throwable) {
            // Most often this is Play Services still fetching the recognition
            // model on a device that has never used it. Nothing to do but say
            // there was no text and let the user try again.
            Log.w(TAG, "read failed: $t")
            null
        } finally {
            busy.set(false)
        }
    }

    /**
     * Put the blocks into reading order and join them into something speakable.
     *
     * ML Kit returns blocks in roughly the order it found them, which for a
     * photograph of a sign is not the order a person reads. Sorting by vertical
     * band first and horizontal position second recovers rows: two labels side
     * by side stay side by side, and a caption underneath stays underneath.
     *
     * The band tolerance matters. Compare raw `top` values and text that is
     * visually on one line but a few pixels out of alignment gets interleaved
     * with the line below, which is unintelligible read aloud.
     */
    private fun order(result: Text, frameWidth: Int): String? {
        val blocks = result.textBlocks.mapNotNull { block ->
            block.boundingBox?.let { it to block.text.trim() }
        }.filter { it.second.isNotEmpty() }
        if (blocks.isEmpty()) return null

        val band = (blocks.map { it.first.height() }.average() * BAND_FRACTION)
            .coerceAtLeast(1.0)
        val sorted = blocks.sortedWith(
            compareBy({ (rect: Rect, _: String) -> (rect.top / band).toInt() }, { it.first.left })
        )

        val joined = sorted.joinToString(". ") { it.second.replace('\n', ' ') }
            .replace(Regex("\\s+"), " ")
            .replace(Regex("\\.\\s*\\."), ".")
            .trim()
        if (joined.isEmpty()) return null
        return if (joined.length > MAX_CHARS) {
            joined.take(MAX_CHARS).substringBeforeLast(' ') + "…"
        } else {
            joined
        }
    }

    /**
     * Read a bundled image whose text is known.
     *
     * "0 blocks" is ambiguous in exactly the way a detection count of zero is:
     * the view may genuinely hold no text, or the recogniser may never have
     * received its model from Play Services, or the wrong script may be
     * selected. On a device being operated with the screen off there is no
     * other way to tell those apart, and the difference decides whether the
     * user should point the camera somewhere else or stop trying.
     *
     * Mirrors `LiteRtDetector.selfTest`, for the same reason.
     */
    fun selfTest(context: android.content.Context): String {
        val bitmap = runCatching {
            context.assets.open("models/ocr-selftest.jpg").use {
                android.graphics.BitmapFactory.decodeStream(it)
            }
        }.getOrNull() ?: return "ocr self-test: no bundled image"
        val text = read(bitmap)
        bitmap.recycle()
        return "ocr self-test ($script): ${text?.take(120) ?: "NOTHING READ"} in ${lastReadMs}ms"
    }

    fun close() {
        runCatching { recognizer?.close() }
    }

    private companion object {
        const val TAG = "SarathiOCR"

        /**
         * How far apart two blocks can sit vertically and still count as one
         * line, as a fraction of the mean block height.
         */
        const val BAND_FRACTION = 0.6

        /**
         * About twenty seconds of speech. A dense sign can carry far more text
         * than anyone wants read at them, and the audio channel is shared with
         * hazard warnings.
         */
        const val MAX_CHARS = 400
    }
}
