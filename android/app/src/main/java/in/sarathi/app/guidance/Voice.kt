package `in`.sarathi.app.guidance

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * Speech and earcons.
 *
 * The policy mirrors the frame pipeline: **drop, don't queue.** If the system
 * is still speaking when the next thing becomes worth saying, the new
 * utterance is discarded rather than queued behind it. Queued speech describes
 * a world the user has already walked through, and a backlog only grows.
 * Silence is a valid output; stale speech is not.
 *
 * Urgency is the exception. It stops whatever is playing, sounds a tone and
 * speaks, because a rising tone reaches the user a few hundred milliseconds
 * before a spoken word can - and for something they are about to walk into
 * that gap is the whole reason earcons exist.
 */
class Voice(context: Context, private val onReady: (Boolean) -> Unit = {}) {

    private var tts: TextToSpeech? = null
    private var ready = false
    private val tone = ToneGenerator(AudioManager.STREAM_NOTIFICATION, 90)

    var spokenCount = 0L; private set
    var droppedCount = 0L; private set
    var interruptedCount = 0L; private set

    init {
        tts = TextToSpeech(context) { status ->
            ready = status == TextToSpeech.SUCCESS
            if (ready) {
                tts?.setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                // Faster than conversational. Experienced screen-reader users
                // run far quicker than this; it is exposed in settings rather
                // than fixed, because the right rate is personal.
                tts?.setSpeechRate(1.15f)
            }
            onReady(ready)
        }
    }

    /**
     * Devanagari read by an English voice is not accented Hindi, it is
     * unintelligible noise - so a missing language is reported rather than
     * silently falling back.
     */
    fun setLanguage(lang: String): Boolean {
        val locale = if (lang == "hi") Locale("hi", "IN") else Locale("en", "IN")
        val result = tts?.setLanguage(locale) ?: TextToSpeech.ERROR
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            Log.w(TAG, "no installed voice for $lang; the user must install one")
            return false
        }
        return true
    }

    val isSpeaking: Boolean get() = tts?.isSpeaking == true

    /** @return true if it was spoken, false if dropped. */
    fun say(text: String, urgent: Boolean, earcon: Boolean = false): Boolean {
        if (!ready || text.isBlank()) return false
        if (urgent) {
            if (isSpeaking) { tts?.stop(); interruptedCount++ }
            if (earcon) tone.startTone(ToneGenerator.TONE_PROP_BEEP2, 140)
            speak(text, TextToSpeech.QUEUE_FLUSH)
            spokenCount++
            return true
        }
        if (isSpeaking) {
            // Not queued: by the time it could play it would describe a scene
            // the user has walked past.
            droppedCount++
            return false
        }
        speak(text, TextToSpeech.QUEUE_FLUSH)
        spokenCount++
        return true
    }

    private fun speak(text: String, mode: Int) {
        tts?.speak(text, mode, Bundle(), "sarathi-${System.nanoTime()}")
    }

    fun stop() { tts?.stop() }

    fun shutdown() {
        runCatching { tts?.stop(); tts?.shutdown(); tone.release() }
        tts = null
    }

    private companion object { const val TAG = "SarathiVoice" }
}
