package `in`.sarathi.app.vlm

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.util.Log
import java.io.File

/**
 * Fetching the scene-description weights, once, from inside the app.
 *
 * The model is 2.4 GB and cannot travel with the APK. Android's asset layer
 * caps a single asset below 2 GB, GitHub caps a repository file at 100 MB and
 * a release asset at 2 GB, and an install that large would exceed the free
 * space on many of the phones this is built for. So the app ships without it
 * and offers to fetch it the first time someone asks for a description.
 *
 * That keeps the promise that matters - one file to install, nothing to set up
 * by hand - without pretending the size problem away. Everything except this
 * one button works offline and always has; this is the only thing in Sarathi
 * that ever touches the network, it happens once, and only if asked.
 *
 * `DownloadManager` rather than a hand-rolled fetch: it survives the app being
 * closed, resumes across connection drops, shows progress in the notification
 * shade where a blind user's screen reader can reach it, and can be told to
 * wait for unmetered Wi-Fi. Re-implementing that badly is how a 2.4 GB
 * download becomes a support burden.
 */
object ModelDownload {

    private const val TAG = "SarathiDownload"
    private const val PREFS = "sarathi-download"
    private const val KEY_ID = "id"

    /**
     * The base build, deliberately - not the one called `-gpu`.
     *
     * The `-gpu` build is smaller, is named after the recommended backend, and
     * contains no vision encoder: it loads cleanly and then fails the moment an
     * image is attached. That cost 1.9 GB to discover once. See
     * docs/05-vlm.md.
     */
    const val URL =
        "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/" +
            "resolve/main/gemma-4-E2B-it.litertlm?download=true"

    const val FILE_NAME = "gemma-4-E2B-it.litertlm"

    /** Roughly what it costs, for a dialog that should not surprise anyone. */
    const val SIZE_GB = 2.4

    /** Where it lands. Readable without permission, visible over USB. */
    fun target(context: Context): File? =
        context.getExternalFilesDir("models")?.let { File(it, FILE_NAME) }

    /** A partial file from an interrupted attempt. */
    private fun partial(context: Context): File? =
        context.getExternalFilesDir("models")?.let { File(it, "$FILE_NAME.part") }

    fun isPresent(context: Context): Boolean = target(context)?.exists() == true

    /**
     * Start the download, or return the id of one already running.
     *
     * @param wifiOnly true keeps a 2.4 GB transfer off mobile data, which is
     * the difference between a slow evening and a month's allowance.
     */
    fun start(context: Context, wifiOnly: Boolean = true): Long? {
        val destination = partial(context) ?: run {
            Log.w(TAG, "no external files dir; cannot download")
            return null
        }
        destination.parentFile?.mkdirs()
        // A half-finished file from a previous attempt would otherwise be
        // appended to and silently corrupt the model.
        if (destination.exists()) destination.delete()

        val manager = context.getSystemService(Context.DOWNLOAD_SERVICE) as? DownloadManager
            ?: return null

        val request = DownloadManager.Request(Uri.parse(URL))
            .setTitle("Sarathi scene description")
            .setDescription("One-time download, about ${SIZE_GB} GB")
            .setNotificationVisibility(
                DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED
            )
            .setDestinationInExternalFilesDir(context, "models", "$FILE_NAME.part")
            .setAllowedOverRoaming(false)
            .setAllowedOverMetered(!wifiOnly)

        return runCatching { manager.enqueue(request) }
            .onSuccess {
                // Persisted, because completion must be read from
                // DownloadManager and the app may be restarted before then.
                context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                    .edit().putLong(KEY_ID, it).apply()
                Log.i(TAG, "queued download $it -> ${destination.path}")
            }
            .onFailure { Log.w(TAG, "could not queue download: $it") }
            .getOrNull()
    }

    /**
     * Move a finished download into place, if and only if it is really finished
     * and really correct.
     *
     * Two checks, and the first version had neither properly.
     *
     * **DownloadManager pre-allocates the destination to the full content
     * length**, so a `.part` file is 2,588,147,712 bytes from the first second.
     * A size check therefore passes immediately, and the original code renamed
     * a still-downloading file. It had the right length and the right header
     * and was wrong from the middle onwards; LiteRT-LM reported only "Failed to
     * load model from buffer". So completion is now taken from DownloadManager
     * itself, never inferred from the file.
     *
     * **And the bytes are hashed** against the sha256 the manifest declares.
     * That is what actually caught this: head matched, tail did not, size was
     * exact. Nothing weaker would have noticed - the file looked complete by
     * every cheap measure. Hashing 2.4 GB costs about half a minute, once.
     */
    fun finalise(context: Context, expectedSha256: String?): Boolean {
        val part = partial(context) ?: return false
        val done = target(context) ?: return false
        if (!part.exists()) return done.exists()

        if (!isComplete(context)) {
            Log.i(TAG, "download still in progress; leaving it alone")
            return false
        }
        if (expectedSha256 != null) {
            val actual = sha256(part)
            if (!actual.equals(expectedSha256, ignoreCase = true)) {
                Log.w(TAG, "checksum mismatch: expected $expectedSha256 got $actual; discarding")
                part.delete()
                return false
            }
            Log.i(TAG, "checksum verified")
        }
        return part.renameTo(done).also { Log.i(TAG, "model ready: $it") }
    }

    /** Whether the recorded download finished, according to DownloadManager. */
    private fun isComplete(context: Context): Boolean {
        val id = context
            .getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getLong(KEY_ID, -1L)
        if (id < 0) return false
        val manager = context.getSystemService(Context.DOWNLOAD_SERVICE) as? DownloadManager
            ?: return false
        manager.query(DownloadManager.Query().setFilterById(id))?.use { cursor ->
            if (!cursor.moveToFirst()) return false
            val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            return status == DownloadManager.STATUS_SUCCESSFUL
        }
        return false
    }

    private fun sha256(file: File): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val read = stream.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /** Human-readable progress, or null when nothing is running. */
    fun progress(context: Context, id: Long): String? {
        val manager = context.getSystemService(Context.DOWNLOAD_SERVICE) as? DownloadManager
            ?: return null
        manager.query(DownloadManager.Query().setFilterById(id))?.use { cursor ->
            if (!cursor.moveToFirst()) return null
            val status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            val soFar = cursor.getLong(
                cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
            )
            val total = cursor.getLong(
                cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
            )
            return when (status) {
                DownloadManager.STATUS_SUCCESSFUL -> "done"
                DownloadManager.STATUS_FAILED -> "failed"
                DownloadManager.STATUS_PAUSED -> "paused — waiting for Wi-Fi"
                else -> if (total > 0) {
                    "%.0f%% of %.1f GB".format(100.0 * soFar / total, total / 1e9)
                } else {
                    "starting…"
                }
            }
        }
        return null
    }
}
