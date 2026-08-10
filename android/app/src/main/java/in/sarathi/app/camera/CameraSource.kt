package `in`.sarathi.app.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/**
 * The phone's own camera, with the drop-don't-queue policy.
 *
 * CameraX's KEEP_ONLY_LATEST is exactly the right backpressure strategy here
 * and is the same rule the prototype enforces by hand: if inference is slower
 * than capture, intermediate frames are discarded rather than buffered. A queue
 * would show healthy throughput while guidance drifted further behind reality -
 * describing a scene the user has already walked through. For an assistive
 * product that is a safety defect, not a performance one.
 *
 * The Y plane is handed to the motion gate directly. Converting every frame to
 * a bitmap just to decide whether to skip it would cost more than the decision
 * saves.
 */
class CameraSource(private val context: Context) {

    private val executor = Executors.newSingleThreadExecutor()
    private var provider: ProcessCameraProvider? = null

    data class Frame(
        val luma: ByteArray,
        val width: Int,
        val height: Int,
        val timestampMs: Long,
        val rotationDegrees: Int,
        private val proxy: ImageProxy,
    ) {
        /** Only called when the frame actually reaches inference. */
        fun toBitmap(): Bitmap? = yuvToBitmap(proxy)

        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    fun start(owner: LifecycleOwner, onFrame: (Frame) -> Unit) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val cameraProvider = future.get()
            provider = cameraProvider

            val analysis = ImageAnalysis.Builder()
                // 640x480 is plenty: the detector runs at 320 and a larger
                // capture only costs conversion time and power.
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            analysis.setAnalyzer(executor) { proxy ->
                val plane = proxy.planes[0]
                val luma = ByteArray(plane.buffer.remaining())
                plane.buffer.get(luma)
                val frame = Frame(
                    luma = luma,
                    width = proxy.width,
                    height = proxy.height,
                    timestampMs = System.currentTimeMillis(),
                    rotationDegrees = proxy.imageInfo.rotationDegrees,
                    proxy = proxy,
                )
                try {
                    onFrame(frame)
                } finally {
                    proxy.close()
                }
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        provider?.unbindAll()
        provider = null
        executor.shutdown()
    }

    companion object {
        fun yuvToBitmap(proxy: ImageProxy): Bitmap? = runCatching {
            val y = proxy.planes[0].buffer
            val u = proxy.planes[1].buffer
            val v = proxy.planes[2].buffer
            val nv21 = ByteArray(y.remaining() + u.remaining() + v.remaining())
            y.get(nv21, 0, y.remaining())
            val chromaStart = nv21.size - u.remaining() - v.remaining()
            v.get(nv21, chromaStart, v.remaining())
            u.get(nv21, chromaStart + v.remaining().coerceAtLeast(0), u.remaining())

            val out = ByteArrayOutputStream()
            YuvImage(nv21, ImageFormat.NV21, proxy.width, proxy.height, null)
                .compressToJpeg(Rect(0, 0, proxy.width, proxy.height), 88, out)
            val bytes = out.toByteArray()
            android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        }.getOrNull()
    }
}
