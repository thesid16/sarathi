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
        fun toBitmap(): Bitmap? = Yuv.toBitmap(proxy)

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
                // Position-safe and stride-aware. Reading the plane buffer
                // directly leaves its position at the end, so the later read
                // for the bitmap saw nothing.
                val luma = Yuv.luma(proxy)
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

}
