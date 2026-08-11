package `in`.sarathi.app.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * YUV_420_888 handling, done properly.
 *
 * Two bugs lived here and both were invisible from the outside - the app ran,
 * inference ran, and every detection score came back at 0.001.
 *
 * **Row stride is not width.** Camera planes are padded to a hardware-friendly
 * alignment, so `rowStride` is usually larger than the image width. Indexing a
 * plane as `y * width + x` walks diagonally across the image, producing a
 * progressively sheared picture that still looks like plausible bytes.
 *
 * **Buffers have a position.** Reading the Y plane for the motion gate leaves
 * its position at the end, so a later read for the bitmap sees `remaining() ==
 * 0` and silently copies nothing. Every read here goes through `duplicate()`,
 * which gives an independent position over the same memory.
 */
object Yuv {

    // Scratch buffers, reused across frames.
    //
    // A 640x480 conversion allocates 1.2 MB for the pixels plus ~460 KB of
    // plane copies, and at even a few frames a second that is megabytes of
    // garbage per second competing with the detector for the same CPU. The
    // camera geometry does not change between frames, so the buffers do not
    // need to either. Only ever touched from the single camera executor.
    private var argb: IntArray = IntArray(0)
    private var yScratch: ByteArray = ByteArray(0)
    private var uScratch: ByteArray = ByteArray(0)
    private var vScratch: ByteArray = ByteArray(0)

    private fun sized(current: ByteArray, needed: Int): ByteArray =
        if (current.size >= needed) current else ByteArray(needed)

    /** Copy the luma plane, honouring row stride. Position-safe. */
    fun luma(image: ImageProxy): ByteArray {
        val plane = image.planes[0]
        val buffer = plane.buffer.duplicate()
        val width = image.width
        val height = image.height
        val out = ByteArray(width * height)
        if (plane.rowStride == width) {
            buffer.position(0)
            buffer.get(out, 0, minOf(out.size, buffer.remaining()))
            return out
        }
        val row = ByteArray(plane.rowStride)
        var offset = 0
        for (r in 0 until height) {
            buffer.position(r * plane.rowStride)
            val n = minOf(plane.rowStride, buffer.remaining())
            buffer.get(row, 0, n)
            System.arraycopy(row, 0, out, offset, minOf(width, n))
            offset += width
        }
        return out
    }

    /**
     * Pack to NV21: full-resolution Y, then interleaved V,U at half resolution.
     *
     * Chroma planes carry a `pixelStride` too - 2 on the semi-planar layouts
     * most phones produce - so they cannot be block-copied either.
     */
    fun toNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val out = ByteArray(ySize + ySize / 2)

        System.arraycopy(luma(image), 0, out, 0, ySize)

        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val uBuf = uPlane.buffer.duplicate()
        val vBuf = vPlane.buffer.duplicate()

        var offset = ySize
        val chromaW = width / 2
        val chromaH = height / 2
        for (r in 0 until chromaH) {
            for (c in 0 until chromaW) {
                val uIndex = r * uPlane.rowStride + c * uPlane.pixelStride
                val vIndex = r * vPlane.rowStride + c * vPlane.pixelStride
                if (vIndex < vBuf.limit()) out[offset++] = vBuf.get(vIndex)
                if (uIndex < uBuf.limit()) out[offset++] = uBuf.get(uIndex)
                if (offset >= out.size) return out
            }
        }
        return out
    }

    /** Bulk-copy a plane into a tightly packed array, honouring row stride. */
    private fun plane(
        buffer: java.nio.ByteBuffer, rowStride: Int, width: Int, height: Int,
        reuse: ByteArray? = null,
    ): ByteArray {
        val out = if (reuse != null && reuse.size >= width * height) reuse
                  else ByteArray(width * height)
        val src = buffer.duplicate()
        if (rowStride == width) {
            src.position(0)
            src.get(out, 0, minOf(out.size, src.remaining()))
            return out
        }
        for (row in 0 until height) {
            val offset = row * rowStride
            if (offset >= src.limit()) break
            src.position(offset)
            src.get(out, row * width, minOf(width, src.remaining()))
        }
        return out
    }

    /**
     * YUV_420_888 straight to an upright ARGB bitmap, in one pass.
     *
     * The original went NV21 -> JPEG -> Bitmap -> rotated Bitmap: four
     * full-image passes, two of them a JPEG codec, to change a pixel format.
     * Measured on a Pixel 8a it cost **78-135 ms per frame** - as much as
     * running the detector, and more than everything else in the pipeline put
     * together. Nothing had ever timed it; the logs reported inference latency,
     * and inference latency was fine.
     *
     * The obvious rewrite - one loop reading the planes directly - was
     * **slower**, at 133 ms. A per-pixel `ByteBuffer.get()` is a virtual call
     * on a direct buffer, and 300k pixels x 3 planes is a million of them
     * against libjpeg's hand-tuned native code. Beating native code from the
     * JVM needs the data in a primitive array first, where the JIT can emit
     * plain loads: the planes are bulk-copied once, and the hot loop then only
     * touches arrays.
     *
     * Rotation is folded into the destination index rather than done as a
     * second pass. It is not optional - a sensor is mounted landscape, so a
     * phone held in portrait delivers frames on their side.
     *
     * BT.601 full-range, integer fixed-point, which is what CameraX carries.
     */
    fun toBitmap(image: ImageProxy, @Suppress("UNUSED_PARAMETER") jpegQuality: Int = 88): Bitmap? =
        runCatching {
            val width = image.width
            val height = image.height
            val chromaW = (width + 1) / 2
            val chromaH = (height + 1) / 2

            val yp = image.planes[0]
            val up = image.planes[1]
            val vp = image.planes[2]
            yScratch = sized(yScratch, width * height)
            val yData = plane(yp.buffer, yp.rowStride, width, height, yScratch)

            // Chroma planes may be interleaved (pixelStride 2), in which case a
            // bulk row copy picks up the other component in between; de-swizzle
            // once here rather than per pixel.
            uScratch = sized(uScratch, chromaW * chromaH)
            vScratch = sized(vScratch, chromaW * chromaH)
            val uData = chroma(up.buffer, up.rowStride, up.pixelStride, chromaW, chromaH, uScratch)
            val vData = chroma(vp.buffer, vp.rowStride, vp.pixelStride, chromaW, chromaH, vScratch)

            val rotation = ((image.imageInfo.rotationDegrees % 360) + 360) % 360
            val swap = rotation == 90 || rotation == 270
            val outW = if (swap) height else width
            val outH = if (swap) width else height
            if (argb.size < outW * outH) argb = IntArray(outW * outH)
            val out = argb

            for (row in 0 until height) {
                val yRow = row * width
                val cRow = (row shr 1) * chromaW
                for (col in 0 until width) {
                    val luma = yData[yRow + col].toInt() and 0xFF
                    val chroma = cRow + (col shr 1)
                    val cb = (uData[chroma].toInt() and 0xFF) - 128
                    val cr = (vData[chroma].toInt() and 0xFF) - 128

                    val r = (luma + ((91881 * cr) shr 16)).coerceIn(0, 255)
                    val g = (luma - ((22554 * cb + 46802 * cr) shr 16)).coerceIn(0, 255)
                    val b = (luma + ((116130 * cb) shr 16)).coerceIn(0, 255)

                    val index = when (rotation) {
                        90 -> col * outW + (outW - 1 - row)
                        180 -> (outH - 1 - row) * outW + (outW - 1 - col)
                        270 -> (outH - 1 - col) * outW + row
                        else -> row * outW + col
                    }
                    out[index] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            Bitmap.createBitmap(out, outW, outH, Bitmap.Config.ARGB_8888)
        }.getOrNull()

    private fun chroma(
        buffer: java.nio.ByteBuffer, rowStride: Int, pixelStride: Int, width: Int, height: Int,
        reuse: ByteArray? = null,
    ): ByteArray {
        if (pixelStride == 1) return plane(buffer, rowStride, width, height, reuse)
        val src = buffer.duplicate()
        val row = ByteArray(rowStride)
        val out = if (reuse != null && reuse.size >= width * height) reuse
                  else ByteArray(width * height)
        for (y in 0 until height) {
            val offset = y * rowStride
            if (offset >= src.limit()) break
            src.position(offset)
            val n = minOf(rowStride, src.remaining())
            src.get(row, 0, n)
            var x = 0
            var i = 0
            while (x < width && i < n) {
                out[y * width + x] = row[i]
                x++
                i += pixelStride
            }
        }
        return out
    }
}
