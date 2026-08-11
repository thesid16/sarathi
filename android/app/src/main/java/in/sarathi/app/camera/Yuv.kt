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

    /**
     * Decode to an **upright** bitmap.
     *
     * The rotation is not cosmetic and skipping it is not a small error. A
     * camera sensor is mounted landscape, so a phone held normally in portrait
     * delivers analysis frames rotated 90 degrees. Feed those to a detector
     * trained on upright photographs and it sees a world lying on its side:
     * people are horizontal, doorways are tunnels, and the ground plane runs
     * off the side of the image instead of the bottom.
     *
     * Two things then break at once. Detection collapses - measured on a Pixel
     * 8a, live frames never scored above 0.06 while the bundled upright
     * self-test image scored 0.75 through the identical code path. And every
     * distance becomes meaningless, because the geometric estimator reads the
     * bottom edge of a box as the point where the object meets the floor, and
     * in a sideways frame the bottom edge is a side wall.
     *
     * It fails quietly, which is why it survived so long: zero detections is
     * exactly what an empty room also produces.
     */
    fun toBitmap(image: ImageProxy, jpegQuality: Int = 88): Bitmap? = runCatching {
        val nv21 = toNv21(image)
        val stream = ByteArrayOutputStream(nv21.size / 4)
        YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
            .compressToJpeg(Rect(0, 0, image.width, image.height), jpegQuality, stream)
        val bytes = stream.toByteArray()
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return@runCatching null
        val degrees = image.imageInfo.rotationDegrees
        if (degrees == 0) return@runCatching decoded
        val matrix = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        android.graphics.Bitmap.createBitmap(
            decoded, 0, 0, decoded.width, decoded.height, matrix, true
        ).also { if (it != decoded) decoded.recycle() }
    }.getOrNull()
}
