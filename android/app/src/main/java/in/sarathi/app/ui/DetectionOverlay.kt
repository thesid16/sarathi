package `in`.sarathi.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.view.View
import `in`.sarathi.app.guidance.Detection
import `in`.sarathi.app.models.Hazard

/**
 * Boxes drawn over the live preview.
 *
 * This is the single most useful thing on the screen, because it answers the
 * question every other readout only hints at: *is the machine seeing what I am
 * seeing?* A latency figure cannot tell you the camera is upside down. A box
 * drawn around a doorway can.
 *
 * Colour carries hazard level rather than class identity. Twenty-six distinct
 * hues would be decoration; four that map onto how urgently something matters
 * mean a glance is enough to know whether the system has understood the scene
 * the way a person would.
 */
class DetectionOverlay(context: Context) : View(context) {

    private var detections: List<Detection> = emptyList()
    private var frameWidth = 0
    private var frameHeight = 0

    private val box = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val labelBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val labelText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        isFakeBoldText = true
    }
    private val emptyText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(150, 255, 255, 255)
        textSize = 38f
        textAlign = Paint.Align.CENTER
    }

    fun update(found: List<Detection>, width: Int, height: Int) {
        detections = found
        frameWidth = width
        frameHeight = height
        invalidate()
    }

    fun clear() = update(emptyList(), frameWidth, frameHeight)

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (frameWidth <= 0 || frameHeight <= 0) return

        // The preview is letterboxed to preserve aspect ratio, so the drawing
        // area is not the whole view. Scaling by width alone would put every
        // box a little too low on a tall screen - subtly wrong in a way that
        // looks like a calibration problem rather than a drawing bug.
        val scale = minOf(width / frameWidth.toFloat(), height / frameHeight.toFloat())
        val drawnW = frameWidth * scale
        val drawnH = frameHeight * scale
        val offsetX = (width - drawnW) / 2f
        val offsetY = (height - drawnH) / 2f

        if (detections.isEmpty()) {
            canvas.drawText("nothing detected", width / 2f, height - 48f, emptyText)
            return
        }

        for (detection in detections) {
            val colour = colourFor(detection.hazard)
            box.color = colour
            val rect = RectF(
                offsetX + detection.box[0] * scale,
                offsetY + detection.box[1] * scale,
                offsetX + detection.box[2] * scale,
                offsetY + detection.box[3] * scale,
            )
            canvas.drawRoundRect(rect, 8f, 8f, box)

            val distance = detection.distanceM?.let { " · %.1f m".format(it) } ?: ""
            val caption = "${detection.label}$distance"
            val textWidth = labelText.measureText(caption)
            // Labels flip below the box near the top edge rather than being
            // clipped off the screen, which is where the nearest and most
            // urgent objects tend to sit.
            val above = rect.top > 52f
            val top = if (above) rect.top - 46f else rect.bottom + 4f
            labelBg.color = colour
            canvas.drawRoundRect(
                RectF(rect.left, top, rect.left + textWidth + 22f, top + 44f), 6f, 6f, labelBg
            )
            canvas.drawText(caption, rect.left + 11f, top + 32f, labelText)
        }
    }

    private fun colourFor(hazard: Hazard): Int = when (hazard) {
        Hazard.CRITICAL -> Color.rgb(255, 82, 82)
        Hazard.HIGH -> Color.rgb(255, 145, 48)
        Hazard.MEDIUM -> Color.rgb(255, 209, 71)
        Hazard.LOW -> Color.rgb(120, 214, 168)
    }
}
