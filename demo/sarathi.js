// Sarathi in the browser.
//
// The same model and the same decoding rules as the phone, reading the same
// letterbox geometry. It is a demonstration, not the product: there is no
// motion gate, no thermal governor and no metric distance, because a laptop
// has neither the sensors nor the constraints those exist for.
//
// What it does carry over is the part that matters most and is hardest to
// explain in prose - that the system chooses what to say and stays quiet about
// the rest.

export const COCO_TO_HAZARD = {
  // Only the classes a walking person meets. Everything else is LOW, which
  // means it is context: shown on screen, never announced.
  person: "HIGH", bicycle: "HIGH", car: "CRITICAL", motorcycle: "CRITICAL",
  bus: "CRITICAL", truck: "CRITICAL", train: "CRITICAL",
  "traffic light": "MEDIUM", "stop sign": "MEDIUM", bench: "MEDIUM",
  chair: "MEDIUM", couch: "MEDIUM", "potted plant": "MEDIUM", dog: "HIGH",
  "fire hydrant": "MEDIUM", "dining table": "MEDIUM", tv: "LOW",
};

const HAZARD_LEVEL = { LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4 };
const HAZARD_COLOUR = {
  CRITICAL: "#FF5252", HIGH: "#FF9130", MEDIUM: "#FFD147", LOW: "#78D6A8",
};

/**
 * Letterbox an image into a square tensor, exactly as the phone does.
 *
 * Aspect-preserving with grey padding. Stretching instead would move every
 * box, and the boxes are the only evidence a viewer has that the thing works.
 * Returns the transform so detections can be mapped back to source pixels.
 */
export function letterbox(source, size, ctx) {
  // Intrinsic size, not layout size. On a <video>, `.width` is the HTML
  // attribute - 0 unless someone set it - while the frame is videoWidth. Using
  // `.width` made ratio Infinity and w/h NaN, and drawImage with non-finite
  // arguments is specified to return silently. The model then ran, fast and
  // happily, on the grey fill below: 0 detected, peak score 0.00, no error.
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  if (!sw || !sh) {
    throw new Error(`letterbox: source has no intrinsic size (${sw}x${sh})`);
  }

  const ratio = Math.min(size / sw, size / sh);
  const w = Math.round(sw * ratio);
  const h = Math.round(sh * ratio);
  const padX = Math.floor((size - w) / 2);
  const padY = Math.floor((size - h) / 2);

  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, w, h);

  const { data } = ctx.getImageData(0, 0, size, size);
  // NCHW float32, 0..1 - the layout the exported graph expects.
  const tensor = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    tensor[p] = data[i] / 255;
    tensor[plane + p] = data[i + 1] / 255;
    tensor[2 * plane + p] = data[i + 2] / 255;
  }
  return { tensor, ratio, padX, padY };
}

/**
 * Decode an Ultralytics head: [1, 4+nc, anchors], centre-form xywh in input
 * pixels, class scores already post-sigmoid, no separate objectness.
 *
 * Treating row 4 as objectness is the classic porting bug and produces
 * plausible-looking boxes with wrong labels, so it is spelled out here.
 */
export function decode(raw, dims, labels, { confThreshold = 0.35 } = {}) {
  const [, channels, anchors] = dims;
  const numClasses = channels - 4;
  const at = (channel, anchor) => raw[channel * anchors + anchor];

  const candidates = [];
  let maxScore = 0;
  for (let i = 0; i < anchors; i++) {
    let best = 0;
    let bestScore = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = at(4 + c, i);
      if (s > bestScore) { bestScore = s; best = c; }
    }
    if (bestScore > maxScore) maxScore = bestScore;
    if (bestScore < confThreshold) continue;
    const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
    candidates.push({
      box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
      score: bestScore,
      label: labels[best] ?? String(best),
    });
  }
  return { candidates, maxScore };
}

function iou(a, b) {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Per-class NMS, then map back to source pixels.
 *
 * Per class rather than class-agnostic: a person standing in a doorway is two
 * overlapping boxes and both matter.
 */
export function nms(candidates, transform, frameW, frameH, { iouThreshold = 0.5, max = 50 } = {}) {
  const byClass = new Map();
  for (const c of candidates) {
    if (!byClass.has(c.label)) byClass.set(c.label, []);
    byClass.get(c.label).push(c);
  }
  const kept = [];
  for (const group of byClass.values()) {
    group.sort((a, b) => b.score - a.score);
    while (group.length) {
      const best = group.shift();
      kept.push(best);
      for (let i = group.length - 1; i >= 0; i--) {
        if (iou(best.box, group[i].box) > iouThreshold) group.splice(i, 1);
      }
    }
  }
  const { ratio, padX, padY } = transform;
  return kept
    .sort((a, b) => b.score - a.score)
    .slice(0, max)
    .map((c) => ({
      ...c,
      hazard: COCO_TO_HAZARD[c.label] ?? "LOW",
      colour: HAZARD_COLOUR[COCO_TO_HAZARD[c.label] ?? "LOW"],
      box: [
        Math.max(0, Math.min(frameW, (c.box[0] - padX) / ratio)),
        Math.max(0, Math.min(frameH, (c.box[1] - padY) / ratio)),
        Math.max(0, Math.min(frameW, (c.box[2] - padX) / ratio)),
        Math.max(0, Math.min(frameH, (c.box[3] - padY) / ratio)),
      ],
    }));
}

/**
 * Rank detections and pick at most one thing worth saying.
 *
 * A simplified port of the phone's saliency engine. Without a calibrated
 * camera there is no metric distance, so box height stands in for proximity -
 * a thing that fills more of the frame is nearer. That is a demo compromise
 * and it is the reason this page shows no distances: a number that cannot be
 * trusted is worse than no number, which is the rule the whole project runs on.
 */
export function rank(detections, frameW, frameH, { scoreFloor = 0.55, announceLow = false } = {}) {
  const ranked = detections.map((d) => {
    const [x1, y1, x2, y2] = d.box;
    const centre = (x1 + x2) / 2;
    // Proximity from apparent size, path from how central it is.
    const proximity = Math.min(1, (y2 - y1) / (frameH * 0.75));
    const offset = Math.abs(centre - frameW / 2) / (frameW / 2);
    const path = Math.exp(-Math.pow(offset / 0.45, 2));
    const hazard = Math.pow(HAZARD_LEVEL[d.hazard] / 4, 1.5);
    let score = Math.min(1, proximity * path + 0.45 * hazard);
    if (d.hazard === "LOW" && !announceLow) score = 0;
    return { ...d, score, inPath: offset < 0.45 };
  }).sort((a, b) => b.score - a.score);

  const best = ranked[0];
  if (!best) return { ranked, chosen: null, reason: "nothing detected" };
  if (best.score < scoreFloor) {
    return {
      ranked,
      chosen: null,
      reason: best.hazard === "LOW"
        ? `${best.label}: low hazard, context only`
        : `${best.label}: ${best.score.toFixed(2)} below ${scoreFloor}`,
    };
  }
  return { ranked, chosen: best, reason: "" };
}

/** Clock-face bearing, the phrasing the phone uses. */
export function bearingPhrase(box, frameW) {
  const centre = (box[0] + box[2]) / 2;
  const offset = (centre - frameW / 2) / (frameW / 2);
  if (Math.abs(offset) < 0.12) return "ahead";
  if (offset < -0.55) return "on your left";
  if (offset > 0.55) return "on your right";
  return offset < 0 ? "eleven o'clock" : "one o'clock";
}
