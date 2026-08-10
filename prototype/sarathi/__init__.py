"""Sarathi - assistive scene understanding for blind and low-vision users.

Desktop prototype. The pipeline validated here is ported to Android once the
perception and guidance behaviour is proven; the model manifests and the
guidance phrasing tables are shared between the two, so a model benchmarked
here drops into the phone without code changes.

Pipeline shape:

    FrameSource -> Scheduler -> Perception -> Saliency -> Phrasing -> Speech
                      |            |
                   gating      detect / distance / (on-demand OCR, VLM)
"""

__version__ = "0.1.0"
