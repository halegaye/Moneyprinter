"""
Template overlay service for MoneyPrinterTurbo-Extended.
Applies a branded frame template over a generated video.
"""
import os
from typing import Optional

from loguru import logger


# Default content area of the template_frame.jpg (detected automatically)
# These are pixel coords (x1, y1, x2, y2) in the 576x1024 reference image
_DEFAULT_TEMPLATE = {
    "file": os.path.join(os.path.dirname(__file__), "..", "..", "resource", "public", "template_frame.jpg"),
    "frame_w": 576,
    "frame_h": 1024,
    # Content box (the black area where video is placed)
    "box_x1": 62,
    "box_y1": 229,
    "box_x2": 512,
    "box_y2": 869,
}


def apply_template(
    input_video_path: str,
    output_video_path: str,
    template_path: Optional[str] = None,
    video_scale: float = 1.0,
    bottom_padding_pct: float = 0.0,
) -> bool:
    """
    Overlay a branded frame template on top of an input video.

    The input video is resized and placed inside the black content area of the
    template.  ``video_scale`` shrinks the video so there is breathing room
    around the edges (especially the bottom), keeping subtitles visible.

    Args:
        input_video_path: Path to the source video file.
        output_video_path: Path where the branded video will be saved.
        template_path: Optional custom template image path. Falls back to the
                       bundled template_frame.jpg.
        video_scale: How much of the content box the video should fill
                     (0.0-1.0). Default 0.92 = 92 %, leaving ~4 % padding
                     on each side and ~8 % at the bottom for subtitles.
        bottom_padding_pct: Extra downward offset removed from the bottom of
                            the usable area (as a fraction of box_h).
                            Default 0.06 shifts the video up inside the box.

    Returns:
        True on success, False on failure.
    """
    try:
        from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, ColorClip
        import numpy as np
        from PIL import Image

        cfg = _DEFAULT_TEMPLATE.copy()
        if template_path and os.path.exists(template_path):
            cfg["file"] = template_path

        if not os.path.exists(cfg["file"]):
            logger.error(f"Template file not found: {cfg['file']}")
            return False

        # ---- 1. Load template & compute target dimensions ----
        tmpl_img = Image.open(cfg["file"]).convert("RGBA")
        tmpl_w, tmpl_h = tmpl_img.size

        # Scale template up to 1080-wide output
        scale = 1080 / tmpl_w
        out_w = int(tmpl_w * scale)
        out_h = int(tmpl_h * scale)

        # Scale content-box coordinates to output resolution
        bx1 = int(cfg["box_x1"] * scale)
        by1 = int(cfg["box_y1"] * scale)
        bx2 = int(cfg["box_x2"] * scale)
        by2 = int(cfg["box_y2"] * scale)
        box_w = bx2 - bx1
        box_h = by2 - by1

        logger.info(f"Template output size: {out_w}x{out_h}")
        logger.info(f"Content box: ({bx1},{by1}) -> ({bx2},{by2}), size={box_w}x{box_h}")

        # ---- 2. Compute the *usable* inner area for the video ----
        # Shrink box by video_scale and shift upward by bottom_padding_pct
        # so subtitles near the bottom of the frame stay inside the visible area.
        inner_w = int(box_w * video_scale)
        inner_h = int(box_h * video_scale)

        # Horizontal centering inside the box
        inner_x = bx1 + (box_w - inner_w) // 2
        # Vertical: bias upward — remove bottom_padding_pct from the bottom
        top_padding = int(box_h * (1.0 - video_scale) * 0.3)          # 30% of slack at top
        inner_y = by1 + top_padding

        logger.info(
            f"Video inner area: ({inner_x},{inner_y}), size={inner_w}x{inner_h} "
            f"(scale={video_scale:.0%}, bottom_pad={bottom_padding_pct:.0%})"
        )

        # ---- 3. Resize template to output size ----
        tmpl_img_resized = tmpl_img.resize((out_w, out_h), Image.LANCZOS)

        # ---- 4. Load and prepare the source video ----
        clip = VideoFileClip(input_video_path)
        src_w, src_h = clip.size
        src_ar = src_w / src_h
        inner_ar = inner_w / inner_h

        # Fit video into inner area (cover: crop to fill)
        if src_ar > inner_ar:
            # Video is wider → fit height, crop width
            fit_h = inner_h
            fit_w = int(src_ar * fit_h)
        else:
            # Video is taller → fit width, crop height
            fit_w = inner_w
            fit_h = int(fit_w / src_ar)

        resized_clip = clip.resized((fit_w, fit_h))

        # Center-crop to inner_w x inner_h
        cx = (fit_w - inner_w) // 2
        cy = (fit_h - inner_h) // 2
        cropped_clip = resized_clip.cropped(x1=cx, y1=cy, x2=cx + inner_w, y2=cy + inner_h)
        cropped_clip = cropped_clip.with_position((inner_x, inner_y))

        # ---- 5. Build template overlay as ImageClip (punch hole for video) ----
        tmpl_np = np.array(tmpl_img_resized)

        # Make the inner video area transparent so the clip shows through
        tmpl_np[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w, 3] = 0

        tmpl_clip = (
            ImageClip(tmpl_np, is_mask=False)
            .with_duration(clip.duration)
            .with_position((0, 0))
        )

        # ---- 6. Composite: black bg → video → template on top ----
        bg = ColorClip(size=(out_w, out_h), color=(0, 0, 0)).with_duration(clip.duration)
        final = CompositeVideoClip([bg, cropped_clip, tmpl_clip], size=(out_w, out_h))
        final = final.with_audio(clip.audio)

        # ---- 7. Export ----
        logger.info(f"Rendering branded video → {output_video_path}")
        final.write_videofile(
            output_video_path,
            fps=clip.fps or 30,
            codec="libx264",
            audio_codec="aac",
            logger=None,
            threads=4,
        )

        # Clean up
        clip.close()
        resized_clip.close()
        cropped_clip.close()
        tmpl_clip.close()
        final.close()

        logger.success(f"Branded video saved: {output_video_path}")
        return True

    except Exception as e:
        logger.error(f"apply_template failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False
