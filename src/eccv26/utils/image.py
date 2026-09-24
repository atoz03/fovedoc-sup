from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return img.convert("RGB")


def resize_to_max_pixels(img: Image.Image, max_pixels: int | None) -> tuple[Image.Image, dict[str, Any]]:
    """
    将图片按最大像素数约束做等比例缩放（max_pixels=None 表示不缩放）。
    返回：(新图, 元信息)
    """
    img = ensure_rgb(img)
    w, h = img.size
    meta = {"orig_width": w, "orig_height": h, "resized": False}
    if max_pixels is None:
        return img, meta

    if w * h <= max_pixels:
        return img, meta

    scale = (max_pixels / float(w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    meta.update({"resized": True, "new_width": new_w, "new_height": new_h})
    return resized, meta

