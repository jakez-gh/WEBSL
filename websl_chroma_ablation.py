# ============================================================
# WEBSL — ABLATION CHROMA (IMAGES OPAQUES)
# Auteur : Sam Hassanine
# ============================================================

import os
import math
import zlib
import struct
import numpy as np
import torch

from PIL import Image
from pytorch_msssim import ms_ssim

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def psnr(a, b):
    e = rmse(a, b)
    if e <= 1e-12:
        return 999.0
    return float(20.0 * np.log10(1.0 / e))

def fmt_kb(n):
    return f"{n/1024:.1f} KB"

def resize_rgb(img01, new_w, new_h, resample=Image.LANCZOS):
    img8 = (clip01(img01) * 255.0 + 0.5).astype(np.uint8)
    out = Image.fromarray(img8).resize((new_w, new_h), resample)
    return np.asarray(out).astype(np.float32) / 255.0

def save_webp(path, img01, quality=80):
    img8 = (clip01(img01) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8).save(path, format="WEBP", quality=int(quality), method=6)
    return os.path.getsize(path)

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

def compute_ms_ssim(img_a, img_b):
    ta = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).float()
    tb = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).float()
    return float(ms_ssim(ta, tb, data_range=1.0, size_average=True))

def rgb_to_ycbcr(img):
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    Y  = 0.299000 * R + 0.587000 * G + 0.114000 * B
    Cb = 0.5 + (-0.168736 * R - 0.331264 * G + 0.500000 * B)
    Cr = 0.5 + ( 0.500000 * R - 0.418688 * G - 0.081312 * B)
    return clip01(np.stack([Y, Cb, Cr], axis=-1))

def ycbcr_to_rgb(ycbcr):
    Y  = ycbcr[..., 0]
    Cb = ycbcr[..., 1] - 0.5
    Cr = ycbcr[..., 2] - 0.5
    R = Y + 1.402000 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772000 * Cb
    return clip01(np.stack([R, G, B], axis=-1))

def quantize_index(x, levels):
    q = np.floor(clip01(x) * (levels - 1) + 0.5).astype(np.int32)
    return np.clip(q, 0, levels - 1)

def dequantize_index(q, levels):
    if levels <= 1:
        return np.zeros_like(q, dtype=np.float32)
    return q.astype(np.float32) / float(levels - 1)

def encode_fixed_grain_quantized(channel01, grain_size=2, levels=32):
    h, w = channel01.shape
    gh = math.ceil(h / grain_size)
    gw = math.ceil(w / grain_size)
    grid_q = np.zeros((gh, gw), dtype=np.int32)
    for gy in range(gh):
        y0 = gy * grain_size
        y1 = min(y0 + grain_size, h)
        for gx in range(gw):
            x0 = gx * grain_size
            x1 = min(x0 + grain_size, w)
            patch = channel01[y0:y1, x0:x1]
            grid_q[gy, gx] = int(quantize_index(np.array([float(np.mean(patch))]), levels)[0])
    return grid_q

def decode_fixed_grain_quantized(grid_q, out_h, out_w, levels=32):
    gh, gw = grid_q.shape
    out = np.zeros((out_h, out_w), dtype=np.float32)
    sy = out_h / gh
    sx = out_w / gw
    for gy in range(gh):
        y0 = int(round(gy * sy))
        y1 = int(round((gy + 1) * sy))
        y1 = max(y1, y0 + 1)
        y1 = min(y1, out_h)
        for gx in range(gw):
            x0 = int(round(gx * sx))
            x1 = int(round((gx + 1) * sx))
            x1 = max(x1, x0 + 1)
            x1 = min(x1, out_w)
            v = float(dequantize_index(np.array([grid_q[gy, gx]]), levels)[0])
            out[y0:y1, x0:x1] = v
    return clip01(out)

def build_native_opaque_image(img, grain_size_Y=2, grain_size_C=2, levels_Y=32, levels_C=16):
    h, w = img.shape[:2]
    ycbcr = rgb_to_ycbcr(img)
    Y, Cb, Cr = ycbcr[...,0], ycbcr[...,1], ycbcr[...,2]
    gridY  = encode_fixed_grain_quantized(Y,  grain_size=grain_size_Y, levels=levels_Y)
    gridCb = encode_fixed_grain_quantized(Cb, grain_size=grain_size_C, levels=levels_C)
    gridCr = encode_fixed_grain_quantized(Cr, grain_size=grain_size_C, levels=levels_C)
    Y_rec  = decode_fixed_grain_quantized(gridY,  h, w, levels=levels_Y)
    Cb_rec = decode_fixed_grain_quantized(gridCb, h, w, levels=levels_C)
    Cr_rec = decode_fixed_grain_quantized(gridCr, h, w, levels=levels_C)
    return ycbcr_to_rgb(np.stack([Y_rec, Cb_rec, Cr_rec], axis=-1))

def run_websl_chroma_ablation(image_path, webp_q=80, max_side=768):
    img = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    h0, w0 = img.shape[:2]
    print(f"Image chargée : {w0} x {h0}")
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        img = resize_rgb(img, int(round(w0*scale)), int(round(h0*scale)))
        print(f"Image redimensionnée : {img.shape[1]} x {img.shape[0]}")
    h, w = img.shape[:2]

    grain_size_Y = 2
    levels_Y = 32
    chroma_configs = [
        {"grain_size_C": 2, "levels_C": 16},
        {"grain_size_C": 2, "levels_C": 24},
        {"grain_size_C": 2, "levels_C": 32},
        {"grain_size_C": 2, "levels_C": 48},
        {"grain_size_C": 1, "levels_C": 16},
        {"grain_size_C": 1, "levels_C": 24},
        {"grain_size_C": 1, "levels_C": 32},
    ]
    output_scales = [1.0, 0.75, 0.5]

    print("\n" + "=" * 110)
    print("COMPARATIF VISUEL FINAL — ABLATION CHROMA")
    print("=" * 110)
    print(f"{'gC':>5}{'lC':>6}{'sortie':>12}{'direct':>12}{'grain':>12}{'écart':>12}{'PSNR déc.':>12}{'MS-SSIM':>12}")

    for cfg in chroma_configs:
        img_native = build_native_opaque_image(
            img,
            grain_size_Y=grain_size_Y,
            grain_size_C=cfg["grain_size_C"],
            levels_Y=levels_Y,
            levels_C=cfg["levels_C"]
        )
        for out_scale in output_scales:
            out_w = max(1, int(round(w * out_scale)))
            out_h = max(1, int(round(h * out_scale)))
            direct_ref = resize_rgb(img, out_w, out_h)
            grain_proj = resize_rgb(img_native, out_w, out_h)
            direct_path = f"direct_{cfg['grain_size_C']}_{cfg['levels_C']}_{out_w}x{out_h}.webp"
            grain_path  = f"grain_{cfg['grain_size_C']}_{cfg['levels_C']}_{out_w}x{out_h}.webp"
            direct_bytes = save_webp(direct_path, direct_ref, quality=webp_q)
            grain_bytes  = save_webp(grain_path, grain_proj, quality=webp_q)
            direct_dec = load_rgb(direct_path)
            grain_dec  = load_rgb(grain_path)
            print(
                f"{cfg['grain_size_C']:>5d}"
                f"{cfg['levels_C']:>6d}"
                f"{(str(out_w)+'x'+str(out_h)):>12}"
                f"{fmt_kb(direct_bytes):>12}"
                f"{fmt_kb(grain_bytes):>12}"
                f"{(grain_bytes-direct_bytes)/1024:>+11.1f} KB"
                f"{psnr(direct_dec, grain_dec):>12.2f}"
                f"{compute_ms_ssim(direct_dec, grain_dec):>12.4f}"
            )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python websl_chroma_ablation.py <image_path>")
    else:
        run_websl_chroma_ablation(sys.argv[1])
