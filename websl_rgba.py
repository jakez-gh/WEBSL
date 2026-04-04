# ============================================================
# WEBSL — PIPELINE RGBA / DÉTOURÉ
# Auteur : Sam Hassanine
# ============================================================

import os
import math
import zlib
import struct
import numpy as np
import matplotlib.pyplot as plt
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
    return f"{n / 1024:.1f} KB"

def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0

def resize_rgba(img01, new_w, new_h, resample=Image.LANCZOS):
    img8 = (clip01(img01) * 255.0 + 0.5).astype(np.uint8)
    out = Image.fromarray(img8, mode="RGBA").resize((new_w, new_h), resample)
    return np.asarray(out).astype(np.float32) / 255.0

def rgba_to_rgb_on_bg(rgba, bg=(1.0, 1.0, 1.0)):
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    return clip01(rgb * a + bg_arr * (1.0 - a))

def save_webp_rgba(path, rgba01, quality=80):
    rgba8 = (clip01(rgba01) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgba8, mode="RGBA").save(path, format="WEBP", quality=int(quality), method=6)
    return os.path.getsize(path)

def load_decoded_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0

def compute_ms_ssim_rgb(a_rgb, b_rgb):
    ta = torch.from_numpy(a_rgb).permute(2, 0, 1).unsqueeze(0).float()
    tb = torch.from_numpy(b_rgb).permute(2, 0, 1).unsqueeze(0).float()
    return float(ms_ssim(ta, tb, data_range=1.0, size_average=True))

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

def build_native_rgba_grain(rgba, grain_size_rgb=2, grain_size_alpha=2, levels_rgb=32, levels_alpha=8, alpha_threshold=0.001):
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    rgb_pm = rgb * alpha[..., None]

    grid_r = encode_fixed_grain_quantized(rgb_pm[..., 0], grain_size=grain_size_rgb, levels=levels_rgb)
    grid_g = encode_fixed_grain_quantized(rgb_pm[..., 1], grain_size=grain_size_rgb, levels=levels_rgb)
    grid_b = encode_fixed_grain_quantized(rgb_pm[..., 2], grain_size=grain_size_rgb, levels=levels_rgb)
    grid_a = encode_fixed_grain_quantized(alpha, grain_size=grain_size_alpha, levels=levels_alpha)

    r_rec_pm = decode_fixed_grain_quantized(grid_r, h, w, levels=levels_rgb)
    g_rec_pm = decode_fixed_grain_quantized(grid_g, h, w, levels=levels_rgb)
    b_rec_pm = decode_fixed_grain_quantized(grid_b, h, w, levels=levels_rgb)
    a_rec    = decode_fixed_grain_quantized(grid_a, h, w, levels=levels_alpha)

    a_safe = np.maximum(a_rec, alpha_threshold)
    rgb_rec = np.stack([r_rec_pm / a_safe, g_rec_pm / a_safe, b_rec_pm / a_safe], axis=-1)
    rgb_rec = clip01(rgb_rec)
    rgba_rec = np.dstack([rgb_rec, a_rec])
    return rgba_rec, {"R": grid_r, "G": grid_g, "B": grid_b, "A": grid_a}

def zigzag_encode_signed(n):
    return 2 * n if n >= 0 else -2 * n - 1

def varint_encode_unsigned(n):
    out = bytearray()
    n = int(n)
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)

def serialize_raw_grid(grid_q, levels):
    gh, gw = grid_q.shape
    out = bytearray()
    out.extend(b"RAWQ")
    out.extend(struct.pack("<H", gh))
    out.extend(struct.pack("<H", gw))
    out.extend(struct.pack("<H", levels))
    out.extend(grid_q.astype(np.uint8).tobytes())
    return bytes(out)

def predictor_left(grid_hat, y, x):
    if x == 0:
        return 0
    return int(grid_hat[y, x - 1])

def predictor_up(grid_hat, y, x):
    if y == 0:
        return 0
    return int(grid_hat[y - 1, x])

def predictor_avg_lu(grid_hat, y, x):
    if x == 0 and y == 0:
        return 0
    if x == 0:
        return int(grid_hat[y - 1, x])
    if y == 0:
        return int(grid_hat[y, x - 1])
    return int(round((int(grid_hat[y, x - 1]) + int(grid_hat[y - 1, x])) / 2.0))

PREDICTORS = {"left": predictor_left, "up": predictor_up, "avg_lu": predictor_avg_lu}

def serialize_prediction_grid(grid_q, levels, predictor_name):
    gh, gw = grid_q.shape
    predictor = PREDICTORS[predictor_name]
    grid_hat = np.zeros_like(grid_q, dtype=np.int32)
    out = bytearray()
    out.extend(b"PRED")
    out.extend(struct.pack("<H", gh))
    out.extend(struct.pack("<H", gw))
    out.extend(struct.pack("<H", levels))
    out.extend(struct.pack("<B", list(PREDICTORS.keys()).index(predictor_name)))
    for y in range(gh):
        for x in range(gw):
            pred = predictor(grid_hat, y, x)
            true = int(grid_q[y, x])
            res = true - pred
            out.extend(varint_encode_unsigned(zigzag_encode_signed(res)))
            grid_hat[y, x] = true
    return bytes(out)

def zsize(data):
    return len(zlib.compress(data, 9))

def run_websl_rgba(image_path, grain_size_rgb=2, grain_size_alpha=2, levels_rgb=32, levels_alpha=8, output_scales=(1.0,0.75,0.5), webp_q=80, max_side=768):
    rgba = load_rgba(image_path)
    h0, w0 = rgba.shape[:2]
    print(f"Image chargée : {w0} x {h0}")

    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        rgba = resize_rgba(rgba, new_w, new_h)
        print(f"Image redimensionnée : {new_w} x {new_h}")

    rgba_native, grids = build_native_rgba_grain(
        rgba,
        grain_size_rgb=grain_size_rgb,
        grain_size_alpha=grain_size_alpha,
        levels_rgb=levels_rgb,
        levels_alpha=levels_alpha,
        alpha_threshold=1/255
    )

    print("\n" + "=" * 80)
    print("COMPARATIF SUPPORT COMPLET RGBA")
    print("=" * 80)
    total_up = 0
    for cname, grid in grids.items():
        levels = levels_alpha if cname == "A" else levels_rgb
        raw_z = zsize(serialize_raw_grid(grid, levels))
        left_z = zsize(serialize_prediction_grid(grid, levels, "left"))
        up_z = zsize(serialize_prediction_grid(grid, levels, "up"))
        avg_z = zsize(serialize_prediction_grid(grid, levels, "avg_lu"))
        total_up += up_z
        print(f"\nCANAL {cname}")
        print(f"RAW        : {fmt_kb(raw_z)}")
        print(f"PRED left  : {fmt_kb(left_z)}")
        print(f"PRED up    : {fmt_kb(up_z)}")
        print(f"PRED avg   : {fmt_kb(avg_z)}")
    print("\n" + "-" * 40)
    print(f"pred_up {fmt_kb(total_up):>25}")
    print("-" * 40)

    h, w = rgba.shape[:2]
    print("\n" + "=" * 120)
    print("COMPARATIF VISUEL FINAL RGBA")
    print("=" * 120)
    print(f"{'sortie':>12}{'direct':>12}{'grain':>12}{'écart':>12}{'PSNR noir':>12}{'MS noir':>12}{'PSNR blanc':>12}{'MS blanc':>12}{'alpha rmse':>12}")

    for out_scale in output_scales:
        out_w = max(1, int(round(w * out_scale)))
        out_h = max(1, int(round(h * out_scale)))

        direct_ref = resize_rgba(rgba, out_w, out_h)
        grain_proj = resize_rgba(rgba_native, out_w, out_h)

        direct_webp_path = f"direct_{out_w}x{out_h}.webp"
        grain_webp_path = f"grain_{out_w}x{out_h}.webp"

        direct_webp_bytes = save_webp_rgba(direct_webp_path, direct_ref, quality=webp_q)
        grain_webp_bytes = save_webp_rgba(grain_webp_path, grain_proj, quality=webp_q)

        direct_dec = load_decoded_rgba(direct_webp_path)
        grain_dec = load_decoded_rgba(grain_webp_path)

        direct_rgb_black = rgba_to_rgb_on_bg(direct_dec, bg=(0.0, 0.0, 0.0))
        grain_rgb_black  = rgba_to_rgb_on_bg(grain_dec,  bg=(0.0, 0.0, 0.0))
        direct_rgb_white = rgba_to_rgb_on_bg(direct_dec, bg=(1.0, 1.0, 1.0))
        grain_rgb_white  = rgba_to_rgb_on_bg(grain_dec,  bg=(1.0, 1.0, 1.0))

        print(
            f"{(str(out_w)+'x'+str(out_h)):>12}"
            f"{fmt_kb(direct_webp_bytes):>12}"
            f"{fmt_kb(grain_webp_bytes):>12}"
            f"{(grain_webp_bytes-direct_webp_bytes)/1024:>+11.1f} KB"
            f"{psnr(direct_rgb_black, grain_rgb_black):>12.2f}"
            f"{compute_ms_ssim_rgb(direct_rgb_black, grain_rgb_black):>12.4f}"
            f"{psnr(direct_rgb_white, grain_rgb_white):>12.2f}"
            f"{compute_ms_ssim_rgb(direct_rgb_white, grain_rgb_white):>12.4f}"
            f"{rmse(direct_dec[..., 3:4], grain_dec[..., 3:4]):>12.4f}"
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python websl_rgba.py <image_path>")
    else:
        run_websl_rgba(sys.argv[1])
