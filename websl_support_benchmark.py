# ============================================================
# WEBSL — SUPPORT BENCHMARK
# Auteur : Sam Hassanine
# ============================================================

import math
import zlib
import struct
import numpy as np
from PIL import Image

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def fmt_kb(n):
    return f"{n/1024:.1f} KB"

def resize_rgb(img01, new_w, new_h, resample=Image.LANCZOS):
    img8 = (clip01(img01) * 255.0 + 0.5).astype(np.uint8)
    out = Image.fromarray(img8).resize((new_w, new_h), resample)
    return np.asarray(out).astype(np.float32) / 255.0

def rgb_to_ycbcr(img):
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]
    Y  = 0.299000 * R + 0.587000 * G + 0.114000 * B
    Cb = 0.5 + (-0.168736 * R - 0.331264 * G + 0.500000 * B)
    Cr = 0.5 + ( 0.500000 * R - 0.418688 * G - 0.081312 * B)
    return clip01(np.stack([Y, Cb, Cr], axis=-1))

def quantize_index(x, levels):
    q = np.floor(clip01(x) * (levels - 1) + 0.5).astype(np.int32)
    return np.clip(q, 0, levels - 1)

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

def run_websl_support_benchmark(image_path, grain_size=2, levels_Y=32, levels_C=16, max_side=768):
    img = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    h0, w0 = img.shape[:2]
    print(f"Image chargée : {w0} x {h0}")
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        img = resize_rgb(img, int(round(w0*scale)), int(round(h0*scale)))
        print(f"Image redimensionnée : {img.shape[1]} x {img.shape[0]}")
    ycbcr = rgb_to_ycbcr(img)
    grids = {
        "Y": encode_fixed_grain_quantized(ycbcr[...,0], grain_size=grain_size, levels=levels_Y),
        "Cb": encode_fixed_grain_quantized(ycbcr[...,1], grain_size=grain_size, levels=levels_C),
        "Cr": encode_fixed_grain_quantized(ycbcr[...,2], grain_size=grain_size, levels=levels_C),
    }

    print("\n" + "=" * 88)
    print("COMPARATIF SUPPORT COMPLET (géométrie canonique ligne par ligne)")
    print("=" * 88)
    total = {k:0 for k in ["raw","pred_left","pred_up","pred_avg_lu"]}
    for cname, grid in grids.items():
        levels = levels_Y if cname == "Y" else levels_C
        sizes = {
            "raw": zsize(serialize_raw_grid(grid, levels)),
            "pred_left": zsize(serialize_prediction_grid(grid, levels, "left")),
            "pred_up": zsize(serialize_prediction_grid(grid, levels, "up")),
            "pred_avg_lu": zsize(serialize_prediction_grid(grid, levels, "avg_lu")),
        }
        print(f"\nCANAL {cname}")
        for k, v in sizes.items():
            print(f"{k:<12}: {fmt_kb(v)}")
            total[k] += v
    print("\n" + "-" * 40)
    for k, v in sorted(total.items(), key=lambda kv: kv[1]):
        print(f"{k:<12} {fmt_kb(v):>12}")
    print("-" * 40)
    print(f"Meilleur mode support : {min(total.items(), key=lambda kv: kv[1])[0]}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python websl_support_benchmark.py <image_path>")
    else:
        run_websl_support_benchmark(sys.argv[1])
