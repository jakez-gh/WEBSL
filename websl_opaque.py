# ============================================================
# WEBSL — PIPELINE OPAQUE
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

# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================
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

def zigzag_encode_signed(n: int) -> int:
    return 2 * n if n >= 0 else -2 * n - 1

def varint_encode_unsigned(n: int) -> bytes:
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

# ============================================================
# RGB <-> YCbCr
# ============================================================
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

# ============================================================
# GRAIN FIXE
# ============================================================
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
            v = float(np.mean(patch))
            grid_q[gy, gx] = int(quantize_index(np.array([v]), levels)[0])

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

# ============================================================
# SUPPORT OPAQUE NATIF
# ============================================================
def build_native_opaque_image(img, grain_size_Y=2, grain_size_C=2, levels_Y=32, levels_C=16):
    h, w = img.shape[:2]
    ycbcr = rgb_to_ycbcr(img)
    Y  = ycbcr[..., 0]
    Cb = ycbcr[..., 1]
    Cr = ycbcr[..., 2]

    gridY  = encode_fixed_grain_quantized(Y,  grain_size=grain_size_Y, levels=levels_Y)
    gridCb = encode_fixed_grain_quantized(Cb, grain_size=grain_size_C, levels=levels_C)
    gridCr = encode_fixed_grain_quantized(Cr, grain_size=grain_size_C, levels=levels_C)

    Y_rec  = decode_fixed_grain_quantized(gridY,  h, w, levels=levels_Y)
    Cb_rec = decode_fixed_grain_quantized(gridCb, h, w, levels=levels_C)
    Cr_rec = decode_fixed_grain_quantized(gridCr, h, w, levels=levels_C)

    img_native = ycbcr_to_rgb(np.stack([Y_rec, Cb_rec, Cr_rec], axis=-1))
    return img_native, {"Y": gridY, "Cb": gridCb, "Cr": gridCr}

# ============================================================
# CODAGE SUPPORT
# ============================================================
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

PREDICTORS = {
    "left": predictor_left,
    "up": predictor_up,
    "avg_lu": predictor_avg_lu,
}

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

# ============================================================
# SCRIPT PRINCIPAL
# ============================================================
def run_websl_opaque(image_path, grain_size_Y=2, grain_size_C=2, levels_Y=32, levels_C=16, output_scales=(1.0,0.75,0.5), webp_q=80, max_side=768):
    img = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
    h0, w0 = img.shape[:2]
    print(f"Image chargée : {w0} x {h0}")

    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        new_w = int(round(w0 * scale))
        new_h = int(round(h0 * scale))
        img = resize_rgb(img, new_w, new_h)
        print(f"Image redimensionnée : {new_w} x {new_h}")

    h, w = img.shape[:2]

    img_native, grids = build_native_opaque_image(
        img,
        grain_size_Y=grain_size_Y,
        grain_size_C=grain_size_C,
        levels_Y=levels_Y,
        levels_C=levels_C
    )

    print("\n" + "=" * 88)
    print("COMPARATIF SUPPORT COMPLET (géométrie canonique ligne par ligne)")
    print("=" * 88)

    support_results = []
    for cname in ["Y", "Cb", "Cr"]:
        grid = grids[cname]
        levels = levels_Y if cname == "Y" else levels_C

        raw_z = zsize(serialize_raw_grid(grid, levels))
        pred_left_z = zsize(serialize_prediction_grid(grid, levels, "left"))
        pred_up_z = zsize(serialize_prediction_grid(grid, levels, "up"))
        pred_avg_z = zsize(serialize_prediction_grid(grid, levels, "avg_lu"))

        best = min(
            [("raw", raw_z), ("pred_left", pred_left_z), ("pred_up", pred_up_z), ("pred_avg_lu", pred_avg_z)],
            key=lambda x: x[1]
        )

        print(f"\nCANAL {cname}")
        print(f"RAW        : {fmt_kb(raw_z)}")
        print(f"PRED left  : {fmt_kb(pred_left_z)}")
        print(f"PRED up    : {fmt_kb(pred_up_z)}")
        print(f"PRED avg   : {fmt_kb(pred_avg_z)}")
        print(f"Meilleur   : {best[0]}")
        support_results.append((cname, best[0], best[1]))

    print("\n" + "=" * 118)
    print("COMPARATIF VISUEL FINAL")
    print("=" * 118)
    print(f"{'sortie':>12}{'webp direct':>14}{'webp grain':>14}{'écart':>12}{'PSNR décodés':>14}{'MS-SSIM déc.':>16}")

    for out_scale in output_scales:
        out_w = max(1, int(round(w * out_scale)))
        out_h = max(1, int(round(h * out_scale)))

        direct_ref = resize_rgb(img, out_w, out_h)
        grain_proj = resize_rgb(img_native, out_w, out_h)

        direct_path = f"direct_{out_w}x{out_h}.webp"
        grain_path  = f"grain_{out_w}x{out_h}.webp"

        direct_bytes = save_webp(direct_path, direct_ref, quality=webp_q)
        grain_bytes  = save_webp(grain_path, grain_proj, quality=webp_q)

        direct_dec = load_rgb(direct_path)
        grain_dec  = load_rgb(grain_path)

        print(
            f"{(str(out_w)+'x'+str(out_h)):>12}"
            f"{fmt_kb(direct_bytes):>14}"
            f"{fmt_kb(grain_bytes):>14}"
            f"{(grain_bytes-direct_bytes)/1024:>+11.1f} KB"
            f"{psnr(direct_dec, grain_dec):>14.2f}"
            f"{compute_ms_ssim(direct_dec, grain_dec):>16.4f}"
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python websl_opaque.py <image_path>")
    else:
        run_websl_opaque(sys.argv[1])
