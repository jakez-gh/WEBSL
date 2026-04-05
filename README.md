# WEBSL — Grain-Support Image Representation for the Web

**Author:** Sam Hassanine  
**Date:** April 3, 2026  
**License:** Open source  
**Zenodo DOI:** [10.5281/zenodo.19394177](https://zenodo.org/records/19394177)

---

## What is WEBSL?

WEBSL is a minimal image representation that describes an image not as a final pixel grid, but as a **regular grid of grains carrying state**, read line by line, then projected to any output size.

The core is deliberately simple:

- Fixed grain support with implicit geometry
- Canonical line-by-line reading order
- Basic predictor (pred_up or pred_avg_lu)
- Uniform quantization per channel
- Deflate (zlib) compression on top
- Final export via standard web format (WebP)

**No ANS. No heavy transform. No deep model. No specialized bitstream.**

Despite this simplicity, WEBSL beats direct WebP in file size **~70% of the time** across tested cases.

---

## Key Results

### ![Opaque Image](Opaque%20Image%20(Times%20Square%2C%20urban%20scene).png)
| Output     | WebP direct | WEBSL grain | Difference |
|------------|-------------|-------------|------------|
| 768 × 510  | 80.3 KB     | 71.5 KB     | **−8.9 KB** |
| 576 × 382  | 55.7 KB     | 50.7 KB     | **−5.0 KB** |
| 384 × 255  | 30.7 KB     | 28.6 KB     | **−2.1 KB** |

MS-SSIM remains above 0.97 at all sizes.

### ![RGBA Cutout](RGBA%20cutout%20image%20(native%20alpha).png)
| Output     | WebP direct | WEBSL grain | Difference | MS-SSIM |
|------------|-------------|-------------|------------|---------|
| 646 × 386  | 22.4 KB     | 15.4 KB     | **−7.1 KB** | 0.986   |
| 484 × 290  | 15.7 KB     | 14.8 KB     | **−0.9 KB** | 0.989   |
| 323 × 193  | 9.1 KB      | 7.9 KB      | **−1.2 KB** | 0.993   |

RGBA cutout images are WEBSL's strongest terrain.

---

## How It Works

1. **Encode:** The source image is converted to YCbCr (opaque) or premultiplied RGBA. Each channel is divided into a grid of fixed-size grains. Each grain stores the quantized average of its patch.
2. **Compress the support:** The grain grid is serialized line by line using a predictor (pred_up, pred_left, or pred_avg_lu) + zigzag + varint encoding, then deflated.
3. **Project:** At decode time, the grain grid is projected (nearest-neighbor) to whatever output resolution is needed.
4. **Export:** The projected image is saved as WebP for final delivery.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `websl_opaque.py` | Full pipeline for opaque (RGB) images |
| `websl_rgba.py` | Full pipeline for RGBA / cutout images |
| `websl_chroma_ablation.py` | Chroma parameter ablation study |
| `websl_support_benchmark.py` | Support encoding comparison (raw, pred_left, pred_up, pred_avg_lu) |

### Requirements

```
numpy
pillow
pytorch-msssim
torch
```

### Usage

```bash
python websl_opaque.py <image_path>
python websl_rgba.py <image_path>
python websl_chroma_ablation.py <image_path>
python websl_support_benchmark.py <image_path>
```

---

## Strengths

- Beats WebP direct ~70% of the time
- Extremely simple core — no specialized codec
- Particularly strong on reduced outputs and RGBA cutouts
- Reproducible: all code is public

## Known Limitations

- Chroma can drift on skin tones and sensitive hues (portraits)
- Not always ahead at full resolution on every image type
- Currently uses WebP as final delivery format

---

## Why Open Source?

The system is simple enough to reverse-engineer. Publishing it openly establishes priority, enables verification, and provides a clean foundation for anyone who wants to extend it.

The Zenodo deposit provides a dated, signed, DOI-referenced archive for prior art.

---

## Citation

If you use or reference WEBSL:

```
Sam Hassanine. "WEBSL — Support-grain web pour images 2D." Zenodo, April 3, 2026.
https://zenodo.org/records/19394177
```
