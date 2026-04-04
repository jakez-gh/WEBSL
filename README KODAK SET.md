# WEBSL

WEBSL is an image pre-processing pipeline designed to reduce final web image size before standard encoding.

The core idea is simple:

- transform the image into a more compressible representation
- then encode it with a standard codec
- compare the final delivered file against direct codec output

This repository currently focuses on a direct comparison against **WebP**.

## Main result

On the **Kodak 24** benchmark at **full native resolution**, WEBSL beats direct WebP on **23 out of 24 images**.

### Kodak 24 summary

- Benchmark: **Kodak 24**
- Resolution: **full resolution only**
- Baseline: **direct WebP**
- Compared method: **WEBSL → WebP**
- Protocol: **fixed across all images**
- Result: **23 / 24 wins**
- Average size reduction: **~19.6%**
- Total reduction over the dataset: **314.8 KB**
- Average reduction per image: **13.1 KB**

Perceptual fidelity remained high across the benchmark, with **MS-SSIM typically in the 0.95–0.98 range**.

## What WEBSL is

WEBSL is **not** a new final codec format.

It is a **pre-transform pipeline** applied before standard encoding.

In practice:

- baseline path: `image -> WebP`
- WEBSL path: `image -> WEBSL transform -> WebP`

The goal is to make the image easier for the codec to compress while preserving good perceptual quality.

## Current tested setup

For the Kodak benchmark, the dominant preset was:

- `gC = 2`
- `lC = 48`

A few images preferred:

- `gC = 1`
- `lC = 32`

But overall, `gC=2, lC=48` clearly dominated the dataset.

## Kodak 24 — Full-resolution benchmark vs direct WebP

| Image   | Size     | WebP direct | WEBSL→WebP | Gain/Loss | Gain/Loss % | PSNR (dB) | MS-SSIM | Best preset |
|---------|----------|-------------|------------|-----------|-------------|-----------|---------|-------------|
| kodim01 | 768x512  | 86.7 KB     | 52.7 KB    | +34.0 KB  | +39.2%      | 24.53     | 0.9735  | gC=2 lC=48  |
| kodim02 | 768x512  | 39.6 KB     | 32.6 KB    | +6.9 KB   | +17.4%      | 30.38     | 0.9434  | gC=1 lC=32  |
| kodim03 | 768x512  | 29.2 KB     | 27.4 KB    | +1.8 KB   | +6.2%       | 30.57     | 0.9604  | gC=1 lC=32  |
| kodim04 | 512x768  | 42.8 KB     | 37.3 KB    | +5.6 KB   | +13.0%      | 30.43     | 0.9593  | gC=2 lC=48  |
| kodim05 | 768x512  | 89.4 KB     | 70.5 KB    | +18.9 KB  | +21.1%      | 24.34     | 0.9787  | gC=2 lC=48  |
| kodim06 | 768x512  | 65.9 KB     | 46.1 KB    | +19.8 KB  | +30.0%      | 26.02     | 0.9670  | gC=2 lC=48  |
| kodim07 | 768x512  | 36.4 KB     | 36.0 KB    | +0.5 KB   | +1.4%       | 29.32     | 0.9762  | gC=2 lC=32  |
| kodim08 | 768x512  | 92.4 KB     | 62.1 KB    | +30.3 KB  | +32.8%      | 22.25     | 0.9764  | gC=2 lC=48  |
| kodim09 | 512x768  | 30.2 KB     | 28.6 KB    | +1.6 KB   | +5.3%       | 29.43     | 0.9628  | gC=2 lC=48  |
| kodim10 | 512x768  | 35.1 KB     | 33.3 KB    | +1.9 KB   | +5.4%       | 29.51     | 0.9630  | gC=2 lC=48  |
| kodim11 | 768x512  | 57.9 KB     | 40.2 KB    | +17.7 KB  | +30.6%      | 27.33     | 0.9676  | gC=2 lC=48  |
| kodim12 | 768x512  | 34.0 KB     | 29.0 KB    | +4.9 KB   | +14.4%      | 29.90     | 0.9565  | gC=2 lC=48  |
| kodim13 | 768x512  | 119.4 KB    | 72.8 KB    | +46.6 KB  | +39.0%      | 22.83     | 0.9701  | gC=2 lC=48  |
| kodim14 | 768x512  | 75.3 KB     | 57.0 KB    | +18.3 KB  | +24.3%      | 26.96     | 0.9716  | gC=2 lC=48  |
| kodim15 | 768x512  | 38.2 KB     | 32.1 KB    | +6.0 KB   | +15.7%      | 29.62     | 0.9645  | gC=2 lC=48  |
| kodim16 | 768x512  | 45.6 KB     | 33.9 KB    | +11.7 KB  | +25.7%      | 29.25     | 0.9582  | gC=2 lC=48  |
| kodim17 | 512x768  | 40.1 KB     | 35.6 KB    | +4.5 KB   | +11.2%      | 29.97     | 0.9709  | gC=2 lC=48  |
| kodim18 | 512x768  | 75.7 KB     | 55.1 KB    | +20.6 KB  | +27.2%      | 26.45     | 0.9675  | gC=2 lC=48  |
| kodim19 | 512x768  | 52.9 KB     | 40.2 KB    | +12.6 KB  | +23.8%      | 26.60     | 0.9624  | gC=2 lC=48  |
| kodim20 | 768x512  | 32.3 KB     | 25.1 KB    | +7.2 KB   | +22.3%      | 28.35     | 0.9775  | gC=2 lC=48  |
| kodim21 | 768x512  | 54.0 KB     | 41.1 KB    | +12.9 KB  | +23.9%      | 26.75     | 0.9637  | gC=2 lC=48  |
| kodim22 | 768x512  | 58.3 KB     | 44.7 KB    | +13.6 KB  | +23.3%      | 28.60     | 0.9619  | gC=2 lC=48  |
| kodim23 | 768x512  | 26.9 KB     | 29.3 KB    | -2.3 KB   | -8.6%       | 31.11     | 0.9639  | gC=2 lC=48  |
| kodim24 | 768x512  | 71.5 KB     | 52.2 KB    | +19.2 KB  | +26.9%      | 25.56     | 0.9715  | gC=2 lC=48  |

## Interpretation

This result does **not** mean that WEBSL has already beaten every image format or every codec in every condition.

What it does mean is more precise:

- on a standard image benchmark
- at full native resolution
- with a fixed protocol
- against direct WebP

WEBSL shows a strong and repeatable advantage.

That is already a serious engineering result.

## Why this matters

If the encoder is run once offline and the resulting image is served many times, the dominant cost becomes delivery, not preprocessing.

That means a reduction in delivered bytes can matter at scale:

- lower bandwidth
- lower CDN traffic
- lower transfer time
- potentially lower energy cost on repeated delivery

This is especially relevant if WEBSL can approach stronger compression territory while still relying on standard downstream encoding.

## Reproducibility

This repository includes the Python scripts used for the benchmark.

Current files include:

- `websl_opaque.py`
- `websl_rgba.py`
- `websl_support_benchmark.py`
- `websl_chroma_ablation.py`

The Kodak benchmark reported above was run with a fixed full-resolution protocol using the opaque image pipeline.

## Important note

A real counterexample is kept in the benchmark:

- `kodim23` is a loss for WEBSL at full resolution

This is intentional and important.

The goal is not to hide failures, but to show the real behavior of the method.

## Current conclusion

At this stage, the result is clear:

**WEBSL is not a cosmetic trick. It is a real pre-processing pipeline that can beat direct WebP repeatedly on a standard benchmark, with strong average gains and high perceptual fidelity.**

## Next steps

The natural next phase is:

- comparison against AVIF
- comparison against JPEG XL
- fixed-quality and fixed-size protocols
- encoding/decoding cost measurements
- validation on additional datasets
- external reproduction

## License

https://zenodo.org/records/19394177