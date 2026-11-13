# Data

No images are included. The archive is public (figshare, doi 10.6084/m9.figshare.27682557, CC BY 4.0) and must be downloaded by the user; point `PCOS_RAW` at the folder that contains `infected/` and `noninfected/`.

Shipped here (metadata derived from the archive, no pixels):

| File | Content |
|---|---|
| `manifest.csv` | One row per image: relative path, label (1 = folder `infected` = PCOS-positive), file family, width, height, byte size, JPEG quantization value, MD5, mean and standard deviation of intensity, duplicate groups `g0` and `g4`. |
| `splits.csv` | Partition (train / val / test) per image for seeds 0 to 4 under three rules: `grp_s*` (g0 groups), `g4_s*` (g4 groups) and `img_s*` (image level, shown only as a contrast). |
| `weights_manifest.json` | Names, URLs and file names of the pretrained ImageNet weights used by `pipeline/03_download_weights.py`. |

Caveats that matter for any use of these files: the archive has no patient identifiers, so group-disjoint splits are not patient-disjoint; labels come from the folder names and their provenance is undocumented; image width and height alone separate the classes almost perfectly (see the paper).
