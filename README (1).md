# Gloss-Free Sign Language Translation with a Keypoint Stream

Extending **GFSLT-VLP** ([Zhou et al., ICCV 2023](https://github.com/zhoubenjia/GFSLT-VLP)) with a MediaPipe hand, face and body keypoint stream.
DLCV course project · Team **Claude5**: Gundapuneni Divija (230121025), Devansh Singh (230106021), Konica Maibam (266102015), Priyanshu Kumar (230103080), Hemakshi Kumar (230108020)

**Result:** adding keypoints raises test BLEU-4 on PHOENIX14T from **11.48 to 15.06 (+3.58)** over our reproduced gloss-free baseline, and from 12.15 to 14.89 (+2.74) over our reproduced GFSLT-VLP, with no extra translation latency.

## Idea

GFSLT-VLP translates sign language videos into sentences without glosses (sign-by-sign annotations), using only RGB frames.
Sign meaning lives in small details (finger shapes, hand positions, mouth and eyebrows) that a CNN sees only coarsely at 224×224.
We add a second input stream: 75 keypoints per frame (11 upper-body, 2×21 hand, 22 face) from MediaPipe,
normalised by shoulder position and width, embedded by a small MLP (225→256) and concatenated with the ResNet18 frame features before the temporal blocks.
Everything else (VLP pretraining, decoder, losses) follows the paper. The keypoints come from an off-the-shelf detector, so the method stays gloss-free.

## Results (PHOENIX14T, gloss-free, one seed each)

| Model | Keypoints | VLP | Dev BLEU-4 | Test BLEU-1 | Test BLEU-4 | Test ROUGE-L | Params (M) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| E1 GFSLT (reproduced) | – | – | 11.26 | 28.62 | 11.48 | 28.59 | 26.44 | 78.0 |
| E3 GFSLT-VLP (reproduced) | – | ✓ | 11.52 | 30.57 | 12.15 | 29.90 | 26.44 | 77.9 |
| **E2 GFSLT + KP (ours)** | ✓ | – | **13.96** | 33.99 | **15.06** | 33.62 | 27.22 | 76.9 |
| **E4 GFSLT-VLP + KP (ours)** | ✓ | ✓ | 13.79 | **34.43** | 14.89 | **33.93** | 27.22 | 76.9 |
| *GFSLT-VLP, paper (full fine-tuning)* | – | ✓ | *22.12* | *43.71* | *21.44* | *42.49* | – | – |

Params = trainable translation model (the frozen ResNet18, 11.18M, is shared by all rows).
Latency = mean per test video on one T4 from pre-extracted features (visual encoder + beam-5 decoding).

Our absolute scores are below the paper's because, to fit a free Kaggle GPU, ResNet18 is frozen and features are pre-extracted, the decoder is trained without mBART, and image augmentation is not possible. These limits apply equally to all four models, so the comparison between them is fair. See `report/report.pdf` for details.

## How to run (Kaggle, free tier)

Each script is one Kaggle notebook cell. Paste it into a new notebook, set the accelerator and inputs below, then **Save & Run All**.

| Step | Script | Accelerator | Inputs to attach | Time | Output |
|---|---|---|---|---|---|
| 1 | `code/01_resnet_features.py` | GPU T4 | none (streams the official dataset) | ~45 min | ResNet18 features per frame |
| 2 | `code/02_keypoints_mediapipe.py` | None (CPU) | none (streams the official dataset) | ~6 h | 75 keypoints per frame |
| 3 | `code/03_train_part1_no_keypoints.py` | GPU T4 | step 1 | ~1.8 h | E1, E3 |
| 4 | `code/04_train_part2_with_keypoints.py` | GPU T4 | steps 1, 2, 3 | ~2 h | E2, E4 + final table |
| 5 | `code/05_measure_latency.py` | GPU T4 | steps 1, 2, 3, 4 | ~5 min | params, latency, BLEU re-check |

All notebooks need **Internet: On**. Steps 1 and 2 can run at the same time.
Total compute: about 4.5 GPU hours and 6 CPU hours.

## Data

RWTH-PHOENIX-Weather 2014T (official split 7096 / 519 / 642 videos), from RWTH Aachen University: [phoenix-2014-T.v3.tar.gz](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz). The scripts stream it directly from this archive; it is not included in this repository. Glosses are never used.

## Repository contents

```
code/      the five Kaggle scripts, in run order
results/   results table, training curves and translations from our runs
report/    course report (LaTeX source and PDF)
```

## Credits

- B. Zhou et al., *Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining*, ICCV 2023. Official code: https://github.com/zhoubenjia/GFSLT-VLP. Our implementation is written from the paper.
- N. C. Camgoz et al., *Neural Sign Language Translation*, CVPR 2018 (PHOENIX14T dataset).
- MediaPipe (Google) for keypoint extraction.
