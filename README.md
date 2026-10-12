# pneuma

Pneumonia detection from pediatric chest X-rays with DenseNet121. Trained and evaluated on the
[Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(Guangzhou Women and Children's Medical Center, patients aged 1-5).

Demo-grade project. Not a diagnostic device.

## Results (locked test set: 624 images, 234 normal / 390 pneumonia, evaluated once)

Operating point chosen for screening: smallest threshold reaching recall >= 0.95 on validation
(threshold = 0.2928).

| Metric | Value | 95% CI (1000-sample bootstrap) |
|---|---|---|
| Recall (sensitivity) | **0.9872** | [0.9739, 0.9974] |
| Precision | 0.8750 | [0.8438, 0.9077] |
| Specificity | 0.7650 | [0.7118, 0.8214] |
| Accuracy | 0.9038 | [0.8814, 0.9279] |
| F1 | 0.9277 | [0.9097, 0.9469] |
| F2 | 0.9625 | - |
| ROC-AUC | 0.9744 | [0.9600, 0.9867] |
| PR-AUC | 0.9790 | - |

Confusion matrix at this threshold: TP 385, FN 5, FP 55, TN 179.

Other thresholds on the same test set:

| Threshold | Recall | Precision | Specificity | Accuracy | F2 |
|---|---|---|---|---|---|
| 0.2928 (recall-first) | 0.9872 | 0.8750 | 0.7650 | 0.9038 | 0.9625 |
| 0.1247 (max F2) | 0.9923 | 0.8395 | 0.6838 | 0.8766 | 0.9574 |
| 0.5 (default) | 0.9718 | 0.9199 | 0.8590 | 0.9295 | 0.9610 |

Validation (10% stratified carve of train, n=522): AUROC 0.9990, AUPRC 0.9996. The val-test gap
(0.999 vs 0.974 AUROC) is real - validation came from the training pool's distribution, the test
set is harder. Trust the test numbers.

## Model and training

- DenseNet121 (timm, ImageNet pretrained), 7.0M params, 224x224, single logit (BCE).
- Phase 1: 5 epochs, head only (1,025 trainable params), Adam 1e-3.
- Phase 2: full unfreeze, AdamW 1e-4, cosine decay to 1e-6, early stop on val AUROC (patience 4 -
  stopped at epoch 10/15, best val AUROC 0.9990), label smoothing 0.05, pos_weight 0.329, batch 32, AMP.
- Test set locked until a single final evaluation. Threshold tuned on validation only.
- Grad-CAM off the last denseblock conv for interpretability.

## False negative / false positive review

At the recall-first operating point: 5 false negatives, 55 false positives.
Galleries with Grad-CAM in `figures/` (gradcam_FN.png, gradcam_FP.png, plus TP/TN and a 4x4 grid),
per-image scores in `test_predictions.csv`.

**The 5 misses (FN)** - all very low confidence, not borderline:
person154_bacteria (p=0.06), person153_bacteria (p=0.10), person16_virus (p=0.12),
person1653_virus (p=0.28), person88_bacteria (p=0.28). Grad-CAM on the misses shows diffuse,
non-focal heat spread over the upper/mid chest instead of locking onto a consolidated region -
the model simply finds no convincing pneumonia pattern in these films. They look like subtle or
atypical presentations. The max-F2 threshold (0.1247) would catch 2 of the 5 (FN=3) at the cost of
19 more false alarms.

**The 55 false alarms (FP)** - 23.5% of normal test films get flagged at the screening threshold.
The worst are confident (up to p=0.97), and their Grad-CAM heatmaps sit on real haze/opacity in the
lower lung and retrocardiac zones - regions that genuinely look infiltrate-like. Two honest reads:
(1) this dataset is known to carry label noise, so a slice of "NORMAL" films with real findings is
arguably mislabeled ground truth (the FP review doubles as a label-noise audit); (2) even so,
specificity 0.765 is the honest price of recall 0.987 at this threshold. If false alarms matter
more than misses, the 0.5 threshold gives specificity 0.859 / recall 0.972.

## Limitations

- Pediatric (ages 1-5), single-center dataset. No adult data, no external-site validation.
- Physician-graded but noisy labels (see FP review).
- Validation and test distributions differ (val AUROC 0.999 vs test 0.974).
- Demo-grade research artifact. Not validated for clinical use, not a diagnostic device.

## Files

- `weights/densenet121_pneumonia_best.pth` - best checkpoint (state_dict + config, ~28 MB)
- `notebooks/pneumonia_densenet121.ipynb` - full training/eval notebook (runs on Kaggle, T4)
- `figures/` - training curves, ROC/PR/confusion, Grad-CAM TP/TN/FP/FN galleries
- `metrics.json` - all test metrics + bootstrap CIs
- `test_predictions.csv` - per-image path, actual, pred, p(pneumonia), outcome for all 624 test images

## Inference

```python
import torch, timm
ckpt = torch.load('weights/densenet121_pneumonia_best.pth', map_location='cpu')
model = timm.create_model('densenet121', pretrained=False, num_classes=1)
model.load_state_dict(ckpt['state_dict']); model.eval()
# preprocess: resize 224, ImageNet normalize; p = torch.sigmoid(model(x))
# screening decision at p >= 0.2928
```

Trained on Kaggle (T4 x2, ~15 min wall). Reproduce: run the notebook on the dataset linked above.
