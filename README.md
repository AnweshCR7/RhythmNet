# RhythmNet: End-to-end Heart Rate Estimation from Face via Spatial-temporal Representation
A reproduction of the RhythmNet model. [Paper link](arxiv.org/abs/1910.11515)

#### Dataset:
VIPL-HR dataset

## Experiments
Shared parameters:
```
batch size: 32
Dataset: VIPL
Model: RhythmNet
initial learning rate: 1e-3
epochs: 50
window_size = 300 frames with stride of 0.5 seconds
```

**Dataset-split**: 5 fold validation
### Experiment for 1-Fold without GRU layer

| Set      |  Loss | MAE  (bpm) | RMSE (bpm) |
|----------|:-----:|:----------:|:----------:|
| Training | 3.096 |    1.817   |    2.834   |
| Eval     | 15.91 |    9.255   |   11.787   |

### Experiment for 1-Fold with GRU layer
| Set      |  Loss | MAE  (bpm) | RMSE (bpm) |
|----------|:-----:|:----------:|:----------:|
| Training | 3.925 |    2.423   |    4.16    |
| Eval     | 14.25 |   13.992   |   17.019   |

