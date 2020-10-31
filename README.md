# Unsupervised Outlier Detection by Distillated Teacher-Student Network Ensemble

## Introduction

<img src="https://i.loli.net/2020/07/20/3G7DNKjwfQkiIz4.png" style="zoom: 50%;" />

<img src="https://i.loli.net/2020/07/20/8Ie2Q3mpdPHtrYF.png" style="zoom:50%;" />

<img src="https://i.loli.net/2020/07/20/OEcQSvZmfBz1ACt.png" style="zoom: 67%;" />

## Experimental Results

|         | Batch Size | Feature Dim |
| ------- | ---------- | ----------- |
| Apascal | 128        | 50          |
| Bank    | 256        | 50          |
| Lung    | 16         | 128         |
| Probe   | 256        | 16          |
| Secom   | 32         | 64          |
| U2R     | 256        | 16          |



### ROC-AUC

|         | RDP              | PBS                  | PB                 | BS                 | PS                 | P                  | B                  |
| ------- | ---------------- | -------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Apascal | $0.823 \pm 0.07$ | $0.8327 \pm 0.01612$ | $0.8390\pm 0.0174$ | $0.8251\pm 0.0221$ | $0.8327\pm 0.0161$ | $0.7217\pm 0.0144$ | $0.8303\pm 0.0226$ |
| Bank    | $0.758\pm 0.007$ |                      |                    |                    |                    |                    |                    |
| Lung    | $0.982\pm0.006$  |                      |                    |                    |                    |                    |                    |
| Probe   | $0.997\pm0.000$  |                      |                    |                    |                    |                    |                    |
| Secom   | $0.570\pm 0.004$ |                      |                    |                    |                    |                    |                    |
| U2R     | $0.986\pm 0.001$ |                      |                    |                    |                    |                    |                    |

### PR-AUC

|         | RDP              | PBS                 | PB                 | BS                 | PS                | P                  | B                  |
| ------- | ---------------- | ------------------- | ------------------ | ------------------ | ----------------- | ------------------ | ------------------ |
| Apascal | $0.042\pm 0.003$ | $0.0610 \pm 0.0127$ | $0.0636\pm 0.0151$ | $0.0601\pm 0.0186$ | $0.0610\pm0.0127$ | $0.0274\pm 0.0042$ | $0.0600\pm 0.0182$ |
| Bank    | $0.364\pm 0.013$ |                     |                    |                    |                   |                    |                    |
| Lung    | $0.705\pm 0.028$ |                     |                    |                    |                   |                    |                    |
| Probe   | $0.955\pm 0.002$ |                     |                    |                    |                   |                    |                    |
| Secom   | $0.096\pm 0.001$ |                     |                    |                    |                   |                    |                    |
| U2R     | $0.261\pm0.005$  |                     |                    |                    |                   |                    |                    |