# Representation Learning for High-dimensional Categorical Dataset

## Introduction





## Experimental Results

### Apascal

conducted with 200 epochs and 10 individual runs.

| Criterion / Improvements | Ori                             | Neither                         | Pairwise Loss                   | Momentum Updating | Both              |
| ------------------------ | ------------------------------- | ------------------------------- | ------------------------------- | ----------------- | ----------------- |
| Sample Distance          | /                               | AUROC: 0.8449<br />AUPR: 0.0618 | AUROC:<br />AUPR:               | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| LOF                      | AUROC: 0.5093<br />AUPR: 0.0137 | AUROC: 0.4983<br />AUPR: 0.0130 | AUROC:<br />AUPR:               | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| Isolation Forest         | AUROC: 0.5021<br />AUPR: 0.0136 | AUROC: 0.5464<br />AUPR: 0.0161 | AUROC: 0.6501<br />AUPR: 0.0287 | AUROC:<br />AUPR: | AUROC:<br />AUPR: |

### Secom



| Criterion / Improvements | Ori                | Neither                         | Pairwise Loss      | Momentum Updating | Both              |
| ------------------------ | ------------------ | ------------------------------- | ------------------ | ----------------- | ----------------- |
| Sample Distance          | /                  | AUROC: 0.5450<br />AUPR: 0.0933 | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| LOF                      | AUROC: <br />AUPR: | AUROC:<br />AUPR:               | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| Isolation Forest         | AUROC: <br />AUPR: | AUROC: <br />AUPR:              | AUROC: <br />AUPR: | AUROC:<br />AUPR: | AUROC:<br />AUPR: |

### Lung



| Criterion / Improvements | Ori                | Neither                         | Pairwise Loss      | Momentum Updating | Both              |
| ------------------------ | ------------------ | ------------------------------- | ------------------ | ----------------- | ----------------- |
| Sample Distance          | /                  | AUROC: 0.8621<br />AUPR: 0.4672 | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| LOF                      | AUROC: <br />AUPR: | AUROC:<br />AUPR:               | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| Isolation Forest         | AUROC: <br />AUPR: | AUROC: <br />AUPR:              | AUROC: <br />AUPR: | AUROC:<br />AUPR: | AUROC:<br />AUPR: |

### Bank



| Criterion / Improvements | Ori                | Neither                         | Pairwise Loss      | Momentum Updating | Both              |
| ------------------------ | ------------------ | ------------------------------- | ------------------ | ----------------- | ----------------- |
| Sample Distance          | /                  | AUROC: 0.7407<br />AUPR: 0.3159 | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| LOF                      | AUROC: <br />AUPR: | AUROC:<br />AUPR:               | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| Isolation Forest         | AUROC: <br />AUPR: | AUROC: <br />AUPR:              | AUROC: <br />AUPR: | AUROC:<br />AUPR: | AUROC:<br />AUPR: |

### Probe



| Criterion / Improvements | Ori                | Neither            | Pairwise Loss      | Momentum Updating | Both              |
| ------------------------ | ------------------ | ------------------ | ------------------ | ----------------- | ----------------- |
| Sample Distance          | /                  | AUROC: <br />AUPR: | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| LOF                      | AUROC: <br />AUPR: | AUROC:<br />AUPR:  | AUROC:<br />AUPR:  | AUROC:<br />AUPR: | AUROC:<br />AUPR: |
| Isolation Forest         | AUROC: <br />AUPR: | AUROC: <br />AUPR: | AUROC: <br />AUPR: | AUROC:<br />AUPR: | AUROC:<br />AUPR: |