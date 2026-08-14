# Methodology

## Question

This study asks when GPU acceleration becomes worthwhile for classical machine
learning after data loading, preprocessing, transfers, training, inference,
and reuse are counted. Estimator and end-to-end conclusions remain separate.

## Hardware and environment

- NVIDIA RTX PRO 4000 Blackwell SFF Edition, 24 GB VRAM.
- CPU comparisons pinned to eight logical CPUs.
- Ubuntu 26.04 host; digest-pinned Ubuntu 24.04 benchmark container.
- Python 3.12, RAPIDS 26.08, CUDA 13.1, scikit-learn 1.9, XGBoost 3.3,
  UMAP 0.5.12, and HDBSCAN 0.8.44.

Every condition runs in a fresh process. A matrix identity binds its complete
configuration to the source revision. Publication conditions use one
unmeasured warmup, three measured trials, and three independent replications.
CPU libraries receive eight threads. GPU work is synchronized at timed
boundaries. Input files are read before the load timer, so loading is a warm
operating-system-cache measurement rather than an NVMe benchmark.

## Estimator matrix

The estimator study compares scikit-learn or the corresponding CPU library,
the same supported API under `cuml.accel`, and native cuML with explicitly
GPU-resident input. XGBoost has CPU and native CUDA paths only. Tested
algorithms are PCA, KMeans, Logistic Regression, Random Forest, UMAP, HDBSCAN,
and XGBoost. Row count and feature count are independent sweep axes.
Configured row counts include both splits: 80% train and 20% held-out test.

Each result separates input loading, explicit transfer, fit, inference by
batch size, and output materialization. It also records host RSS, sampled peak
VRAM and utilization, and an energy estimate. Random Forest uses 100 trees,
depth 16, and `max_features="sqrt"` on both devices.

## Pipeline residency matrix

The measured pipeline is:

```text
load -> standardize -> nonlinear features -> PCA -> logistic regression
     -> probability inference -> host materialization
```

Five topologies expose placement costs: all CPU; unchanged APIs under
`cuml.accel`; fully GPU-resident after one upload; CPU preprocessing followed
by one GPU transition; and an intentional CPU island that forces repeated
host/device movement. A trial starts before loading and ends only when
probabilities are host resident.

## Quality gates

Performance is summarized only after algorithm-appropriate controls pass.
PCA compares explained variance; KMeans and HDBSCAN use adjusted Rand score;
UMAP uses trustworthiness; classifiers compare ROC AUC, log loss, and
accuracy. Different implementations are not required to produce identical
coefficients, component signs, or cluster labels. A GPU result may improve on
the CPU reference. The one-sided maximum allowed drops in the primary metric
are 0.03 for PCA explained variance, 0.12 for KMeans adjusted Rand score, 0.02
for Logistic Regression ROC AUC, 0.03 for Random Forest ROC AUC, 0.05 for UMAP
trustworthiness, 0.15 for HDBSCAN adjusted Rand score, and 0.02 for XGBoost ROC
AUC. Every end-to-end pipeline mode has a 0.02 maximum ROC AUC drop from its
same-replication CPU reference.
`cuml.accel` logs must confirm GPU dispatch; unintended fallback invalidates
the matrix.

## Limits

The data is dense, numeric, synthetic, and warm-cache. This is not evidence
for sparse text, categorical-native models, distributed training,
hyperparameter search, cold object storage, or production serving queues.
Near- or beyond-VRAM execution requires a separate managed-memory or external-
memory study.
