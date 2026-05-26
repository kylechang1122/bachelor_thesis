from typing import Tuple, Literal

import numpy as np
from torch.utils.data import Dataset


# adapted from https://github.com/wernerth94/A-Cross-Domain-Benchmark-for-Active-Learning/blob/898b3281992e468c6175261b2a5f901db8b7c0cb/datasets/sythData.py#L75
def diverging_sine(pool_rng, n_samples=1000, test_ratio=0.8, divergence_factor=0.5, sin_freq=2, cov=0.3):

    x = np.linspace(0, 10, n_samples)
    sin_curve = np.sin(sin_freq*x)

    # Cluster above the curve
    cluster_above_x = x
    cluster_above_y = sin_curve + divergence_factor * x + pool_rng.normal(0, cov, n_samples)
    cluster_above = np.c_[cluster_above_x, cluster_above_y]

    # Cluster below the curve
    cluster_below_x = x
    cluster_below_y = sin_curve - divergence_factor * x + pool_rng.normal(0, cov, n_samples)
    cluster_below = np.c_[cluster_below_x, cluster_below_y]

    x = np.concatenate((cluster_above, cluster_below), axis=0)
    y = np.concatenate((np.ones(len(cluster_above_y)), np.zeros(len(cluster_below_y))), axis=0)

    ids = np.arange(len(x), dtype=int)
    pool_rng.shuffle(ids)
    cut = int(len(ids) * test_ratio)
    train_ids = ids[cut:]
    test_ids = ids[:cut]

    x_train = x[train_ids]
    y_train = y[train_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]

    return x_train, y_train, x_test, y_test

# adapted from https://github.com/wernerth94/A-Cross-Domain-Benchmark-for-Active-Learning/blob/898b3281992e468c6175261b2a5f901db8b7c0cb/datasets/sythData.py#L41
def honey_pot(pool_rng, n_samples=1000, test_ratio=0.8, cov=[[1, 0], [0, 1]] ):

    mean1 = [0, 0]
    cluster1 = pool_rng.multivariate_normal(mean1, cov, n_samples // 4)

    mean2 = [4, 3]
    cluster2 = pool_rng.multivariate_normal(mean2, cov, n_samples // 4)

    mean3 = [0, 6]
    cluster3 = pool_rng.multivariate_normal(mean3, cov, n_samples // 4)

    mean4 = [4, 3]
    cluster4 = pool_rng.multivariate_normal(mean4, cov, n_samples // 4)

    data_pos = np.concatenate((cluster1, cluster2), axis=0)
    data_neg = np.concatenate((cluster3, cluster4), axis=0)

    x = np.concatenate((data_pos, data_neg), axis=0)
    y = np.concatenate((np.ones(len(data_pos)), np.zeros(len(data_neg))), axis=0)

    ids = np.arange(len(x), dtype=int)
    pool_rng.shuffle(ids)
    cut = int(len(ids) * test_ratio)
    train_ids = ids[cut:]
    test_ids = ids[:cut]

    x_train = x[train_ids]
    y_train = y[train_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]

    return x_train, y_train, x_test, y_test

class SyntheticDataset(Dataset):
    def __init__(
            self,
            dataset: Literal["DivergingSine", "HoneyPot"],
            train: bool = True,
            **_
    ):
        super().__init__()
        self.pool_rng = np.random.default_rng(42)
        if dataset == "DivergingSine":
            x_train, y_train, x_test, y_test = diverging_sine(self.pool_rng)
        elif dataset == "HoneyPot":
            x_train, y_train, x_test, y_test = honey_pot(self.pool_rng)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if train:
            self.data = x_train.astype(np.float32)
            self.targets = y_train.astype(np.int32)
        else:
            self.data = x_test.astype(np.float32)
            self.targets = y_test.astype(np.int32)

        self.transform = None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[TensorDict, int]:
        x = self.data[index]
        y = int(self.targets[index])
        return x, y
