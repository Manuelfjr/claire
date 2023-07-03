import pandas as pd
from sklearn.datasets import make_blobs


class GenerateData:
    def __init__(
        self,
        n_datasets=1,
        n_samples=[300],
        centers=[[(0, 2.5), (0, 1), (2.3, 2.3)]],
        n_features=[2],
        cluster_std=[0.2],
        random_seed=[None],
    ):
        self.n_datasets = n_datasets
        self.n_samples = n_samples
        self.centers = centers
        self.n_features = n_features
        self.cluster_std = cluster_std
        self.random_seed = random_seed
        self.args = {
            i: {
                "samples": self.n_samples[i],
                "centers": self.centers[i],
                "features": self.n_features[i],
                "cluster_std": self.cluster_std[i],
                "random_seed": self.random_seed[i],
            }
            for i in range(self.n_datasets)
        }

    def create_data(self):
        self.datasets = {i: {"X": pd.DataFrame, "Y": pd.DataFrame} for i in self.args}
        for i in self.args:
            X, y = make_blobs(
                n_samples=self.args[i]["samples"],
                centers=self.args[i]["centers"],
                n_features=self.args[i]["features"],
                cluster_std=self.args[i]["cluster_std"],
                random_state=self.args[i]["random_seed"],
            )
            self.datasets[i]["X"] = self.datasets[i]["X"](X)
            self.datasets[i]["Y"] = self.datasets[i]["Y"](y)
        return self.datasets
