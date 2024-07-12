import os
import torch
import zarr
from typing import Iterator, Sequence
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Sampler
from lightning.pytorch import LightningDataModule
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)
# TODO: change this to your own path
WORKSPACE_PATH = Path(os.environ["PYTHONPATH"]).absolute()
DATASET_PATH = WORKSPACE_PATH / "dataset" / "20150423_90"


class SubsetSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in range(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


class MovieDatasetV2(Dataset):
    def __init__(
        self,
        path,
        movie_name="MOVIE01",
        keys=["spike_avg", "stimuli"],
        random_window_size=700,
        **kwargs,
    ):
        assert movie_name in [
            "MOVIE01",
            "MOVIE03",
        ], "movie_name must be MOVIE01 or MOVIE03"
        assert "spike_avg" in keys, "spike_avg must be in keys"

        # check if zarr data has been generated
        self.random_window_size = random_window_size
        self.parent_folder = path
        self.movie_path = self.parent_folder / f"{movie_name.upper()}_zarr"
        if not os.path.exists(self.movie_path):
            raise FileNotFoundError(
                f"Cannot find the zarr dataset, please run the migration.py script first."
            )

        movie_zarr = zarr.open(self.movie_path, mode="r")
        self.rf_to_dinov2 = torch.tensor(movie_zarr["rf_to_dinov2"][:])
        self.dataset = dict()
        for key in keys:
            self.dataset[key] = torch.tensor(movie_zarr[key][:])

        self.train_idx = movie_zarr.attrs["split_train"]
        self.test_idx = movie_zarr.attrs["split_test"]

    @property
    def attrs(self):
        return zarr.open(self.movie_path, mode="r").attrs

    def __len__(self):
        # For single mov, using only half of the movie
        return len(self.attrs["split_train"])

    def __getitem__(self, index):
        if index in self.train_idx:  # train
            index = self.train_idx.index(index)
            retrievals = self.train_idx[index : index + self.random_window_size]
            ret = {key: self.dataset[key][retrievals] for key in self.dataset.keys()}
            ret["rf_to_dinov2"] = self.rf_to_dinov2
            return ret
        if index == -1:  # test
            test_idx = self.attrs["split_test"]
            ret = {key: self.dataset[key][test_idx] for key in self.dataset.keys()}
            ret["rf_to_dinov2"] = self.rf_to_dinov2
            return ret
        raise IndexError("Index must be in split_train or split_test")


class MovieCrossValDatasetV2(Dataset):
    def __init__(
        self,
        path,
        movie_name_for_train="MOVIE01",
        keys=["spike_avg", "stimuli"],
        random_window=True,
        random_window_size=700,
    ):
        assert movie_name_for_train in [
            "MOVIE01",
            "MOVIE03",
        ], "movie_name must be MOVIE01 or MOVIE03"
        assert "spike_avg" in keys, "spike_avg must be in keys"
        movie_name_for_train = (
            "MOVIE01" if movie_name_for_train == "MOVIE01" else "MOVIE03"
        )
        movie_name_for_test = (
            "MOVIE03" if movie_name_for_train == "MOVIE01" else "MOVIE01"
        )
        log.info(f"movie_name_for_train: {movie_name_for_train}")
        log.info(f"movie_name_for_test: {movie_name_for_test}")

        # check if zarr data has been generated
        self.random_window = random_window
        self.random_window_size = random_window_size
        self.parent_folder = path
        self.movie_path_for_train = (
            self.parent_folder / f"{movie_name_for_train.upper()}_zarr"
        )
        self.movie_path_for_test = (
            self.parent_folder / f"{movie_name_for_test.upper()}_zarr"
        )

        assert os.path.exists(
            self.movie_path_for_train
        ), f"Cannot find the zarr dataset, please run the migration.py script first."
        assert os.path.exists(
            self.movie_path_for_test
        ), f"Cannot find the zarr dataset, please run the migration.py script first."

        movie_zarr_for_train = zarr.open(self.movie_path_for_train, mode="r")
        movie_zarr_for_test = zarr.open(self.movie_path_for_test, mode="r")
        self.rf_to_dinov2 = torch.tensor(movie_zarr_for_train["rf_to_dinov2"][:])
        self.dataset_for_train = dict()
        for key in keys:
            self.dataset_for_train[key] = torch.tensor(movie_zarr_for_train[key][:])
        self.dataset_for_test = dict()
        for key in keys:
            self.dataset_for_test[key] = torch.tensor(movie_zarr_for_test[key][:])
        self.idxs = [
            self.attrs(train=True)["split_train"],
            self.attrs(train=True)["split_test"],
            self.attrs(train=False)["split_train"],
            self.attrs(train=False)["split_test"],
        ]

    def attrs(self, train=True):
        p = self.movie_path_for_train if train else self.movie_path_for_test
        return zarr.open(p, mode="r").attrs

    def __len__(self):
        # For multi-mov, using only 2 half of the movie
        # only provide train, cause test is only 2 batches
        return len(self.idxs[0]) + len(self.idxs[1])

    def __getitem__(self, index):
        # calc retrievals
        retrievals = []
        if (
            index < len(self.idxs[0]) - self.random_window_size and index >= 0
        ):  # train 1
            retrievals = self.idxs[0][index : index + self.random_window_size]
            ds = self.dataset_for_train
        elif (
            index < len(self.idxs[0]) + len(self.idxs[1]) - 2 * self.random_window_size
            and index >= 0
        ):  # train 2
            index = index - len(self.idxs[0]) + self.random_window_size
            retrievals = self.idxs[1][index : index + self.random_window_size]
            ds = self.dataset_for_train
        elif index == -1:  # test 1
            retrievals = self.idxs[2]
            ds = self.dataset_for_test
        elif index == -2:  # test 2
            retrievals = self.idxs[3]
            ds = self.dataset_for_test
        else:
            raise IndexError("Index must be in split_train or split_test")

        ret = {key: ds[key][retrievals] for key in ds.keys()}
        ret["rf_to_dinov2"] = self.rf_to_dinov2
        return ret


class TaskV2(LightningDataModule):
    def __init__(
        self,
        movie_name,
        parent_folder=DATASET_PATH,
        keys=["spike_avg", "stimuli"],
        batch_size=1,
        prefetch_batches=3,
        cross_val_movie=False,
        random_window=True,
        random_window_size=64,  # 700 60
        *args,
        **kwargs,
    ):
        super().__init__()
        movie_name = movie_name.upper()
        assert movie_name in ["MOVIE01", "MOVIE03"]
        parent_folder = Path(parent_folder)
        self.random_window_size = random_window_size
        if cross_val_movie:
            self.dataset = MovieCrossValDatasetV2(
                parent_folder,
                movie_name_for_train=movie_name,
                keys=keys,
                random_window=random_window,
                random_window_size=random_window_size,
            )
        else:
            self.dataset = MovieDatasetV2(
                parent_folder,
                movie_name=movie_name,
                keys=keys,
                random_window=random_window,
                random_window_size=random_window_size,
            )
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.cross_val_movie = cross_val_movie

        self.train_dl = None
        self.val_dl = None

    def setup(self, stage: str) -> None:
        if self.cross_val_movie:
            until = len(self.dataset.idxs[0]) + len(self.dataset.idxs[1])
            train_idx = list(range(0, until - 2 * self.random_window_size))
            test_idx = [-1, -2]
        else:
            # For single movie, half for train, half for test
            # odd index for train, even index for test
            train_idx = self.dataset.attrs["split_train"][: -self.random_window_size]
            test_idx = [-1]
        self.train_dl = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            # sampler=SubsetSampler(train_idx),
            pin_memory=False,
            drop_last=True,
        )
        self.val_dl = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            # sampler=SubsetRandomSampler(test_idx),
            sampler=SubsetSampler(test_idx),
            pin_memory=False,
        )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.val_dl

    def predict_dataloader(self):
        return self.val_dl
