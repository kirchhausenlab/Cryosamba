import os, glob
import numpy as np
import tifffile
import mrcfile
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from core.utils.data_utils import normalize_imgs, denormalize_imgs, augment_dataset


class DatasetBase(Dataset):
    def __init__(self, args, data, metadata, split="train"):
        self.data = data
        self.metadata = metadata
        self.split = split

        self.shape = self.data.shape
        self.max_frame_gap = args.max_frame_gap
        self.patch_overlap = [2 * self.max_frame_gap] + args.patch_overlap
        self.patch_shape = [2 * self.max_frame_gap + 1] + args.patch_shape

        self.indices = [
            np.arange(-overlap, shape, patch_shape - overlap)
            for overlap, shape, patch_shape in zip(
                self.patch_overlap, self.shape, self.patch_shape
            )
        ]
        self.indices[0] = np.arange(
            0,
            self.shape[0] - self.patch_overlap[0],
            self.patch_shape[0] - self.patch_overlap[0],
        )

        if self.split != "test":
            self.split_ratio = args.split_ratio
            length = int(len(self.indices[0]) * self.split_ratio)
            if self.split == "train":
                self.indices[0] = self.indices[0][:length]
            elif self.split == "val":
                if length < len(self.indices[0]):
                    self.indices[0] = self.indices[0][length:]
                else:
                    self.indices[0] = self.indices[0][-1:]

        self.index_shape = [len(idx) for idx in self.indices]

        self.dataset_length = (
            self.index_shape[0] * self.index_shape[1] * self.index_shape[2]
        )

        self.coords_list, self.border_pad_list = list(
            zip(*[self.get_crop_params(index) for index in range(self.dataset_length)])
        )

    def get_crop_params(self, index):
        index_unravel = np.unravel_index(index, self.index_shape)

        coords_start = np.asarray(
            [self.indices[i][idx] for i, idx in enumerate(index_unravel)]
        ).astype("int")
        coords_end = np.asarray(
            [coord + shape for coord, shape in zip(coords_start, self.patch_shape)]
        ).astype("int")

        border_pad = np.asarray([[0, 0], [0, 0], [0, 0]]).astype("int")
        for i in range(3):
            if coords_start[i] < 0:
                border_pad[i][0] = -coords_start[i]
                coords_start[i] = 0
            if coords_end[i] >= self.shape[i]:
                border_pad[i][1] = coords_end[i] - self.shape[i]
                coords_end[i] = self.shape[i]

        coords = np.stack((coords_start, coords_end), axis=-1)

        return coords, border_pad

    def __getitem__(self, index):
        coords, border_pad = self.coords_list[index], self.border_pad_list[index]

        imgs = self.data[
            coords[0, 0] : coords[0, 1],
            coords[1, 0] : coords[1, 1],
            coords[2, 0] : coords[2, 1],
        ]
        imgs = np.pad(imgs, border_pad, mode="reflect")
        imgs = torch.from_numpy(imgs).float()

        if self.split == "train":
            imgs = augment_dataset(imgs)

        imgs = normalize_imgs(imgs, params=self.metadata)

        if self.split == "test":
            crop_params = np.concatenate((coords, border_pad), axis=0)
            return imgs, crop_params

        return imgs

    def __len__(self):
        return self.dataset_length


def get_dataloader(args, data_list, metadata_list, split, is_ddp, shuffle=False):
    dataset = ConcatDataset(
        [
            DatasetBase(args, data, metadata, split=split)
            for data, metadata in zip(data_list, metadata_list)
        ]
    )

    sampler = DistributedSampler(dataset, shuffle=shuffle) if is_ddp else None
    shuffle = None if is_ddp else shuffle

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
        sampler=sampler,
        shuffle=shuffle,
    )
