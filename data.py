import os
import traceback
from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm

from extractors import get_feature_network

NUM_AUGS = 16


@torch.inference_mode()
def extract_features(input_dir, features_file, feature_net, image_size, batch_size, workers):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = Transforms(image_size, jitter=0.5).train_transform
        dataset = Images(root=input_dir, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers)

        feature_extractor = get_feature_network(feature_net).eval().float().to(device)

        num_feats = len(dataset) * NUM_AUGS
        features = open_memmap(features_file, mode="w+", dtype="float32", shape=(num_feats, feature_extractor.rep_dim))

        i = 0
        with tqdm(total=num_feats, desc="Extracting features...", smoothing=0) as pbar:
            for n in range(NUM_AUGS):
                for f, x in data_loader:
                    features[i : i + len(x), :] = feature_extractor(x.to(device)).cpu().numpy()
                    i += len(x)
                    pbar.update(len(x))
        features.flush()
        return feature_extractor.rep_dim

    except Exception:
        traceback.print_exc()
    except KeyboardInterrupt:
        traceback.print_exc()

    print("\n\nException occurred while extracting features! Deleting broken cache file...\n\n")
    os.remove(features_file)
    exit(1)


class Images(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super().__init__()
        extensions = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"]
        self.imgs = sum([glob(root + "/*" + ext) for ext in extensions], [])
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    @torch.inference_mode()
    def __getitem__(self, idx):
        file = self.imgs[idx]
        return file, self.transform(Image.open(file).convert("RGB"))


class Transforms:
    def __init__(self, size: int, jitter: float = 1.0):
        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8 * jitter, 0.8 * jitter, 0.8 * jitter, 0.2 * jitter)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
            ]
        )
        self.test_transform = T.Compose([T.Resize(size, antialias=True), T.CenterCrop(size), T.ToTensor()])


class Features(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.feats = np.load(filename, mmap_mode="r")

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        dataset_size = len(self) // NUM_AUGS  # number of images in the dataset

        # get index of a different augmented version of this image
        which_augment = np.random.randint(1, NUM_AUGS)
        other_idx = (idx + dataset_size * which_augment) % len(self)

        # return the image and the augmented version (copy because PyTorch tensors don't respect read-only)
        return self.feats[idx].copy(), self.feats[other_idx].copy()


class DataPrefetcher:
    """From https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py

    BSD 3-Clause "New" or "Revised" License:

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
       disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
       following disclaimer in the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
       products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_filepath, self.next_batch = next(self.loader)
        except StopIteration:
            self.next_filepath = None
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        filepath = self.next_filepath
        batch = self.next_batch
        if batch is not None:
            batch.record_stream(torch.cuda.current_stream())
        self.preload()
        return filepath, batch
