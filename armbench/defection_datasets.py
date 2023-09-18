import os
import json
import random

import torch
import glob
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, \
    IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform
from tqdm import tqdm

from beit3_tools import utils
from beit3_tools.glossary import normalize_word
from beit3_tools.randaug import RandomAugment

# AtoMiC
from datasets import load_dataset
import sys
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from armbench.armbench_datasets import Beit3baseDataset
from beit3_tools.beit3_datasets import create_dataloader, build_transform, get_sentencepiece_model_for_beit3
import armbench.modeling

class ArmbenchDefectionDataset(Beit3baseDataset):
    def __init__(
            self, data_path, split, transform,
            tokenizer, num_max_bpe_tokens, args, task=None,
    ):
        super().__init__(
            split, transform,
            tokenizer, num_max_bpe_tokens,
        )
        self.args = args
        self.split = split
        self.data_path = data_path
        if args.task in ['defection1by1']:
            self.items = np.load(os.path.join(data_path, f'{split}_data.npy'), allow_pickle=True)

    def _get_image_text_example(self, index, data):
        if self.args.task in ['defection1by1']:
            img_path = self.items[index]['image']
            box = self.items[index]['box']
            if not img_path.endswith('.jpg'):
                img_path = img_path + '.jpg'
            img_path = os.path.join(self.data_path, img_path)
            if box != [0,0,0,0]:
                box = [int(x) for x in box]
                img = Image.open(img_path).convert("RGB").crop(box)
            else:
                img = Image.open(img_path).convert("RGB")

            img = self.transform(img)
            data["image"] = img
            data["label"] = self.items[index]['label']

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        if self.args.task in ['defection1by1']:
            return len(self.items)


def create_armbench_dataset(args, split):
    is_train = split == "train"
    tokenizer = get_sentencepiece_model_for_beit3(args)
    transform = build_transform(is_train=is_train, args=args)

    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    dataset = ArmbenchDefectionDataset(
        data_path=args.data_path,
        split=split,
        transform=transform,
        tokenizer=tokenizer,
        num_max_bpe_tokens=args.num_max_bpe_tokens,
        args=args,
    )

    dataloader = create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, dist_eval=args.distributed,
        pin_mem=args.pin_mem,
    )

    return dataloader