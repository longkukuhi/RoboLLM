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

class Beit3baseDataset(torch.utils.data.Dataset):
    def __init__(
            self, split, transform,
            tokenizer, num_max_bpe_tokens,
    ):

        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.transform = transform
        self.split = split

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


class ArmbenchDataset(Beit3baseDataset):
    def __init__(
            self, data_path, split, transform,
            tokenizer, num_max_bpe_tokens, args, task=None,
    ):
        super().__init__(
            split, transform,
            tokenizer, num_max_bpe_tokens,
        )
        self.data_path = data_path

        self.args = args
        if args.task == 'armbench3t1' or args.task == 'armbenchpick1':
            with open(os.path.join(data_path,  f'{split}_data_3t1.json'), 'r') as f:
                self.items = json.load(f)
            # self.items = ImageFolder(os.path.join(data_path, 'Picks', split+'_data_3t1'))
            with open(os.path.join(data_path,  f'all_labels.json'), 'r') as f:
                self.pickid_to_itemid = json.load(f)
            self.pick_ids = list(self.items.keys())
        else:
            raise NotImplementedError

        # self.ref_items = ImageFolder(os.path.join(data_path, 'Reference_Images'))

        # self.clean_pick()

    def _get_pick_class_id(self, pickidx):
        if isinstance(pickidx, list):
            ids = []
            for idx in pickidx:
                ids.append(self.pickid_to_itemid[self.pick_ids[idx]])
            return ids
        elif isinstance(pickidx, int):
            return self.pickid_to_itemid[self.pick_ids[pickidx]]
        else:
            raise RuntimeError("pickidx must be either int or list")

    def clean_pick(self):
        if self.args.task == 'armbench3t1':
            self.classes = np.load(os.path.join(self.data_path, 'Picks_splits', f'{self.split}_classes_over3image.npy'))
            self.target_ids = np.load(os.path.join(self.data_path, 'Picks_splits', f'{self.split}_target_ids_over3image.npy'),
                                      allow_pickle=True)
        else:
            raise NotImplementedError
        # self.classes = self.pick_items.classes.copy()
        # print("Before cleaning: ", len(self.classes), " classes")
        # self.target_ids = []
        # for clsidx in tqdm(range(len(self.pick_items.classes))):
        #     ids = [idx for idx, x in enumerate(self.pick_items.targets) if x == clsidx]
        #     if len(ids) < 3:
        #         self.classes.remove(self.pick_items.classes[clsidx])
        #         continue
        #     if len(ids) > 3:
        #         ids = ids[1:]
        #     self.target_ids.append(ids)
        #
        # print("After cleaning: ", len(self.classes), " classes")

    def _get_pick_image(self, pickid):
        if self.args.task == 'armbench3t1':
            image_paths = self.items[pickid]['pickimg_3t1_paths']
            images = [self.transform(Image.open(os.path.join(self.data_path, 'Picks', pickid, image_path)).convert('RGB')) for image_path in image_paths]
            return images
        elif self.args.task == 'armbenchpick1':
            image = self.transform(
                Image.open(os.path.join(self.data_path, 'Picks', pickid, 'PickRGB.jpg')).convert('RGB'))
            return image
        else:
            raise NotImplementedError

    def _get_ref_image(self, pickid):
        # ids = [idx for idx, x in enumerate(self.ref_items.targets) if x == item_id]
        image_paths = self.items[pickid]['refimg_path']
        ref_id = self.items[pickid]['ref_id']
        if len(image_paths) > 1:
            path = random.sample(image_paths, 1)
            path = os.path.join(self.data_path, 'Reference_Images', ref_id, path[0])
        else:
            path = os.path.join(self.data_path, 'Reference_Images', ref_id, image_paths[0])
        image = self.transform(Image.open(path).convert('RGB'))
        # images = [self.transform(self.ref_items[idx][0].convert('RGB')) for idx in ids]
        return image

    def _get_image_text_example(self, index: int, data: dict):
        pick_id = self.pick_ids[index]
        if self.args.task == 'armbench3t1':
            pick_imgs = self._get_pick_image(pick_id)
            data["image0"] = pick_imgs[0] #OnArmLowRGB
            data["image1"] = pick_imgs[1] #PickRGB.jpg
            data["image2"] = pick_imgs[2] #ToteWallRGB.jpg
        elif self.args.task == 'armbenchpick1':
            pick_img = self._get_pick_image(pick_id)
            data["pick_image"] = pick_img

        # data["pick_images"] = pick_imgs
        data['pick_id'] = index #pick_id
        # item_id = self._get_pick_class_id(index)
        ref_imgs = self._get_ref_image(pick_id)
        data["ref_image"] = ref_imgs

        return data

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)



class ArmbenchPickDataset(Beit3baseDataset):
    def __init__(
            self, data_path, split, transform,
            tokenizer, num_max_bpe_tokens, args, task=None,
    ):
        super().__init__(
             split, transform,
            tokenizer, num_max_bpe_tokens,
        )
        self.data_path = data_path

        self.args = args
        if args.task == 'armbench3t1' or args.task == 'armbenchpick1':
            with open(os.path.join(data_path,  f'{split}_data_3t1.json'), 'r') as f:
                self.pick_items = json.load(f)

            with open(os.path.join(data_path,  f'all_labels.json'), 'r') as f:
                self.pickid_to_itemid = json.load(f)

            self.pick_ids = list(self.pick_items.keys())
            # self.classes = self.items.classes
            # self.target_ids = self.items.targets
        else:
            raise RuntimeError("Task not supported")
        # self.clean_pick()

    def _get_class_id(self, pickidx):
        if isinstance(pickidx, list):
            ids = []
            for idx in pickidx:
                ids.append(self.pickid_to_itemid[self.pick_ids[idx]])
            return ids
        elif isinstance(pickidx, int):
            return self.pickid_to_itemid[self.pick_ids[pickidx]]
        else:
            raise RuntimeError("pickidx must be either int or list")

    def clean_pick(self):
        if self.args.task == 'armbench3t1':
            self.classes = np.load(os.path.join(self.data_path, 'Picks_splits', f'{self.split}_classes_over3image.npy'))
            self.target_ids = np.load(os.path.join(self.data_path, 'Picks_splits', f'{self.split}_target_ids_over3image.npy'), allow_pickle=True)
        else:
            raise RuntimeError("Task not supported")
        # print("Before cleaning: ", len(self.classes), " classes")

        # self.target_ids = []
        # for clsidx in tqdm(range(len(self.items.classes))):
        #     ids = [idx for idx, x in enumerate(self.items.targets) if x == clsidx]
        #     if len(ids) < 3:
        #         self.classes.remove(self.items.classes[clsidx])
        #         continue
        #     if len(ids) > 3:
        #         ids = ids[1:]
        #     self.target_ids.append(ids)
        #
        # print("After cleaning: ", len(self.classes), " classes")

    def _get_image(self, pickid):
        if self.args.task == 'armbench3t1':
            image_paths = self.pick_items[pickid]['pickimg_3t1_paths']
            images = [self.transform(Image.open(os.path.join(self.data_path, 'Picks', pickid, image_path)).convert('RGB')) for image_path in image_paths]
            return images
        elif self.args.task == 'armbenchpick1':
            image = self.transform(Image.open(os.path.join(self.data_path, 'Picks', pickid, 'PickRGB.jpg')).convert('RGB'))
            return image
        else:
            raise NotImplementedError

    def _get_image_text_example(self, index: int, data: dict):
        if self.args.task == 'armbench3t1':
            pick_id = self.pick_ids[index]
            imgs = self._get_image(pick_id)
            # data["images"] = imgs
            data["image0"] = imgs[0] #OnArmLowRGB
            data["image1"] = imgs[1] #PickRGB.jpg
            data["image2"] = imgs[2] #ToteWallRGB.jpg
            data["pick_id"] = index
        elif self.args.task == 'armbenchpick1':
            pick_id = self.pick_ids[index]
            img = self._get_image(pick_id)
            data["pick_image"] = img
            data["pick_id"] = index
        else:
            raise NotImplementedError

        # text_segment = item["text_segment"]
        # language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        # data["language_tokens"] = language_tokens
        # data["padding_mask"] = padding_mask

        return data

    def __getitem__(self, cls_index: int):
        data = dict()
        self._get_image_text_example(cls_index, data)
        return data

    def __len__(self):
        return len(self.pick_items)


class ArmbenchRefDataset(Beit3baseDataset):
    def __init__(
            self, data_path, split, transform,
            tokenizer, num_max_bpe_tokens, args, task=None,
    ):
        super().__init__(
             split, transform,
            tokenizer, num_max_bpe_tokens,
        )
        self.data_path = data_path
        with open(os.path.join(data_path,  f'all_{split}_ref_img_paths.json'), 'r') as f:
            self.items = json.load(f)
        # self.items = ImageFolder(os.path.join(data_path, 'Reference_Images'))
        self.ref_ids = list(self.items.keys())

        self.all_img_paths = []
        self.index_to_refid = []
        for key, items in self.items.items():
            for path in items:
                self.all_img_paths.append(os.path.join(key, path))
                self.index_to_refid.append(key)

    def _get_item_id(self, index):
        if isinstance(index, list):
            return [self.index_to_refid[idx] for idx in index]
        elif isinstance(index, int):
            return self.index_to_refid[index]
        else:
            raise RuntimeError("index must be either int or list")

    def _get_image(self, index):
        # ids = [idx for idx, x in enumerate(self.items.targets) if x == itemid]
        # img = self.transform(self.items[index][0].convert('RGB'))
        # ref_id = self.index_to_refid[index]
        img = self.transform(Image.open(os.path.join(self.data_path, 'Reference_Images',  self.all_img_paths[index])).convert('RGB'))
        return img

    def _get_image_text_example(self, index: int, data: dict):
        img = self._get_image(index)
        data["ref_image"] = img
        data["ref_id"] = index #self.index_to_refid[index]

        # text_segment = item["text_segment"]
        # language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        # data["language_tokens"] = language_tokens
        # data["padding_mask"] = padding_mask

        return data

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self):
        return len(self.all_img_paths)


from beit3_tools.beit3_datasets import create_dataloader, build_transform, get_sentencepiece_model_for_beit3

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

    dataset = ArmbenchDataset(
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

def creat_query_dataset(args, split):
    is_train = False
    tokenizer = get_sentencepiece_model_for_beit3(args)
    transform = build_transform(is_train=is_train, args=args)

    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    dataset = ArmbenchPickDataset(
        data_path=args.data_path,
        split=split,
        transform=transform,
        tokenizer=tokenizer,
        num_max_bpe_tokens=args.num_max_bpe_tokens,
        args=args,
    )

    dataloader = create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )

    return dataloader

def creat_ref_dataset(args, split):
    is_train = False
    tokenizer = get_sentencepiece_model_for_beit3(args)
    transform = build_transform(is_train=is_train, args=args)

    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    dataset = ArmbenchRefDataset(
        data_path=args.data_path,
        split=split,
        transform=transform,
        tokenizer=tokenizer,
        num_max_bpe_tokens=args.num_max_bpe_tokens,
        args=args,
    )

    dataloader = create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )

    return dataloader

