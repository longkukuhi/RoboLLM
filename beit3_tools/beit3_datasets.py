# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

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

from beit3_tools import utils
from beit3_tools.glossary import normalize_word
from beit3_tools.randaug import RandomAugment

# AtoMiC
from datasets import load_dataset
import sys
import numpy as np
from PIL import Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self, data_path, split, transform,
            tokenizer, num_max_bpe_tokens, task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)

        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

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


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write('\n')
    print("Write %s with %d items !" % (jsonl_file, len(items)))


def _make_retrieval_coco_karpathy_dataset_index(
        data_path,
        tokenizer,
        split=("train", "restval"),
        split_name="train",
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                for sent in item["sentences"]:
                    tokens = tokenizer.tokenize(sent["raw"])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append({
                        "image_path": image_path,
                        "text_segment": token_ids,
                        "image_id": len(image_counter),
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
          (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_retrieval.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass


def _make_captioning_coco_karpathy_dataset_index(
        data_path,
        tokenizer,
        split=("train", "restval"),
        split_name="train",
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                if item["split"] in ["train", "restval"]:
                    for sent in item["sentences"]:
                        tokens = tokenizer.tokenize(sent["raw"])
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                        items.append({
                            "image_path": image_path,
                            "text_segment": token_ids,
                            "image_id": item["cocoid"],
                        })
                else:
                    items.append({
                        "image_path": image_path,
                        "text_segment": None,
                        "image_id": item["cocoid"],
                    })
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print("Find %d images and %d image-text pairs for karpathy dataset %s split !" % \
          (len(image_counter), len(items), split_name))
    index_file = os.path.join(data_path, "coco_captioning.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass


def _make_nocaps_dataset_index(
        data_path,
        split="val",
):
    if split == "val":
        json_file = "nocaps_val_4500_captions.json"
    elif split == "test":
        json_file = "nocaps_test_image_info.json"
    nocaps_split_json_file = os.path.join(data_path, json_file)
    items = []
    image_counter = set()
    print("read %s" % nocaps_split_json_file)
    with open(nocaps_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            image_path = os.path.join(split, item["file_name"])
            items.append({
                "image_path": image_path,
                "text_segment": None,
                "image_id": item["id"],
            })

            if image_path not in image_counter:
                image_counter.add(image_path)

    print("Find %d images and %d image-text pairs for nocaps dataset %s split !" % \
          (len(image_counter), len(items), split))
    index_file = os.path.join(data_path, "nocaps.%s.jsonl" % split)
    _write_data_into_jsonl(items, index_file)


class NLVR2Dataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("nlvr2.train.index.jsonl",)
        elif split == "val":
            return ("nlvr2.dev.index.jsonl",)
        elif split == "test":
            return ("nlvr2.test-P.index.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        item = self.items[index]
        img_path = item["image2_path"]
        img = self._get_image(img_path)
        data["image2"] = img
        data["label"] = self.items[index]["label"]
        return data

    @staticmethod
    def __preprocess_json(preifx, json_file, tokenizer, index_file):
        items = []
        with open(json_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                path = os.path.join(preifx, str(data["directory"])) if "directory" in data else preifx
                path = os.path.join(path, "-".join(data["identifier"].split("-")[:-1]))
                tokens = tokenizer.tokenize(data["sentence"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                items.append({
                    "image_path": path + "-img0.png",
                    "image2_path": path + "-img1.png",
                    "text_segment": token_ids,
                    "label": 1 if data["label"] == "True" else 0,
                    "identifier": data["identifier"],
                })
        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, nlvr_repo_path):
        cls.__preprocess_json(
            preifx="images/train", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/train.json"),
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("train")[0]),
        )
        cls.__preprocess_json(
            preifx="dev", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/dev.json"),
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("val")[0]),
        )
        cls.__preprocess_json(
            preifx="test1", json_file=os.path.join(nlvr_repo_path, "nlvr2/data/test1.json"),
            tokenizer=tokenizer, index_file=os.path.join(data_path, cls.get_index_files("test")[0]),
        )


class ImageNetDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("imagenet.train.index.jsonl",)
        elif split == "val":
            return ("imagenet.val.index.jsonl",)
        elif split == "test":
            return ("imagenet.val.index.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["label"] = item["label"]
        return data

    @staticmethod
    def _find_classes(dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def _make_imagenet_index(data_path, index_path, data_path_prefix, class_to_idx, split):
        items = []
        index_file = os.path.join(index_path, f"imagenet.{split}.index.jsonl")
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(data_path, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    path = path.replace(data_path_prefix, "")
                    items.append({
                        "image_path": path,
                        "label": class_index,
                    })

        _write_data_into_jsonl(items, index_file)

    @classmethod
    def make_dataset_index(cls, train_data_path, val_data_path, index_path):
        data_path_prefix = train_data_path[:[x[0] == x[1] for x in zip(train_data_path, val_data_path)].index(0)]
        classes, class_to_idx = cls._find_classes(train_data_path)
        cls._make_imagenet_index(
            data_path=train_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
            class_to_idx=class_to_idx, split="train",
        )
        cls._make_imagenet_index(
            data_path=val_data_path, index_path=index_path, data_path_prefix=data_path_prefix,
            class_to_idx=class_to_idx, split="val",
        )


class VQAv2Dataset(BaseDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        ans2label_file = os.path.join(data_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                ans2label[ans] = i
                label2ans.append(ans)

        self.ans2label = ans2label
        self.label2ans = label2ans

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("vqa.train.jsonl", "vqa.trainable_val.jsonl")
        elif split == "val":
            return ("vqa.rest_val.jsonl",)
        elif split == "test":
            return ("vqa.test.jsonl",)
        elif split == "test-dev":
            return ("vqa.test-dev.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        if "labels" in self.items[index] and len(self.items[index]["labels"]) > 0:
            labels = [0.] * len(self.label2ans)
            for l, s in zip(self.items[index]["labels"], self.items[index]["scores"]):
                labels[l] = s
            data["labels"] = torch.FloatTensor(labels)
        else:
            data["qid"] = self.items[index]["qid"]
        return data

    @staticmethod
    def get_score(occurences):
        if occurences == 0:
            return 0.0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1.0

    @classmethod
    def make_dataset_index(cls, data_path, tokenizer, annotation_data_path):
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_train2014_questions.json"), "r") as fp:
            questions_train2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_val2014_questions.json"), "r") as fp:
            questions_val2014 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test2015_questions.json"), "r") as fp:
            questions_test2015 = json.load(fp)["questions"]
        with open(os.path.join(annotation_data_path, "v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as fp:
            questions_test_dev2015 = json.load(fp)["questions"]

        with open(os.path.join(annotation_data_path, "v2_mscoco_train2014_annotations.json"), "r") as fp:
            annotations_train2014 = json.load(fp)["annotations"]
        with open(os.path.join(annotation_data_path, "v2_mscoco_val2014_annotations.json"), "r") as fp:
            annotations_val2014 = json.load(fp)["annotations"]

        annotations = dict()

        for split, questions in zip(
                ["train", "val", "test", "test-dev"],
                [questions_train2014, questions_val2014, questions_test2015, questions_test_dev2015],
        ):
            _annot = defaultdict(dict)
            for q in questions:
                question_text = q["question"]
                tokens = tokenizer.tokenize(question_text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                assert q["question_id"] not in _annot[q["image_id"]]
                _annot[q["image_id"]][q["question_id"]] = {
                    "question": question_text,
                    "token_ids": token_ids,
                }

            annotations[split] = _annot

        all_major_answers = list()

        for split, annots in zip(
                ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            # _annot = annotations[split]
            for q in annots:
                all_major_answers.append(q["multiple_choice_answer"])

        all_major_answers = [normalize_word(word) for word in all_major_answers]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
        ans2label = {k: i for i, k in enumerate(counter.keys())}
        label2ans = list(counter.keys())

        for split, annots in zip(
                ["train", "val"], [annotations_train2014, annotations_val2014],
        ):
            _annot = annotations[split]
            for q in annots:
                answers = q["answers"]
                answer_count = {}
                for answer in answers:
                    answer_ = answer["answer"]
                    answer_count[answer_] = answer_count.get(answer_, 0) + 1

                labels = []
                scores = []
                for answer in answer_count:
                    if answer not in ans2label:
                        continue
                    labels.append(ans2label[answer])
                    score = cls.get_score(answer_count[answer])
                    scores.append(score)

                assert "labels" not in _annot[q["image_id"]][q["question_id"]]
                assert "question" in _annot[q["image_id"]][q["question_id"]]
                _annot[q["image_id"]][q["question_id"]]["labels"] = labels
                _annot[q["image_id"]][q["question_id"]]["scores"] = scores

        for split in ["train", "val"]:
            filtered_annot = dict()
            for ik, iv in annotations[split].items():
                new_q = dict()
                for qk, qv in iv.items():
                    if len(qv["labels"]) != 0:
                        new_q[qk] = qv
                if len(new_q) != 0:
                    filtered_annot[ik] = new_q
            annotations[split] = filtered_annot

        split2items = {}
        for split in ["train", "val", "test", "test-dev"]:
            annot = annotations[split]
            split_name = {
                "train": "train2014",
                "val": "val2014",
                "test": "test2015",
                "test-dev": "test2015",
            }[split]
            paths = list(glob.glob(f"{data_path}/{split_name}/*.jpg"))
            random.shuffle(paths)
            annot_paths = [path for path in paths \
                           if int(path.split("/")[-1].split("_")[-1][:-4]) in annot]

            if len(paths) == len(annot_paths):
                print("all images have caption annotations")
            else:
                print("not all images have caption annotations")
            print(len(paths), len(annot_paths), len(annot))

            items = []
            for path in annot_paths:
                iid = int(path.split("/")[-1].split("_")[-1][:-4])
                _annot = annotations[split][iid]
                for qid in _annot:
                    q = _annot[qid]
                    if split in ["train", "val"]:
                        labels = q["labels"]
                        scores = q["scores"]
                    else:
                        labels, scores = [], []

                    items.append({
                        # original
                        # "image_path": os.path.join(split_name, path.split('/')[-1]),
                        # windows
                        "image_path": path.split('/')[-1],
                        "text_segment": q["token_ids"],
                        "labels": labels,
                        "scores": scores,
                        "qid": qid,
                    })
            split2items[split] = items

            _write_data_into_jsonl(items=items, jsonl_file=os.path.join(data_path, "vqa.%s.jsonl" % split))

        # Following ViLT, we use 1000 images of the original val set as the final val set
        val_image2items = defaultdict(list)
        for item in split2items["val"]:
            val_image2items[item["image_path"]].append(item)

        print("Contains %d image and %d pairs for val set!" % (len(val_image2items), len(split2items["val"])))

        val_images = list(val_image2items.keys())
        random.shuffle(val_images)
        trainable_val = []
        rest_val = []
        for i, image_id in enumerate(val_images):
            if i < 1000:
                rest_val += val_image2items[image_id]
            else:
                trainable_val += val_image2items[image_id]

        _write_data_into_jsonl(items=trainable_val, jsonl_file=os.path.join(data_path, "vqa.trainable_val.jsonl"))
        _write_data_into_jsonl(items=rest_val, jsonl_file=os.path.join(data_path, "vqa.rest_val.jsonl"))

        with open(os.path.join(data_path, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans,
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))


class RetrievalDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl",)
        elif split == "val":
            return (f"{task}.val.jsonl",)
        elif split == "test":
            return (f"{task}.test.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["image_id"] = self.items[index]["image_id"]
        return data

    @staticmethod
    def make_flickr30k_dataset_index(data_path, tokenizer, karpathy_path):

        with open(os.path.join(karpathy_path, "dataset_flickr30k.json"), "r") as reader:
            captions = json.loads(reader.read())

        captions = captions["images"]
        split2items = defaultdict(list)
        split2images = defaultdict(set)

        for each_item in captions:
            image_path = os.path.join("flickr30k-images", each_item["filename"])
            split = each_item["split"]

            for text_segment in each_item["sentences"]:
                tokens = tokenizer.tokenize(text_segment["raw"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                split2items[split].append({
                    "image_path": image_path,
                    "text_segment": token_ids,
                    "image_id": len(split2images[split]),
                })

            assert each_item["filename"] not in split2images[split]
            split2images[split].add(each_item["filename"])

        for split in split2items:
            print("%d images and %d image-text pairs!" % (len(split2images[split]), len(split2items[split])))
            _write_data_into_jsonl(split2items[split], os.path.join(data_path, "flickr30k.%s.jsonl" % split))

    @staticmethod
    def make_coco_dataset_index(data_path, tokenizer):
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"),
                                                    split_name="train")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("val",), split_name="val")
        _make_retrieval_coco_karpathy_dataset_index(data_path, tokenizer, split=("test",), split_name="test")


class CaptioningDataset(BaseDataset):

    def __init__(self, data_path, split, transform,
                 tokenizer, num_max_bpe_tokens, task, mask_prob):
        super().__init__(
            data_path=data_path, split=split,
            transform=transform, tokenizer=tokenizer,
            num_max_bpe_tokens=num_max_bpe_tokens, task=task,
        )
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = mask_prob

    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return ("coco_captioning.train.jsonl",)
        elif split == "val":
            return (f"{task}.val.jsonl",)
        elif split == "test":
            return (f"{task}.test.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def _get_mask_token(self, token):
        p = random.random()
        if p < 0.8:
            return self.mask_token_id
        elif p < 0.9:
            return token
        else:
            return random.randint(3, self.language_vocab_size - 1)

    def _masking_on_text_tokens(self, tokens, num_tokens, mask_prob):
        bool_masked_pos = [0] * len(tokens)
        to_mask = min(int(num_tokens * mask_prob + 0.5), num_tokens - 1)
        to_mask = max(to_mask, 1)
        num_masked_tokens = 0
        while num_masked_tokens < to_mask:
            i = random.randint(1, num_tokens - 1)
            if bool_masked_pos[i] == 0:
                bool_masked_pos[i] = 1
                tokens[i] = self._get_mask_token(tokens[i])
                num_masked_tokens += 1

        return tokens, bool_masked_pos

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        text_segment = item["text_segment"]
        if text_segment is not None:
            language_tokens, padding_mask, num_tokens = self._get_text_segment(text_segment)
            masked_tokens = language_tokens[:]
            masked_tokens, language_masked_pos = \
                self._masking_on_text_tokens(masked_tokens, num_tokens, self.mask_prob)
            data["language_tokens"] = language_tokens
            data["masked_tokens"] = masked_tokens
            data["language_masked_pos"] = language_masked_pos
            data["padding_mask"] = padding_mask
        return data

    @staticmethod
    def make_coco_captioning_dataset_index(data_path, tokenizer):
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("train", "restval"),
                                                     split_name="train")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("val",), split_name="val")
        _make_captioning_coco_karpathy_dataset_index(data_path, tokenizer, split=("test",), split_name="test")

    @staticmethod
    def make_nocaps_captioning_dataset_index(data_path):
        _make_nocaps_dataset_index(data_path, split="val")
        _make_nocaps_dataset_index(data_path, split="test")


from tqdm import tqdm



class BaseAtoMicDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path=None, load_tokenized_text=False,
                  load_image_from_huggingface_hub=True, ):


        self.load_image_from_huggingface_hub = load_image_from_huggingface_hub
        if self.load_image_from_huggingface_hub:
            print("load images data from huggingface hub: ")
            if args.cluster:
                self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train', cache_dir='../../datasets/ATOMIC/',
                                           num_proc=4)
            else:
                self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train',
                                           num_proc=4)
            print("build index to map image_id to index in self.images: ")
            image_ids = self.images['image_id']
            self.image_id2row_dict = self._getRowIdx(image_ids)
        else:
            print("load images data from local file: ------------ ")
        # else:
        #     image_ids = json.loads(open("../datasets/ATOMIC/all_image_ids.json").read())
        #     self.image_id2row_dict = self._getRowIdx(image_ids)

        if not load_tokenized_text:
            print("load texts data from huggingface hub: ")
            if args.cluster:
                self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train', cache_dir='../../datasets/ATOMIC/',
                                          num_proc=4)
            else:
                self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train',
                                          num_proc=4)
            print("build index to map text_id to index in self.texts: ")
            text_ids = self.texts['text_id']
            self.text_id2row_dict = self._getRowIdx(text_ids)

        # self.load_tokenized_text = load_tokenized_text

        # if not load_tokenized_text:
        #     with _log_time_usage("load text data from json file: "):
        #         print("load text data from json file: ")
        #         self.texts = []
        #         if data_path is None:
        #             data_path = f"../datasets/ATOMIC/Atomic_text_tokenized_{split}.json"
        #         else:
        #             data_path = os.path.join(data_path, f"Atomic_text_tokenized_{split}.json")
        #         with open(data_path, "r") as reader:
        #             for line in tqdm(reader):
        #                 data = json.loads(line)
        #                 self.texts.append(data)
        # else:
        #     self.texts = []
        #     with open(f"../datasets/ATOMIC/Atomic_text_{split}.json", "r") as reader:
        #         for line in reader:
        #             data = json.loads(line)
        #             self.texts.append(data)

    def _getRowIdx(self, id_list: list) -> dict:
        id2pos = {}
        for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
            id2pos[_id] = pos
        return id2pos


class AtoMicDataset(BaseAtoMicDataset):
    def __init__(self, args, split, transform,
                 tokenizer, num_max_bpe_tokens, text_features=None,
                 data_path=None, load_tokenized_text=True):
        # load dataset from huggingface
        super().__init__(args, data_path, load_tokenized_text=args.load_tokenized_text,
                         load_image_from_huggingface_hub=args.load_image_from_huggingface_hub)

        self.args = args
        print("load qrels data from huggingface hub: ")
        # split = 'test'
        # self.qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split, cache_dir='../datasets/ATOMIC/', )
        self.qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split, )

        if text_features is None:
            # text_features = ["page_title", "section_title",
            #                  "context_sec_des", ] #"context_page_des",
            text_features = ["page_title", "section_title", "context_section_description",
                              ] #"context_page_description"
        self.text_features = text_features
        self.split = split
        self.data_path = data_path
        self.transform = transform

        if args.load_tokenized_text:
            print("load tokenized text data from json file: ")
            self.texts = []
            if not args.eval:
                if data_path is None:
                    data_path = f"../datasets/ATOMIC/Atomic_text_tokenized_{split}.json"
                else:
                    data_path = os.path.join(data_path, f"Atomic_text_tokenized_{split}.json")
            else:
                data_path = f"../datasets/ATOMIC/Atomic_text_tokenized_{split}.json" #_all_fileds

            with open(data_path, "r") as reader:
                for line in tqdm(reader):
                    data = json.loads(line)
                    self.texts.append(data)

            text_ids = set(self.qrels['text_id'])
            self.text_id2row_dict = self._getRowIdx(text_ids)

        self.load_tokenized_text = args.load_tokenized_text

        # BEIT3 parameters
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = args.captioning_mask_prob

    #   self.check_dataset_intergrity()

    def check_dataset_intergrity(self):
        print("check dataset intergrity: ")
        for image_id in self.qrels['image_id']:
            idx = self.image_id2row_dict[image_id]
            try:
                iamge = self.images[idx]['image']
                image = self.transform(image)
            except:
                print(f"image_id: {image_id} does not have image")

        sys.exit()
        return True

    def _getRowIdx(self, id_list: list) -> dict:
        id2pos = {}
        for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
            id2pos[_id] = pos
        return id2pos

    def _split_images(self, images, qrels):
        image_ids = set(qrels["image_id"])
        valid_image_ids = []
        for idx, image_id in tqdm(enumerate(images["image_id"]), total=len(images)):
            if image_id in image_ids:
                valid_image_ids.append(images[idx])
        return valid_image_ids

    def _get_image(self, image_id):
        if self.load_image_from_huggingface_hub:
            idx = self.image_id2row_dict[image_id]
            image = self.images[idx]['image']
            image = self.transform(image)
            return image
        else:

            # image = np.load(os.path.join('../datasets/ATOMIC/images', self.split, f"{image_id}.npy"))
            if self.args.cluster:
                img_path = os.path.join('../datasets/ATOMIC/images', self.split, f"{image_id}.jpg")
            else:
                img_path = os.path.join('\\\\longkukunas\\PhD data\\datasets\\ATOMIC\\images', self.split, f"{image_id}.jpg")

            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image

    def _tokenize_text(self, new_dict):
        result = []
        for key in self.text_features:
            content = new_dict[key]
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
            if len(content) > self.num_max_bpe_tokens:
                content = content[:self.num_max_bpe_tokens]

            result += content
            if len(result) >= (self.num_max_bpe_tokens - 2) :
                break
        return result

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
        # retreive image and text pair from qrels
        item = self.qrels[index]
        img_id = item['image_id']
        text_id = item['text_id']
        # data["image_id"] = self.image_id2row_dict[img_id]
        data["image_id"] = index
        # data["text_id"] = self.text_id2row_dict[text_id]
        data["image"] = self._get_image(img_id)

        text_dict = self.texts[self.text_id2row_dict[text_id]]
        # if not self.load_tokenized_text:
        #     text_dict = self._tokenize_text(text_dict)
        if not self.load_tokenized_text:
            text_segment = self._tokenize_text(text_dict)
            # text_segment = [token for feature in self.text_features for token in text_dict[feature]]
        else:
            text_segment = [token for feature in self.text_features for token in text_dict[f"{feature}_tokens_ids"]]

        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

        return data

    def _get_img_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.qrels[idx]['image_id'])
            return id_list
        elif isinstance(index, int):
            return self.qrels[index]['image_id']
        else:
            raise ValueError("index should be either list or int")

    def _get_text_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.qrels[idx]['text_id'])
            return id_list

        elif isinstance(index, int):
            return self.qrels[index]['text_id']
        else:
            raise ValueError("index should be either list or int")

    def __getitem__(self, index):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self):
        # return 1200
        # print("The length of qrels is",len(self.qrels))
        return len(self.qrels)


class AtoMicDatasetanswer(torch.utils.data.Dataset):
    def __init__(self, args, transform,
                 tokenizer, num_max_bpe_tokens, text_features=None,
                 data_path=None, load_tokenized_text=True):

        # super().__init__(data_path, load_tokenized_text=args.load_tokenized_text,
        #                  load_image_from_huggingface_hub=args.load_image_from_huggingface_hub)
        super().__init__()

        self.args = args
        self.load_image_from_huggingface_hub = args.load_image_from_huggingface_hub
        self.load_tokenized_text = args.load_tokenized_text

        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = args.captioning_mask_prob
        if text_features is None:
            # text_features = ["page_title", "section_title",
            #                  "context_sec_des", ] #"context_page_des",
            text_features = ["page_title", "section_title", "context_section_description",
                              ] #"context_page_description"

        self.text_features = text_features

        self.transform = transform

        self.retrieval_mode = args.retrieval_mode



        if self.retrieval_mode == "text_to_image":
            if args.load_image_from_huggingface_hub:
                # load all images
                print("load images data from huggingface hub: ")
                # self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train', cache_dir='../datasets/ATOMIC/',
                #                            num_proc=4)
                self.images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train',
                                           num_proc=4)
                print("build index to map image_id to index in self.images: ")
                self.image_ids = self.images['image_id']
                # self.image_id2row_dict = self._getRowIdx(image_ids)
            else:
                print("load images data from local: ")
                self.image_ids = np.load('../datasets/ATOMIC/all_image_ids.npy')

        elif self.retrieval_mode == "image_to_text":
            self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train',
                                      num_proc=4)
            self.text_ids = self.texts['text_id']

    def _get_img_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.image_ids[idx])
            return id_list
        elif isinstance(index, int):
            return self.image_ids[index]
        else:
            raise ValueError("index should be either list or int")

    def _get_text_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.texts[idx]['text_id'])
            return id_list

        elif isinstance(index, int):
            return self.texts[index]['text_id']
        else:
            raise ValueError("index should be either list or int")

    def _get_image(self, idx):
        if self.load_image_from_huggingface_hub:
            image = self.images[idx]['image']
            image = self.transform(image)
            return image
        else:
            image_id = self.image_ids[idx]
            if self.args.cluster:
                img_path = os.path.join('../../datasets/ATOMIC/all_images', self.split, f"{image_id}.jpg")
            else:
                img_path = f"..\\datasets\\ATOMIC\\all_images\\{image_id}"#.jpg
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image

    def _tokenize_text(self, new_dict):
        result = []
        for key in self.text_features:
            content = new_dict[key]
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
            if len(content) > self.num_max_bpe_tokens:
                content = content[:self.num_max_bpe_tokens]

            result += content
            if len(result) >= (self.num_max_bpe_tokens - 2) :
                break
        return result

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


    def __getitem__(self, index):
        if self.retrieval_mode == "text_to_image":
            # offset = 4800000
            # index += offset
            data = dict()
            data["image_id"] = index
            data["image"] = self._get_image(index)
            return data

        else:
            data = dict()
            data["text_id"] = index
            text_dict = self.texts[index]
            # if not self.load_tokenized_text:
            #     text_dict = self._tokenize_text(text_dict)
            if not self.load_tokenized_text:
                text_segment = self._tokenize_text(text_dict)
                # text_segment = [token for feature in self.text_features for token in text_dict[feature]]
            else:
                text_segment = [token for feature in self.text_features for token in text_dict[f"{feature}_tokens_ids"]]

            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
            return data

    def __len__(self):
        if self.retrieval_mode == "text_to_image":
            return len(self.image_ids)
        else:
            return len(self.text_ids)

class AtoMicDatasetquery(torch.utils.data.Dataset):
    def __init__(self, args, transform,
                 tokenizer, num_max_bpe_tokens, text_features=None,
                 data_path=None, load_tokenized_text=True):

        # super().__init__(data_path, load_tokenized_text=args.load_tokenized_text,
        #                  load_image_from_huggingface_hub=args.load_image_from_huggingface_hub)
        super().__init__()
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.language_vocab_size = tokenizer.vocab_size
        self.mask_prob = args.captioning_mask_prob
        if text_features is None:
            # text_features = ["page_title", "section_title",
            #                  "context_sec_des", ] #"context_page_des",
            text_features = ["page_title", "section_title", "context_section_description",
                              ] #"context_page_description"

        self.text_features = text_features

        self.transform = transform

        self.retrieval_mode = args.retrieval_mode

        self.querys = load_dataset(args.query_url, split='train',
                                     num_proc=4)
        self.load_tokenized_text = args.load_tokenized_text
        # if args.retrieval_mode == "text_to_image":
        #     self.text_ids = self.querys['text_id']
        #     self.text_id2row_dict = self._getRowIdx(self.text_ids)
        #
        # elif args.retrieval_mode == "image_to_text":
        #     self.image_ids = self.querys['image_id']
        #     self.image_id2row_dict = self._getRowIdx(self.image_ids)
        #
    def _get_img_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.querys[idx]['image_id'])
            return id_list
        elif isinstance(index, int):
            return self.querys[index]['image_id']
        else:
            raise ValueError("index should be either list or int")

    def _get_text_id(self, index):
        if isinstance(index, list):
            id_list = []
            for idx in index:
                id_list.append(self.querys[idx]['text_id'])
            return id_list

        elif isinstance(index, int):
            return self.querys[index]['text_id']
        else:
            raise ValueError("index should be either list or int")

    def _getRowIdx(self, id_list: list) -> dict:
        id2pos = {}
        for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
            id2pos[_id] = pos
        return id2pos

    def _get_image(self, idx):
        image = self.querys[idx]['image']
        image = self.transform(image)
        return image

    def _tokenize_text(self, new_dict):
        result = []
        for key in self.text_features:
            content = new_dict[key]
            content = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(content))
            if len(content) > self.num_max_bpe_tokens:
                content = content[:self.num_max_bpe_tokens]

            result += content
            if len(result) >= (self.num_max_bpe_tokens - 2) :
                break
        return result

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


    def __getitem__(self, index):
        if self.retrieval_mode == "text_to_image":
            # query is text, return text
            data = dict()
            data["text_id"] = index
            text_dict = self.querys[index]
            # if not self.load_tokenized_text:
            #     text_dict = self._tokenize_text(text_dict)
            if not self.load_tokenized_text:
                text_segment = self._tokenize_text(text_dict)
                # text_segment = [token for feature in self.text_features for token in text_dict[feature]]
            else:
                text_segment = [token for feature in self.text_features for token in text_dict[f"{feature}_tokens_ids"]]

            language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
            data["language_tokens"] = language_tokens
            data["padding_mask"] = padding_mask
            return data


        else:
            data = dict()
            data["image_id"] = index
            data["image"] = self._get_image(index)
            return data


    def __len__(self):
        return len(self.querys)



task2dataset = {
    "nlvr2": NLVR2Dataset,
    "vqav2": VQAv2Dataset,
    "flickr30k": RetrievalDataset,
    "coco_retrieval": RetrievalDataset,
    "coco_captioning": CaptioningDataset,
    "nocaps": CaptioningDataset,
    "imagenet": ImageNetDataset,
    "atomic": AtoMicDataset,
    "atomic_answer": AtoMicDatasetanswer,
    "atomic_query": AtoMicDatasetquery,
}


def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )

    else:
        sampler = torch.utils.data.SequentialSampler(dataset,)

    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):
    if args.task in ["imagenet"]:
        return build_imagenet_transform(is_train, args)

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(args.input_size, scale=(0.5, 1.0),
                                              interpolation=args.train_interpolation),
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
                    ]))
        t += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t


def build_imagenet_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args.sentencepiece_model)


def create_dataset_by_split(args, split, is_train=True):
    transform = build_transform(is_train=is_train, args=args)
    dataset_class = task2dataset[args.task]
    tokenizer = get_sentencepiece_model_for_beit3(args)

    opt_kwargs = {}
    if args.task in ["coco_captioning", "nocaps"]:
        opt_kwargs["mask_prob"] = args.captioning_mask_prob

    if args.task == 'atomic':
        dataset = dataset_class(args=args,
                                data_path=args.data_path, split=split,
                                transform=transform, tokenizer=tokenizer,
                                num_max_bpe_tokens=args.num_max_bpe_tokens,
                                **opt_kwargs,
                                )
    else:
        dataset = dataset_class(
            data_path=args.data_path, split=split,
            transform=transform, tokenizer=tokenizer,
            num_max_bpe_tokens=args.num_max_bpe_tokens,
            task=args.task, **opt_kwargs,
        )

    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    return create_dataloader(
        dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )



def create_query_answer_dataset(args):
    is_train = False
    transform = build_transform(is_train=is_train, args=args)

    tokenizer = get_sentencepiece_model_for_beit3(args)
    if is_train:
        batch_size = args.batch_size
    elif hasattr(args, "eval_batch_size") and args.eval_batch_size is not None:
        batch_size = args.eval_batch_size
    else:
        batch_size = int(args.batch_size * 1.5)

    # create query dataset
    dataset_class = AtoMicDatasetquery

    query_dataset = dataset_class(args=args,
                                data_path=args.data_path,
                                transform=transform, tokenizer=tokenizer,
                                num_max_bpe_tokens=args.num_max_bpe_tokens, )



    query_dataloader = create_dataloader(
        query_dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )

    # create answer dataset
    dataset_class = AtoMicDatasetanswer

    answer_dataset = dataset_class(args=args,
                                data_path=args.data_path,
                                transform=transform, tokenizer=tokenizer,
                                num_max_bpe_tokens=args.num_max_bpe_tokens, )

    answer_dataloader = create_dataloader(
        answer_dataset, is_train=is_train, batch_size=batch_size,
        num_workers=args.num_workers, pin_mem=args.pin_mem, dist_eval=args.dist_eval,
    )


    return query_dataloader, answer_dataloader


def create_downstream_dataset(args, is_eval=False, eval_on_test_set=False):
    if is_eval:
        return create_dataset_by_split(args, split="test", is_train=False)

    elif eval_on_test_set:
        return create_dataset_by_split(args, split="submission", is_train=False)

    else:
        return \
            create_dataset_by_split(args, split="train", is_train=True), \
                create_dataset_by_split(args, split="validation", is_train=False)
