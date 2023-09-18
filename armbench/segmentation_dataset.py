# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")
import cv2
import numpy as np
from detectron2.structures import BoxMode
from armbench.armbench_datasets import Beit3baseDataset
import os
import json
import torch

class ArmbenchSegDataset(Beit3baseDataset):
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
        if args.task in ['seg']:
            self.items = np.load(os.path.join(data_path, f'{split}_annotation.npy'), allow_pickle=True)
            # with open(os.path.join(data_path, f'{split}_data.json'), 'r') as f:
            #     self.items = json.load(f)

    def _get_seg_sample(self, index, data):

        filename = os.path.join(self.data_path, 'images',
                                self.items[index][1]["image"])

        height, width = cv2.imread(filename).shape[:2]

        data["file_name"] = filename
        data["image_id"] = index
        data["height"] = height
        data["width"] = width


        objs = []
        for _, anno in self.items[index].items():
            # anno = anno["shape_attributes"]
            # px = anno["all_points_x"]
            # py = anno["all_points_y"]

            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = anno['annotations']['points']
            poly = [p + 0.5 for x in poly for p in x]

            obj = {
                "bbox": self.items[index]['annotations']['bbox'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)

        data["annotations"] = objs

        return data


    def _get_image_text_example(self, index, data):
        if self.args.task in ['seg']:
            self._get_seg_sample(index, data)
            return data

            # img_path = self.items['images'][index]["file_name"]
            # box = self.items[index]['box']
            # if not img_path.endswith('.jpg'):
            #     img_path = img_path + '.jpg'
            # img_path = os.path.join(self.data_path, img_path)
            # if box != [0,0,0,0]:
            #     box = [int(x) for x in box]
            #     img = Image.open(img_path).convert("RGB").crop(box)
            # else:
            #     img = Image.open(img_path).convert("RGB")
            #
            # img = self.transform(img)
            # data["image"] = img
            # data["label"] = self.items[index]['label']

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        if self.args.task in ['defection1by1']:
            return len(self.items)





def ArmbenchSegDataset(data_path, split):

    items = np.load(os.path.join(data_path, f'{split}_annotation.npy'), allow_pickle=True)

    dataset_dicts = []

    for index in range(len(items)):
        data = dict()
        filename = os.path.join(data_path, 'images',
                                items[index][1]["image"])

        height, width = cv2.imread(filename).shape[:2]

        data["file_name"] = filename
        data["image_id"] = index
        data["height"] = height
        data["width"] = width


        objs = []
        for _, anno in items[index].items():
            # anno = anno["shape_attributes"]
            # px = anno["all_points_x"]
            # py = anno["all_points_y"]

            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = anno['annotations']['points']
            poly = [p + 0.5 for x in poly for p in x]

            obj = {
                "bbox": items[index]['bbox'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)

        data["annotations"] = objs

        dataset_dicts.append(data)

    return dataset_dicts

DatasetCatalog.register("ArmbenchSegDataset_train", ArmbenchSegDataset('../dataset/armbench/armbench-segmentation-0.1/mix-object-tote', 'train'))
DatasetCatalog.register("ArmbenchSegDataset_val", ArmbenchSegDataset('../dataset/armbench/armbench-segmentation-0.1/mix-object-tote', 'val'))
DatasetCatalog.register("ArmbenchSegDataset_test", ArmbenchSegDataset('../dataset/armbench/armbench-segmentation-0.1/mix-object-tote', 'test'))
