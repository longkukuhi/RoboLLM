# from datasets import load_dataset
# from tqdm import tqdm
# import numpy as np
# import os
#
#
# split = 'train'
# images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split=split)
#
# qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split)
# id2pos = {}
# for pos, _id in tqdm(enumerate(images['image_id']), total=len(images['image_id'])):
#     id2pos[_id] = pos
#
#
#
# image_ids = set(qrels["image_id"])
# valid_image_ids = []
# for image_id in tqdm(qrels["image_id"], total=len(qrels["image_id"])):
#     valid_image_ids.append({'image': images[id2pos[image_id]]['image']})
# np.save(os.path.join("../datasets/ATOMIC", f"{split}_images.npy"), valid_image_ids)
#
# split = 'validation'
# qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split)
# image_ids = set(qrels["image_id"])
#
# valid_image_ids = []
# for image_id in tqdm(qrels["image_id"], total=len(qrels["image_id"])):
#     valid_image_ids.append({'image': images[id2pos[image_id]]['image']})
# np.save(os.path.join("../datasets/ATOMIC", f"{split}_images.npy"), valid_image_ids)
#
# split = 'test'
# qrels = load_dataset("TREC-AToMiC/AToMiC-Qrels-v0.2", split=split)
# image_ids = set(qrels["image_id"])
#
# valid_image_ids = []
# for image_id in tqdm(qrels["image_id"], total=len(qrels["image_id"])):
#     valid_image_ids.append({'image': images[id2pos[image_id]]['image']})
#
# np.save(os.path.join("../datasets/ATOMIC", f"{split}_images.npy"), valid_image_ids)
import os

from datasets import load_dataset


# dataset_image = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train', cache_dir='../../datasets/ATOMIC/', num_proc=4)
dataset_image = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train', num_proc=4)

# image_ids = set(dataset_image["image_id"])



from tqdm.auto import tqdm

for idx in tqdm(range(7749495, len(dataset_image))):
    instance = dataset_image[idx]
    # instance['image'].save(f"\\\\longkukunas\\PhD data\\datasets\\ATOMIC\\images\\all\\{instance['image_id']}.jpg")
    instance['image'].save(f"..\\..\\datasets\\ATOMIC\\all_images\\{instance['image_id']}.jpg")

