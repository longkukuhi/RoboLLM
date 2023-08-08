import json
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

def _getRowIdx( id_list: list) -> dict:
    id2pos = {}
    for pos, _id in tqdm(enumerate(id_list), total=len(id_list)):
        id2pos[_id] = pos
    return id2pos

images = load_dataset("TREC-AToMiC/AToMiC-Images-v0.2", split='train',
                                       num_proc=4)
texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train',
                          num_proc=4)

# images = load_dataset('TREC-AToMiC/TREC-2023-Image-to-Text', split='train',
#                                      num_proc=4)
#
image_ids = images['image_id']
text_ids = texts['text_id']

# np.save('../datasets/ATOMIC/all_image_ids.npy', image_ids)
# np.save('../datasets/ATOMIC/all_text_ids.npy', text_ids)

image_id2row_dict = _getRowIdx(image_ids)
text_id2row_dict = _getRowIdx(text_ids)

text_ids = ['projected-00006678-002']
image_ids = ['44bcf02e-05fb-3556-abd7-dda52c821351', 'bda5ccaa-8ab1-3370-a65d-40dd733a06f7',
             '68724e89-d7c9-3b91-8b60-f24e13b3c594', 'd9ed5801-cf18-3930-ad29-caa3b54dbdfe']

from PIL import Image

for image_id in image_ids:
    images[image_id2row_dict[image_id]]['image'].show()

for text_id in text_ids:
    print('text_id:', text_id)
    print('page_title:', texts[text_id2row_dict[text_id]]['page_title'])
    print('section_title:', texts[text_id2row_dict[text_id]]['section_title'])
    print('context_section_description:', texts[text_id2row_dict[text_id]]['context_section_description'])
    print('-------------------------------------------------')