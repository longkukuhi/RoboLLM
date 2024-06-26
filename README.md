# [RoboLLM: Robotic Vision Tasks Grounded on Multimodal Large Language Models](https://arxiv.org/abs/2310.10221)

Official PyTorch implementation and pretrained models for paper:   "[RoboLLM: Robotic Vision Tasks Grounded on Multimodal Large Language Models.](https://arxiv.org/pdf/2310.10221.pdf)"

## Updates
- [2024/01/31] Accepted to ICRA 2024!
- [2024/02/15] Updated fine-tuned checkpoints for the identification and defection detection tasks.

## Setup

### Download pre-trained Checkpoints


   - [`BEiT3-base`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 222M
   - [`BEiT3-large`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D): #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16; #parameters: 674M

### Download Text Tokenizer

[beit3.spm](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm) is the sentencepiece model used for tokenizing texts.
```
from transformers import XLMRobertaTokenizer
tokenizer = XLMRobertaTokenizer("/your_beit3_model_path/beit3.spm")

```

### Set up the environment

```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel bash
```

install required packages:
```
pip install -r requirements.txt
```

### Download our preprossed json files

#### For the Armbench identification task
Download this file and put them into the armbench dataset dir.
[json_files.zip](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EdUm-c9sJwpPp4Ir-ve7xjoBVBdGtXuyy4S3nz9RJHbFkA?e=3mL2U0)

Additional json files for 3to1 task.
[ID_json_3t1.zip](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EUnQKxyYDPRMjJDl8Lq4u5UB4CNqF_4HYIe8q9yPpBa_Qg?e=PbM1nl)
<!-- #### For the Armbench identification task -->




### (Optional) Download our fine-tuned checkpoints

#### For the Armbench identification task
[RoboLLM Base whole gallary](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EdMk_of-cipEhngncnskodYBQTTmQ2q_eiENc5rx95q1tA?e=jOzm12)

[RoboLLM Base within basket](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EZMjZt--T4JLv6LJtAJKYZABFDWj2oOyzxLqDe-y2rS_VQ?e=EX1CeV)


#### For the Armbench defection detection task
[RoboLLM Base](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EWkqfuv_35FPlRlL9ztEKTYB9kmjsdncDHo9DwLrUkoOiQ?e=w8I70c)

[RoboLLM Large](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/ESQKOIKn2RhHhT39_aufJRwB5osYBvHhYshGICHPlu7r_A?e=ksQDa1)


## Object Identification
```bash
python armbench/ID.py --model 'beit3_base_patch16_224' --input_size 224 --task 'armbenchpick1to1' --batch_size 128 \
 --layer_decay 0.65 --lr 2e-4 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model 'beit3.spm' \
 --data_path 'path/to/your/dataset' --output_dir 'your_output_path/' --log_dir '/your_log_path/' --weight_decay 0.05  \
 --save_ckpt_freq 1 --finetune 'path/to/ckpt/beit3_base_patch16_224.pth'
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics using armbenchpick1to1 for only use pre-pick images, armbench3t1 for use both pre-pick and post-pick images. 
- `data_path` is the folder dir that stores the datasets.
- `finetune` specifics the dir to pretrained weight of BEiT-3 model.

## Defect Detection
```bash
python armbench/defection_id.py --model 'beit3_base_patch16_224' --input_size 224 --task 'defection1by1' --batch_size 128 \
 --layer_decay 0.65 --lr 2e-4 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model 'beit3.spm' \
 --data_path 'path/to/your/dataset'  --output_dir 'your_output_path/' --log_dir '/your_log_path/' --weight_decay 0.05  \
 --save_ckpt_freq 1 --finetune 'path/to/ckpt/beit3_base_patch16_224.pth'
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics for defect detection.
- `data_path` is the folder dir that stores the datasets.
- `finetune` specifics the dir to pretrained weight of BEiT-3 model.

## Citation
If you find this repository useful, please consider citing our work:
```
@misc{long2023robollm,
      title={RoboLLM: Robotic Vision Tasks Grounded on Multimodal Large Language Models}, 
      author={Zijun Long and George Killick and Richard McCreadie and Gerardo Aragon Camarasa},
      year={2023},
      eprint={2310.10221},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

<!-- ## Todo
- . -->



## Acknowledgement

This repository is built using the [BEiT](https://github.com/microsoft/unilm/tree/master/beit), the [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit2), the [BEiTv3](https://github.com/microsoft/unilm/tree/master/beit3), the [CLIP](https://github.com/openai/CLIP), the [open_clip](https://github.com/mlfoundations/open_clip), the [Oscar](https://github.com/microsoft/Oscar), the [DeiT](https://github.com/facebookresearch/deit), the [Dino](https://github.com/facebookresearch/dino) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

