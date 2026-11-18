# RoboLLM

RoboLLM uses BEiT-3 for two ARMBench tasks: identifying a picked object by retrieving reference images and detecting defects in images of objects. The paper was accepted at ICRA 2024. [Paper](https://arxiv.org/abs/2310.10221) · [PDF](https://arxiv.org/pdf/2310.10221).

## Choose a task and checkpoint

| Task | Input | Fine-tuned weights |
| --- | --- | --- |
| Object identification | Pick images, product references and split JSON files | [Base, whole gallery](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EdMk_of-cipEhngncnskodYBQTTmQ2q_eiENc5rx95q1tA?e=jOzm12) · [Base, within basket](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EZMjZt--T4JLv6LJtAJKYZABFDWj2oOyzxLqDe-y2rS_VQ?e=EX1CeV) |
| Defect detection | Images, bounding boxes and labels in the split arrays | [Base](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EWkqfuv_35FPlRlL9ztEKTYB9kmjsdncDHo9DwLrUkoOiQ?e=w8I70c) · [Large](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/ESQKOIKn2RhHhT39_aufJRwB5osYBvHhYshGICHPlu7r_A?e=ksQDa1) |

The authors released these fine-tuned checkpoints in February 2024. To start from an upstream encoder, choose [BEiT-3 Base](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) (12 layers, hidden size 768) or [BEiT-3 Large](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) (24 layers, hidden size 1024). The examples below use Base. The repository includes the [SentencePiece tokenizer](beit3.spm). An [upstream copy](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm) is also available.

## Prepare the environment and data

Use a CUDA PyTorch environment compatible with the original PyTorch 1.8.1/CUDA 11.1 release. The utilities import `torch._six`. Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The model module imports `SupConLoss` from `CL_tools.losses` when either task starts. This package is missing from the source tree and dependency list. Supply the matching module on the Python import path before running the examples below.

For identification, extract the [preprocessed split files](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EdUm-c9sJwpPp4Ir-ve7xjoBVBdGtXuyy4S3nz9RJHbFkA?e=3mL2U0) into the ARMBench data directory. Add the [3-to-1 split files](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EUnQKxyYDPRMjJDl8Lq4u5UB4CNqF_4HYIe8q9yPpBa_Qg?e=PbM1nl) for experiments combining pre-pick and post-pick views. The pick-only loader reads `all_<split>_data.json` and `all_labels.json`. Defect detection reads `<split>_data.npy` records containing `image`, `box` and `label`. Keep their referenced images under the data root. See the [identification loader](armbench/armbench_datasets.py) and [defect loader](armbench/defection_datasets.py).

## Run a task

Set `DATA` and `WEIGHTS` to the prepared dataset root and the matching BEiT-3 initialization checkpoint. Run these examples from the repository root.

```bash
DATA=/path/to/armbench
WEIGHTS=/path/to/beit3_base_patch16_224.pth

python -m armbench.ID --model beit3_base_patch16_224 --input_size 224 \
  --task armbenchpick1to1 --data_path "$DATA" --sentencepiece_model beit3.spm \
  --finetune "$WEIGHTS" --output_dir runs/id --log_dir logs/id \
  --batch_size 128 --layer_decay 0.65 --lr 2e-4 --epochs 30 \
  --warmup_epochs 3 --drop_path 0.2 --weight_decay 0.05 --save_ckpt_freq 1

python -m armbench.defection_id --model beit3_base_patch16_224 --input_size 224 \
  --task defection1by1 --data_path "$DATA" --sentencepiece_model beit3.spm \
  --finetune "$WEIGHTS" --output_dir runs/defect --log_dir logs/defect \
  --batch_size 128 --layer_decay 0.65 --lr 2e-4 --epochs 30 \
  --warmup_epochs 3 --drop_path 0.2 --weight_decay 0.05 --save_ckpt_freq 1
```

Use `armbench3t1` for multi-view identification. For evaluation, pass `--eval` and set `--finetune` to the fine-tuned checkpoint for the chosen task and model size.

## Citation
```bibtex
@misc{long2023robollm,
      title={RoboLLM: Robotic Vision Tasks Grounded on Multimodal Large Language Models}, 
      author={Zijun Long and George Killick and Richard McCreadie and Gerardo Aragon Camarasa},
      year={2023},
      eprint={2310.10221},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Acknowledgments

This repository is built using the [BEiT](https://github.com/microsoft/unilm/tree/master/beit), the [BEiTv2](https://github.com/microsoft/unilm/tree/master/beit2), the [BEiTv3](https://github.com/microsoft/unilm/tree/master/beit3), the [CLIP](https://github.com/openai/CLIP), the [open_clip](https://github.com/mlfoundations/open_clip), the [Oscar](https://github.com/microsoft/Oscar), the [DeiT](https://github.com/facebookresearch/deit), the [Dino](https://github.com/facebookresearch/dino) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


The original release points to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct). A repository-level license file is absent from the default branch.
