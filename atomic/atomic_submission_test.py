# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from beit3_tools.optim_factory import create_optimizer, get_parameter_groups, \
    LayerDecayValueAssigner, get_is_head_flag_for_vit

from beit3_tools.engine_for_finetuning import get_handler, evaluate_submission
from beit3_tools.beit3_datasets import create_query_answer_dataset
from beit3_tools.utils import NativeScalerWithGradNormCount as NativeScaler
from beit3_tools import utils
import datasets
import beit3_tools.modeling_finetune
import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--task', type=str, required=True,
                        choices=['nlvr2', 'vqav2', 'flickr30k', 'coco_retrieval', 'coco_captioning', 'nocaps',
                                 'imagenet', 'atomic', 'atomic_submission'],
                        help='Name of task to fine-tuning')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--checkpoint_activations', action='store_true', default=None,
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--sentencepiece_model', type=str, required=True,
                        help='Sentencepiece model path for the pretrained model.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--task_head_lr_weight', type=float, default=0)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Augmentation parameters
    parser.add_argument('--randaug', action='store_true', default=False)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # parameter for dump predictions (VQA, COCO captioning, NoCaps)
    parser.add_argument('--task_cache_path', default=None, type=str)

    # parameter for imagenet finetuning
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # augmentation parameters for imagenet finetuning
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # evaluation parameters for imagenet
    parser.add_argument('--crop_pct', type=float, default=None)

    # random Erase params for imagenet finetuning
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # parameter for captioning finetuning
    parser.add_argument('--captioning_mask_prob', type=float, default=0.6)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--num_beams', type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=0.6)

    # label smoothing for imagenet and captioning
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # deepspeed parameters
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--initial_scale_power', type=int, default=16)
    parser.add_argument('--zero_stage', default=0, type=int,
                        help='ZeRO optimizer stage (default: 0)')

    # distributed parameters

    # wandb options
    parser.add_argument('--enable_wandb', action='store_true', default=False, )

    # dataloading options
    parser.add_argument('--load_tokenized_text', action='store_true', default=False)
    parser.add_argument('--load_image_from_huggingface_hub', action='store_true', default=False, )

    #
    parser.add_argument('--build_ranking', action='store_true', default=False, )
    parser.add_argument('--eval_on_test_set', action='store_true', default=False, )
    parser.add_argument('--query_url', type=str, default='dataset_url',
                        help='dataset name in huggingface')
    parser.add_argument('--retrieval_mode', type=str, default='text_to_image',
                        help='image_to_text or text_to_image')
    parser.add_argument('--run_id', type=str, default='base_pretrain', help='')
    parser.add_argument('--load_embeddings_from_npy', action='store_true', default=False, )
    parser.add_argument('--embeddings_file_path', type=str, default=None, help='')
    parser.add_argument('--cluster', action='store_true', default=False, )


    known_args, _ = parser.parse_known_args()
    if known_args.enable_deepspeed:
        raise Exception("Windows does not support DeepSpeed")
    else:
        ds_init = None

    return parser.parse_args(), ds_init

from tqdm import tqdm
import json

def write_ranking_result_into_trec_sytle(json_path, output_path, tag):
    # read data
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    with open(output_path, 'w') as f:
        for i in tqdm(range(len(data))):
            topic_id = list(data[i].keys())[0]
            rank, score = data[i][topic_id]['rank'], data[i][topic_id]['scores']
            for j in range(len(rank)):
                f.write(f'{topic_id} Q0 {rank[j]} {j} {score[j]} {tag}\n')
        print(f'Write ranking result into {output_path} successfully!')

def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    if args.task_cache_path is None:
        args.task_cache_path = args.output_dir

    print(args)


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # datasets.config.IN_MEMORY_MAX_SIZE = 32000000000
    print('Set dataset loading memory size to:', datasets.config.IN_MEMORY_MAX_SIZE)

    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None


    query_dataloader, answer_dataloader = create_query_answer_dataset(args)
    data_loader_train, data_loader_val = query_dataloader, answer_dataloader
    #  set output dir
    if args.eval:
        exp_tag = f"eval_{args.num_max_bpe_tokens}tokens_{''.join(data_loader_train.dataset.text_features)}_{args.finetune.split('/')[-1]}/"
    else:
        exp_tag = f"{args.model}_{args.epochs}epochs_{args.num_max_bpe_tokens}tokens_{''.join(data_loader_train.dataset.text_features)}_{args.finetune.split('/')[-1]}/"

    args.output_dir = os.path.join(args.output_dir, exp_tag)

    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    if not args.model.endswith(args.task):
        if args.task in ("flickr30k", "coco_retrieval", "atomic", 'atomic_submission'):
            model_config = "%s_retrieval" % args.model
        elif args.task in ("coco_captioning", "nocaps"):
            model_config = "%s_captioning" % args.model
        elif args.task in ("imagenet"):
            model_config = "%s_imageclassification" % args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model


    # print("model_config = %s" % model_config)
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
    )

    if args.finetune:
        utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)


    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train.dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(data_loader_train.dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)


    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        lrs = list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        assigner = LayerDecayValueAssigner(lrs)
    elif args.task_head_lr_weight > 1:
        assigner = LayerDecayValueAssigner([1.0, args.task_head_lr_weight], scale_handler=get_is_head_flag_for_vit)
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))


    skip_weight_decay_list = model.no_weight_decay()


    if args.distributed:
        torch.distributed.barrier()
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            print("Using torch.nn.parallel.DistributedDataParallel ... on GPU %d" % args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()


    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    task_handler = get_handler(args)

    # mixup for imagenet
    mixup_fn = None
    if args.task in ["imagenet", "in1k"]:
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=args.nb_classes)



    if args.task in ["nlvr2", "flickr30k", "coco_retrieval", "imagenet", 'atomic', 'atomic_submission']:

        rank_result \
        = evaluate_submission(query_dataloader, answer_dataloader, model, device, task_handler, args)

        json_file = f"output/{args.task}_{args.num_max_bpe_tokens}tokens_{''.join(query_dataloader.dataset.text_features)}_{args.model}_{args.finetune.split('/')[-1]}_test_results.json"

        rank_file = f"{args.task}_{args.num_max_bpe_tokens}tokens_{''.join(query_dataloader.dataset.text_features)}_{args.model}_{args.finetune.split('/')[-1]}_test_ranking.json"
        print("Write ranking results to %s" % json_file)

        if args.retrieval_mode == 'image_to_text':
            image_to_text_rank_file = args.output_dir + '/image_to_text_ranking_' + rank_file
            with open(image_to_text_rank_file, mode="w", encoding="utf-8") as writer:
                for key, value in rank_result.items():
                    writer.write(json.dumps({key: value}, indent=None))
                    # writer.write(json.dumps(image_to_text_rank, indent=None))
                    writer.write('\n')
            print("Write image to text ranking results to %s" % image_to_text_rank_file)

            # convert ranking results into trec style
            trec_rank_file_name = args.output_dir + '/image_to_text_ranking_' + f"{args.task}_{args.num_max_bpe_tokens}tokens_{''.join(query_dataloader.dataset.text_features)}_{args.model}_{args.finetune.split('/')[-1]}" + '_trec_submission' +'.txt'
            write_ranking_result_into_trec_sytle(image_to_text_rank_file, trec_rank_file_name, args.run_id)


        elif args.retrieval_mode == 'text_to_image':
            text_to_image_rank_file = args.output_dir + '/text_to_image_ranking_' + rank_file
            with open(text_to_image_rank_file, mode="w", encoding="utf-8") as writer:
                for key, value in rank_result.items():
                    writer.write(json.dumps({key: value}, indent=None))
                    writer.write('\n')
            print("Write text to image ranking results to %s" % text_to_image_rank_file)

            # convert ranking results into trec style
            rank_file_name = args.output_dir + '/text_to_image_ranking_' + f"{args.task}_{args.num_max_bpe_tokens}tokens_{''.join(query_dataloader.dataset.text_features)}_{args.model}_{args.finetune.split('/')[-1]}" + '_trec_submission' + '.txt'
            write_ranking_result_into_trec_sytle(text_to_image_rank_file, rank_file_name, args.run_id)

        else:
            raise ValueError("Invalid retrieval mode: %s" % args.retrieval_mode)


    else:
        raise ValueError("Invalid task: %s" % args.task)



if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
