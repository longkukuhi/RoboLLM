import math
import os.path
import sys
import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from beit3_tools.beit3_datasets import get_sentencepiece_model_for_beit3
import numpy as np
from beit3_tools import utils
from tqdm import tqdm
import os


class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class ArmbenchHandler(object):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None


    def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None):
        # calculate query and ref features
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }


    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  query_images=None,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

    def calculate_recall_at_k(self, topk, labels, ref_item_ids, k=10):
        correct = 0.0
        for i in range(len(labels)):
            if labels[i] in [ref_item_ids[index] for index in topk[i].tolist()]:
                correct += 1.0

        return correct / len(labels)

    def calculate_accuracy_at_k(self, topk, labels, ref_item_ids, k=10):
        correct = 0.0
        from collections import Counter
        for i in range(len(labels)):
            predict_label = Counter([ref_item_ids[index] for index in topk[i].tolist()]).most_common(1)[0][0]
            if predict_label == labels[i]:
                correct += 1.0

        return correct / len(labels)

    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0))  # .to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0))  # .to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()

        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))

        # topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk3 = scores.topk(3, dim=1)
        topk2 = scores.topk(2, dim=1)
        topk1 = scores.topk(1, dim=1)

        # topk10_iids = refids[topk10.indices]
        topk5_iids = refids[topk5.indices]
        topk3_iids = refids[topk3.indices]
        topk2_iids = refids[topk2.indices]
        topk1_iids = refids[topk1.indices]

        # topk10_iids = topk10_iids.detach().cpu()
        topk5_iids = topk5_iids.detach().cpu()
        topk3_iids = topk3_iids.detach().cpu()
        topk2_iids = topk2_iids.detach().cpu()
        topk1_iids = topk1_iids.detach().cpu()

        # tr_r10 = self.calculate_recall_at_k(topk10_iids, labels, ref_item_ids, k=10)
        tr_r5 = self.calculate_recall_at_k(topk5_iids, labels, ref_item_ids, k=5)
        tr_r3 = self.calculate_recall_at_k(topk3_iids, labels, ref_item_ids, k=3)
        tr_r2 = self.calculate_recall_at_k(topk2_iids, labels, ref_item_ids, k=2)
        tr_r1 = self.calculate_recall_at_k(topk1_iids, labels, ref_item_ids, k=1)

        # acc_r10 = self.calculate_accuracy_at_k(topk10_iids, labels, ref_item_ids, k=10)
        acc_r5 = self.calculate_accuracy_at_k(topk5_iids, labels, ref_item_ids, k=5)
        acc_r3 = self.calculate_accuracy_at_k(topk3_iids, labels, ref_item_ids, k=3)
        acc_r2 = self.calculate_accuracy_at_k(topk2_iids, labels, ref_item_ids, k=2)
        acc_r1 = self.calculate_accuracy_at_k(topk1_iids, labels, ref_item_ids, k=1)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,

            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"


class Armbench3t1Handler(TaskHandler):
    def __init__(self,) -> None:
        super().__init__()
        self.query_image_feats = []
        self.ref_image_feats = []
        self.text_feats = []
        self.pick_ids = []
        self.ref_ids = []
        self.metric_logger = None
        # with open(os.path.join(data_path, 'Picks_splits', 'train_label_dict.json',  'r')) as f:
        #     self.train_labels = json.load(f)
        #
        # with open(os.path.join(data_path, 'Picks_splits', 'test_label_dict.json',  'r')) as f:
        #     self.test_labels = json.load(f)

    # def before_train(self):
    #     self.ref_image_feats.clear()
    #     self.ref_ids.clear()

    # def build_ref_feats(self, model, ref_image, ref_id, language_tokens=None, padding_mask=None):
    #     _, ref_vision_cls = model(query_images=None, ref_image=ref_image)
    #     self.ref_image_feats.append(ref_vision_cls.clone())
    #     self.ref_ids.append(ref_id)


    # def train_batch(self, model, query_images, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
    #     # calculate query and ref features
    #     pick_images = [data['image0'], data['image1'], data['image2']]
    #     loss, _, _ = model(query_images=query_images, ref_image=ref_image)
    #     # calculate cross entropy loss
    #
    #     return {
    #         "loss": loss,
    #     }

    def train_batch(self, model, image0, image1, image2, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        query_images = [image0, image1, image2]
        loss, _, _ = model(query_images=query_images, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }



    def before_eval(self, metric_logger, **kwargs):
        self.query_image_feats.clear()
        self.ref_image_feats.clear()
        self.text_feats.clear()
        self.pick_ids.clear()
        self.ref_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model,  image0, image1, image2,  ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = [image0, image1, image2]

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())




    def build_rank(self, data_loader, topk, values, query_data_type, k=1000):
        all_rank = []
        topk = topk.detach().cpu() #.reshape(-1, k)
        values = values.detach().cpu() #.reshape(-1, k)

        if query_data_type == 'img':
            retrieval_type = 'text'
        elif query_data_type == 'text':
            retrieval_type = 'img'

        print(f"Build rank list for {query_data_type} to {retrieval_type}")

        for idx in tqdm(range(topk.shape[0])):
            if query_data_type == 'img':
                item_id = data_loader.dataset._get_img_id(idx)
            elif query_data_type == 'text':
                item_id = data_loader.dataset._get_text_id(idx)

            rank_list = topk[idx].tolist()
            # transfer rank idx to item id
            if retrieval_type == 'img':
                rank_list = data_loader.dataset._get_img_id(rank_list)
            elif retrieval_type == 'text':
                rank_list = data_loader.dataset._get_text_id(rank_list)

            all_rank.append({'query_id': item_id,
                            'rank': rank_list,
                            'scores': values[idx].tolist()})

        return all_rank

    
    def calculate_recall_at_k(self, topk, labels, ref_item_ids, k=10):
        correct = 0.0
        for i in range(len(labels)):
            if labels[i] in [ref_item_ids[index] for index in topk[i].tolist()]:
                correct += 1.0

        return correct / len(labels)
    
    def calculate_accuracy_at_k(self, topk, labels, ref_item_ids, k=10):
        correct = 0.0
        from collections import Counter
        for i in range(len(labels)):
            predict_label = Counter([ref_item_ids[index] for index in topk[i].tolist()]).most_common(1)[0][0]
            if predict_label == labels[i]:
                correct += 1.0

        return correct / len(labels)



    def after_eval(self, query_dataloader, ref_dataloader, build_ranking=False, **kwargs):

        # query_image_feats = {}
        # for feats, ids in zip(self.query_image_feats, self.pick_ids):
        #     for i, _idx in enumerate(ids):
        #         idx = _idx.item()
        #         if idx not in query_image_feats:
        #             query_image_feats[idx] = feats[i]
        #
        pickids = torch.cat(self.pick_ids, dim=0)
        refids = torch.cat(self.ref_ids, dim=0)
        # iids = []
        # sorted_tensors = []
        # for key in sorted(query_image_feats.keys()):
        #     sorted_tensors.append(query_image_feats[key].view(1, -1))
        #     iids.append(key)
        #
        query_cls_feats = torch.Tensor.float(torch.cat(self.query_image_feats, dim=0)) #.to('cuda')
        ref_cls_feats = torch.Tensor.float(torch.cat(self.ref_image_feats, dim=0)) #.to('cuda')

        labels = query_dataloader.dataset._get_class_id(pickids.tolist())
        ref_item_ids = ref_dataloader.dataset._get_item_id(refids.tolist())

        scores = query_cls_feats @ ref_cls_feats.t()


        pickids = torch.LongTensor(pickids).to(scores.device)
        refids = torch.LongTensor(refids).to(scores.device)
        # labels = labels.to(scores.device)
        # ref_item_ids = ref_item_ids.to(scores.device)

        # iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("pickids: {}".format(pickids.size()))
        print("refids: {}".format(refids.size()))



        # topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk3 = scores.topk(3, dim=1)
        topk2 = scores.topk(2, dim=1)
        topk1 = scores.topk(1, dim=1)

        # topk10_iids = refids[topk10.indices]
        topk5_iids = refids[topk5.indices]
        topk3_iids = refids[topk3.indices]
        topk2_iids = refids[topk2.indices]
        topk1_iids = refids[topk1.indices]

        # topk10_iids = topk10_iids.detach().cpu()
        topk5_iids = topk5_iids.detach().cpu()
        topk3_iids = topk3_iids.detach().cpu()
        topk2_iids = topk2_iids.detach().cpu()
        topk1_iids = topk1_iids.detach().cpu()

        # tr_r10 = self.calculate_recall_at_k(topk10_iids, labels, ref_item_ids, k=10)
        tr_r5 = self.calculate_recall_at_k(topk5_iids, labels, ref_item_ids, k=5)
        tr_r3 = self.calculate_recall_at_k(topk3_iids, labels, ref_item_ids, k=3)
        tr_r2 = self.calculate_recall_at_k(topk2_iids, labels, ref_item_ids, k=2)
        tr_r1 = self.calculate_recall_at_k(topk1_iids, labels, ref_item_ids, k=1)
        
        # acc_r10 = self.calculate_accuracy_at_k(topk10_iids, labels, ref_item_ids, k=10)
        acc_r5 = self.calculate_accuracy_at_k(topk5_iids, labels, ref_item_ids, k=5)
        acc_r3 = self.calculate_accuracy_at_k(topk3_iids, labels, ref_item_ids, k=3)
        acc_r2 = self.calculate_accuracy_at_k(topk2_iids, labels, ref_item_ids, k=2)
        acc_r1 = self.calculate_accuracy_at_k(topk1_iids, labels, ref_item_ids, k=1)

        eval_result = {
            "tr_r1": tr_r1 * 100.0,
            "tr_r2": tr_r2 * 100.0,
            "tr_r3": tr_r3 * 100.0,
            "tr_r5": tr_r5 * 100.0,

            # "tr_r10": tr_r10 * 100.0,
            
            "acc_r1": acc_r1 * 100.0,
            "acc_r2": acc_r2 * 100.0,
            "acc_r3": acc_r3 * 100.0,
            "acc_r5": acc_r5 * 100.0,
            # "acc_r10": acc_r10 * 100.0,

            "average_score": (tr_r1 + tr_r2 + tr_r3 + tr_r5 + acc_r1 + acc_r2 + acc_r3 + acc_r5) / 8.0
        }

        print('* Eval result = %s' % json.dumps(eval_result))
        return eval_result, "average_score"

        # if build_ranking:
        #     return eval_result, "average_score", text_to_image_rank, image_to_text_rank
        #
        # else:
        #     return eval_result, "average_score"


class ArmbenchPick1Handler(ArmbenchHandler):
    def __init__(self, ) -> None:
        super().__init__()

    def train_batch(self, model, pick_image, ref_image, pick_id, language_tokens=None, padding_mask=None, **kwargs):
        # calculate query and ref features
        loss, _, _ = model(query_images=pick_image, ref_image=ref_image)
        # calculate cross entropy loss

        return {
            "loss": loss,
        }

    def eval_batch(self, model, pick_image=None, ref_image=None, pick_id=None, ref_id=None, padding_mask=None):
        query_images = pick_image

        if query_images is not None and ref_image is not None:
            query_vision_cls, ref_vision_cls = model(query_images=query_images, ref_image=ref_image, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            # self.text_feats.append(language_cls.clone())
            self.pick_ids.append(pick_id.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())

        elif query_images is not None and ref_image is None:
            query_vision_cls, _ = model(query_images=query_images, only_infer=True)
            self.query_image_feats.append(query_vision_cls.detach().cpu())
            self.pick_ids.append(pick_id.detach().cpu())

        elif query_images is None and ref_image is not None:
            _, ref_vision_cls = model(ref_image=ref_image, only_infer=True)
            self.ref_image_feats.append(ref_vision_cls.detach().cpu())
            self.ref_ids.append(ref_id.detach().cpu())





def get_handler(args):
    if args.task == "armbench3t1":
        return Armbench3t1Handler()

    elif args.task == "armbenchpick1":
        return ArmbenchPick1Handler()

    else:
        raise NotImplementedError("Sorry, %s is not support." % args.task)




def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: TaskHandler, epoch: int, start_steps: int,
        lr_schedule_values: list, loss_scaler, max_norm: float = 0,
        update_freq: int = 1, model_ema: Optional[ModelEma] = None,
        log_writer: Optional[utils.TensorboardLogger] = None,
        task=None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            # print("input %s = %s" % (tensor_key, data[tensor_key]))
            if loss_scaler is None and tensor_key.startswith("image"):
                data[tensor_key] = data[tensor_key].half()

        # pick_images = [data['image0'], data['image1'], data['image2']]
        #
        # if loss_scaler is None:
        #     results = handler.train_batch(model, query_images=pick_images,
        #                                   ref_image=data["ref_image"], pick_id=data["pick_id"])
        # else:
        #     with torch.cuda.amp.autocast():
        #         results = handler.train_batch(model, query_images=pick_images,
        #                                   ref_image=data["ref_image"], pick_id=data["pick_id"])
        if loss_scaler is None:
            results = handler.train_batch(model, **data)
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model, **data)

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value,
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value,
                "lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay_value,
                "grad_norm": grad_norm,
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(query_dataloader, answer_dataloader, model, device, handler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # prepare query data
    # if args.retrieval_mode == 'text_to_image':

    # elif args.retrieval_mode == 'image_to_text':
    #     querys = load_dataset(args.dataset_url, split='train',
    #                                  num_proc=4)

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger)

    # build query embeddings

    for data in tqdm(query_dataloader):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        #     images = [data['image0'], data['image1'], data['image2']]
        #     handler.eval_batch(model=model, query_images=images, pick_id=data["pick_id"])
        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # if args.load_embeddings_from_npy:
    #     raise NotImplementedError
        # # print(sorted(os.listdir(args.embeddings_file_path)))
        # freq = 3000
        # paths = [f'image_feats_{pointer}_freq_3000_gpu_0.npy' for pointer in range(0, 168001, freq)]
        # for file in paths:
        #     if file.endswith('.npy'):
        #         handler.query_image_feats.append(np.load(os.path.join(args.embeddings_file_path, file)))
        #
        # handler.query_image_feats = np.concatenate(handler.query_image_feats, axis=0)
        # handler.query_image_feats = torch.from_numpy(handler.query_image_feats)
        # handler.pick_ids = torch.arange(handler.query_image_feats.shape[0])
    # else:
    for data in tqdm(answer_dataloader):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

            # if len(handler.query_image_feats) % handler.store_feq == 0:
            #     if args.dist_eval:
            #         handler.store_feats(mode='image', tag=args.model+'_'+args.finetune.split('/')[-1], gpu_id= args.gpu)
            #     else:
            #         handler.store_feats(mode='image', tag=args.model+'_'+args.finetune.split('/')[-1], gpu_id= 0)
            #     handler.store_pointer += handler.store_feq


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return handler.after_eval(query_dataloader, answer_dataloader, args)
