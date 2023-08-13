import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np
from beit3_tools import utils
from beit3_tools.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from beit3_tools.modeling_finetune import TwoLayerMLP, Pooler
from CL_tools.losses import SupConLoss


class BEiT3ForArmBench3t1Retrieval(BEiT3Wrapper):
    def __init__(
            self,
            args,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super(BEiT3ForArmBench3t1Retrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        # self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_pooler_0 =  Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.vision_pooler_1 = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.vision_pooler_2 = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )
        self.head = TwoLayerMLP(
            in_features=embed_dim * 3,
            hidden_features=embed_dim * 3,
            out_features=embed_dim,
            norm_layer=norm_layer,
        )

        self.ref_vision_pooler = Pooler(
            input_features=embed_dim,
            output_features=embed_dim,
            norm_layer=norm_layer,
        )

        # self.language_head.apply(self._init_weights)
        self.vision_pooler_0.apply(self._init_weights)
        self.vision_pooler_1.apply(self._init_weights)
        self.vision_pooler_2.apply(self._init_weights)
        self.head.apply(self._init_weights)

        self.criterion = utils.ClipLoss(
            rank=utils.get_rank(),
            world_size=utils.get_world_size(),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



    def forward(self, query_images=None, ref_image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if query_images is not None:
            image0 =query_images[0]
            image1 =query_images[1]
            image2 =query_images[2]

            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image0,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            vision_cls_0 = self.vision_pooler_0(x)

            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image1,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            vision_cls_1 = self.vision_pooler_1(x)

            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image2,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            vision_cls_2 = self.vision_pooler_2(x)

            query_vision_cls = self.head(torch.cat([vision_cls_0, vision_cls_1, vision_cls_2], dim=-1))
        else:
            query_vision_cls = None

        if ref_image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=ref_image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            ref_vision_cls = self.ref_vision_pooler(x)
        else:
            ref_vision_cls = None


        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None

        if only_infer:
            return query_vision_cls, ref_vision_cls
        else:
            loss, logits_per_query, logits_per_ref = self.criterion(
                query_vision_cls, ref_vision_cls, self.logit_scale.exp())

            return loss, query_vision_cls, language_cls


class BEiTForArmBenchRetrieval(BEiT3Wrapper):
    def __init__(
            self,
            args,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super(BEiTForArmBenchRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        # self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)

        # self.language_head.apply(self._init_weights)

        self.fc_norm = norm_layer(embed_dim)
        self.fc_norm.apply(self._init_weights)

        self.criterion = utils.ClipLoss(
            rank=utils.get_rank(),
            world_size=utils.get_world_size(),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, query_images=None, ref_image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if query_images is not None:
            image0 = query_images

            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image0,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            t = x[:, 1:, :]
            cls_x = self.fc_norm(t.mean(1))
            query_vision_cls = cls_x
        else:
            query_vision_cls = None

        if ref_image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=ref_image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            t = x[:, 1:, :]
            cls_x = self.fc_norm(t.mean(1))
            ref_vision_cls = cls_x
        else:
            ref_vision_cls = None


        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None

        if only_infer:
            return query_vision_cls, ref_vision_cls
        else:
            loss, logits_per_query, logits_per_ref = self.criterion(
                query_vision_cls, ref_vision_cls, self.logit_scale.exp())

            return loss, query_vision_cls, language_cls


class BEiTForArmbench(BEiT3Wrapper):
    def __init__(
            self,
            args,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        def __init__(
                self,
                args,
                norm_layer=nn.LayerNorm,
                **kwargs
        ):
            super(BEiTForArmBenchRetrieval, self).__init__(args=args)
            embed_dim = args.encoder_embed_dim
            # self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)

            # self.language_head.apply(self._init_weights)

            self.fc_norm = norm_layer(embed_dim)
            self.fc_norm.apply(self._init_weights)

            self.criterion = utils.ClipLoss(
                rank=utils.get_rank(),
                world_size=utils.get_world_size(),
            )
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        def forward(self, query_images=None, ref_image=None, text_description=None, padding_mask=None, only_infer=False,
                    **kwargs):
            if query_images is not None:
                image0 = query_images

                outputs = self.beit3(
                    textual_tokens=None,
                    visual_tokens=image0,
                    text_padding_position=None,
                )
                x = outputs["encoder_out"]
                t = x[:, 1:, :]
                cls_x = self.fc_norm(t.mean(1))
                query_vision_cls = cls_x
            else:
                query_vision_cls = None

            if ref_image is not None:
                outputs = self.beit3(
                    textual_tokens=None,
                    visual_tokens=ref_image,
                    text_padding_position=None,
                )
                x = outputs["encoder_out"]
                t = x[:, 1:, :]
                cls_x = self.fc_norm(t.mean(1))
                ref_vision_cls = cls_x
            else:
                ref_vision_cls = None

            if text_description is not None:
                outputs = self.beit3(
                    textual_tokens=text_description,
                    visual_tokens=None,
                    text_padding_position=padding_mask,
                )
                x = outputs["encoder_out"]
                language_cls = self.language_head(x[:, 0, :])
                language_cls = F.normalize(language_cls, dim=-1)
            else:
                language_cls = None

            if only_infer:
                return query_vision_cls, ref_vision_cls
            else:
                loss, logits_per_query, logits_per_ref = self.criterion(
                    query_vision_cls, ref_vision_cls, self.logit_scale.exp())

                return loss, query_vision_cls, language_cls


class BEiTForArmBenchCLloss(BEiT3Wrapper):
    def __init__(
            self,
            args,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super(BEiTForArmBenchCLloss, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        # self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)

        # self.language_head.apply(self._init_weights)

        self.args = args
        self.fc_norm = norm_layer(embed_dim)
        self.fc_norm.apply(self._init_weights)

        # self.criterion = SupConLoss(temperature=args.temp)
        #
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.criterion = utils.ClipLossOneWay(
            rank=utils.get_rank(),
            world_size=utils.get_world_size(),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, query_images=None, ref_image=None, ref_id=None,
                text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if query_images is not None:
            image0 = query_images
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image0,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            t = x[:, 1:, :]
            cls_x = self.fc_norm(t.mean(1))
            query_vision_cls = cls_x

        else:
            query_vision_cls = None

        if ref_image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=ref_image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            t = x[:, 1:, :]
            cls_x = self.fc_norm(t.mean(1))
            ref_vision_cls = cls_x
        else:
            ref_vision_cls = None


        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None

        if only_infer:
            return query_vision_cls, ref_vision_cls
        else:
            loss, logits_per_query, = self.criterion(
                query_vision_cls, ref_vision_cls, self.logit_scale.exp())
            # bsz = query_vision_cls.size(0)
            # features = torch.cat([query_vision_cls.unsqueeze(1), ref_vision_cls.unsqueeze(1)], dim=1)
            # # normalize the features for each sample
            # features = F.normalize(features, dim=-1)
            # if self.args.cl_loss == 'SupCon':
            #     labels = ref_id
            #     loss = self.criterion(features, labels)
            # elif self.args.cl_loss == 'SimCLR':
            #     loss = self.criterion(features)
            # else:
            #     raise ValueError('contrastive method not supported: {}'.
            #                     format(self.args.method))

            return loss, query_vision_cls, ref_vision_cls



class BEiTForArmBenchRetrievalNearestRef(BEiT3Wrapper):
    def __init__(
            self,
            args,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super(BEiTForArmBenchRetrievalNearestRef, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        # self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)

        # self.language_head.apply(self._init_weights)

        # self.fc_norm = norm_layer(embed_dim)
        # self.fc_norm.apply(self._init_weights)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head.apply(self._init_weights)
        self.ref_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ref_head.apply(self._init_weights)

        self.criterion = utils.ClipLossOneWay(
            rank=utils.get_rank(),
            world_size=utils.get_world_size(),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _copy_weights(self):
        self.ref_head.weight.data.copy_(self.vision_head.weight.data)

    def forward(self, query_images=None, ref_images=None, ref_image=None,
                text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if query_images is not None:
            image0 = query_images

            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image0,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            query_vision_cls =  self.vision_head(x[:, 0, :])
        else:
            query_vision_cls = None

        if ref_images is not None:
                ref_vision_cls = []
                for ref_images_per_query in ref_images:

                    ref_images = [ref_image.to('cuda', non_blocking=True).unsqueeze(0) for ref_image in ref_images_per_query]
                    ref_images = torch.cat(ref_images, dim=0)
                    outputs = self.beit3(
                        textual_tokens=None,
                        visual_tokens=ref_images,
                        text_padding_position=None,
                    )
                    x = outputs["encoder_out"]
                    cls_x = self.ref_head(x[:, 0, :])
                    ref_vision_cls.append(cls_x)

        elif ref_image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=ref_image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            cls_x = self.ref_head(x[:, 0, :])
            ref_vision_cls = cls_x

        else:
            ref_vision_cls = None



        if only_infer:
            return query_vision_cls, ref_vision_cls

        else:
            # choose the nearest ref base on l2 distance

            # normalize the features for each sample
            query_vision_cls = F.normalize(query_vision_cls, dim=-1)
            nearest_ref_cls = []
            for i in range(len(ref_vision_cls)):
                ref_vision_cls[i] = F.normalize(ref_vision_cls[i], dim=-1)
                # calulate the L2 distance between query and ref
                distances = torch.tensor([((query_vision_cls[i]-cls)**2).sum(axis=0)  for cls in ref_vision_cls[i]])
                # distances = [torch.cdist(query_vision_cls[i], cls) for cls in ref_vision_cls[i]]
                min_idx = torch.argmin(distances) #.tolist()
                nearest_ref_cls.append(ref_vision_cls[i][[min_idx]])
            nearest_ref_cls = torch.stack(nearest_ref_cls, dim=0)
            # distances = [torch.cdist(query_vision_cls, cls) for cls_list in ref_vision_cls for cls in cls_list]
            # nearest_ref_cls = ref_vision_cls[torch.argmin(distances)[0]]
            # ref_vision_cls = F.normalize(ref_vision_cls, dim=-1)


            loss, logits_per_query = self.criterion(
                query_vision_cls, nearest_ref_cls, self.logit_scale.exp())

            return loss, query_vision_cls




@register_model
def beit3_base_patch16_224_armbench3t1(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForArmBench3t1Retrieval(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_armbenchpick1(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiTForArmBenchRetrieval(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_armbenchpick1to1(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiTForArmBenchRetrieval(args, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_armbenchpick1_clloss(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiTForArmBenchCLloss(args, **kwargs)
    return model

@register_model
def beit3_base_patch16_224_armbenchpick1_nearestref(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiTForArmBenchRetrievalNearestRef(args, **kwargs)
    return model