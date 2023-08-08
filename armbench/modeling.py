import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np
from beit3_tools import utils
from beit3_tools.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from beit3_tools.modeling_finetune import TwoLayerMLP, Pooler


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
