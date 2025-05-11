import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor


class DisReins(nn.Module):
    def __init__(
            self,
            num_layers: int,
            embed_dims: int,
            patch_size: int,
            token_length: int = 100,
            use_softmax: bool = True,
            scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.token_length = token_length
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.create_model()

    def create_model(self):

        self.sty_learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )

        self.str_learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )

        self.sty_scale = nn.Parameter(torch.tensor(self.scale_init))
        self.str_scale = nn.Parameter(torch.tensor(self.scale_init))

        self.sty_mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.sty_mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)

        self.str_mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.str_mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )

        nn.init.uniform_(self.sty_learnable_tokens.data, -val, val)
        nn.init.uniform_(self.str_learnable_tokens.data, -val, val)

        nn.init.kaiming_uniform_(self.sty_mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.sty_mlp_token2feat.weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.str_mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.str_mlp_token2feat.weight, a=math.sqrt(5))

    def get_tokens(self, layer: int, token_type=None) -> Tensor:
        if layer == -1:
            if token_type == 'sty':
                return self.sty_learnable_tokens
            elif token_type == 'str':
                return self.str_learnable_tokens
        else:
            if token_type == 'sty':
                return self.sty_learnable_tokens[layer]
            elif token_type == 'str':
                return self.str_learnable_tokens[layer]

    def forward(
            self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        str_tokens = self.get_tokens(layer, 'str')
        str_delta_feat = self.str_forward_delta_feat(
            feats,
            str_tokens,
            layer,
        )
        str_delta_feat = str_delta_feat * self.str_scale
        str_feats = feats + str_delta_feat
        if has_cls_token:
            str_feats = torch.cat([cls_token, str_feats], dim=0)
        if batch_first:
            str_feats = str_feats.permute(1, 0, 2)
        return str_feats

    def forward_sty(
            self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        sty_tokens = self.get_tokens(layer, 'sty')
        sty_delta_feat = self.sty_forward_delta_feat(
            feats,
            sty_tokens,
            layer,
        )
        sty_delta_feat = sty_delta_feat * self.sty_scale
        sty_feats = feats + sty_delta_feat
        if has_cls_token:
            sty_feats = torch.cat([cls_token, sty_feats], dim=0)
        if batch_first:
            sty_feats = sty_feats.permute(1, 0, 2)
        return sty_feats

    def str_forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.str_mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.str_mlp_delta_f(delta_f + feats)
        return delta_f

    def sty_forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.sty_mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.sty_mlp_delta_f(delta_f + feats)
        return delta_f


class DisLoRAReins(DisReins):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.sty_learnable_tokens
        del self.str_learnable_tokens

        self.sty_learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.sty_learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )

        self.str_learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.str_learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )

        nn.init.uniform_(self.sty_learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.sty_learnable_tokens_b.data, -val, val)
        nn.init.uniform_(self.str_learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.str_learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer,token_type=None):
        if layer == -1:
            if token_type == 'sty':
                return self.sty_learnable_tokens_a @ self.sty_learnable_tokens_b
            elif token_type == 'str':
                return self.str_learnable_tokens_a @ self.str_learnable_tokens_b
        else:
            if token_type == 'sty':
                return self.sty_learnable_tokens_a[layer] @ self.sty_learnable_tokens_b[layer]
            elif token_type == 'str':
                return self.str_learnable_tokens_a[layer] @ self.str_learnable_tokens_b[layer]