from deeplearning.models.simple_reins.reins import LoRAReins
from deeplearning.models.simple_reins.peft import set_requires_grad, set_train, get_pyramid_feature
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class RETFound(VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super().__init__(
            global_pool=global_pool,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs
        )


class ReduceChannels(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ReduceChannels, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=1,
                              padding=0)

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        # nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x


class SegHead(nn.Module):
    '''
    f1 -- Size/4
    f2 -- Size/8
    f3 -- Size/16
    f4 -- Size/32
    '''

    def __init__(self, num_classes):
        super(SegHead, self).__init__()
        channels = [64, 128, 256, 512]

        self.reduce_channels_1024_512 = ReduceChannels(1024, 512)
        self.reduce_channels_1024_256 = ReduceChannels(1024, 256)
        self.reduce_channels_1024_128 = ReduceChannels(1024, 128)
        self.reduce_channels_1024_64 = ReduceChannels(1024, 64)
        self.decode4 = Decoder(channels[3], channels[2])
        self.decode3 = Decoder(channels[2], channels[1])
        self.decode2 = Decoder(channels[1], channels[0])
        # self.decode1 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        #                              nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
        #                              nn.ReLU(inplace=True))
        # self.decode0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                              nn.Conv2d(32, num_classes, kernel_size=1, bias=False))
        self.decode0 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                     nn.Conv2d(64, num_classes, kernel_size=1, bias=False))

    def forward(self, features):
        f1, f2, f3, f4 = features
        f4_512 = self.reduce_channels_1024_512(f4)
        f3_256 = self.reduce_channels_1024_256(f3)
        f2_128 = self.reduce_channels_1024_128(f2)
        f1_64 = self.reduce_channels_1024_64(f1)

        out = self.decode4(f4_512, f3_256)
        out = self.decode3(out, f2_128)
        out = self.decode2(out, f1_64)
        # out = self.decode1(out)
        out = self.decode0(out)
        return out

class ReinsRETFound(RETFound):
    def __init__(self,
                 reins_config=None,
                    **kwargs,
                 ):
        super().__init__(**kwargs)
        self.reins: LoRAReins = LoRAReins(**reins_config)
        self.patch_size = 16
        self.out_indices=[7, 11, 15, 23]
    '''
        def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    '''
    def forward(self, x):
        B, _, h, w = x.shape
        # H, W = h // self.patch_size, w // self.patch_size (patch_size = 16)
        H, W = h // self.patch_size, w // self.patch_size

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return get_pyramid_feature(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])


class Reins_SegHead(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super(Reins_SegHead, self).__init__()
        checkpoint_path ="RETFound_cfp_weights.pth"
        self.rein_RETFound = ReinsRETFound(reins_config=dict(
            token_length=100,
            embed_dims=1024,
            num_layers=24,
            patch_size=16,
            lora_dim=16, ),
            img_size=224,
            init_values=1.0e-5,
            proj_bias=True,)
        self.rein_RETFound.load_state_dict(torch.load(checkpoint_path, 'cpu'),strict=False)
        print("Load checkpoint rein_RETFound successfully!")
        self.seg_head = SegHead(num_classes=num_classes)



    def forward(self, x):
        features = self.rein_RETFound(x)
        output = self.seg_head(features)
        return output


if __name__ == '__main__':
    input_tensor = torch.randn(4, 3, 224, 224).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Reins_SegHead().to(device)
    output = model(input_tensor)
    print(f"output.shape:{output.shape}")
    # print(f"len(output):{len(output)}")
    # for i in range(len(output)):
    #     print(f"{i} -- {output[i].shape}")

    '''
    h=512,w=512,x.shape=torch.Size([4, 1025, 1024]),pos_embed.shape=torch.Size([1, 1025, 1024])
    len(output):4
    0 -- torch.Size([4, 1024, 128, 128])
    1 -- torch.Size([4, 1024, 64, 64])
    2 -- torch.Size([4, 1024, 32, 32])
    3 -- torch.Size([4, 1024, 16, 16])
    '''


