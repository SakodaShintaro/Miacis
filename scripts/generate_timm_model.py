#!/usr/bin/env python3
import argparse
import torch
import torch.jit
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import timm


class TimmModel(nn.Module):
    def __init__(self, input_channel_num, block_num, channel_num, policy_channel_num, board_size):
        super(TimmModel, self).__init__()
        self.policy_dim_ = board_size * board_size * policy_channel_num
        num_classes = self.policy_dim_ + 51
        # self.body_ = VisionTransformer(
        #     depth=block_num,
        #     embed_dim=channel_num,
        #     img_size=9,
        #     patch_size=1,
        #     in_chans=input_channel_num,
        #     num_classes=num_classes)
        self.body_ = timm.create_model("efficientnetv2_s", pretrained=False, in_chans=input_channel_num, num_classes=num_classes)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.body_(x)
        policy, value = x.split(self.policy_dim_, 1)
        return policy, value


def main():
    print(timm.list_models())
    parser = argparse.ArgumentParser()
    parser.add_argument("-game", default="shogi", choices=["shogi", "othello", "go"])
    parser.add_argument("-value_type", default="cat", choices=["sca", "cat"])
    parser.add_argument("--block_num", type=int, default=12)
    parser.add_argument("--channel_num", type=int, default=384)
    args = parser.parse_args()

    if args.game == "shogi":
        input_channel_num = 42
        board_size = 9
        policy_channel_num = 27
    elif args.game == "othello":
        input_channel_num = 2
        board_size = 8
        policy_channel_num = 2
    else:
        exit(1)

    model = TimmModel(input_channel_num, args.block_num, args.channel_num, policy_channel_num, board_size)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"パラメータ数 : {params:,}")

    input_data = torch.randn([16, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    script_model = torch.jit.script(model)
    model_path = f"./{args.game}_cat_timm_bl{args.block_num}_ch{args.channel_num}.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")

    model = torch.jit.load(model_path)
    out = model(input_data)
    print(out[0].shape, out[1].shape)


if __name__ == "__main__":
    main()
