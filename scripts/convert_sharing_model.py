#!/usr/bin/env python3
from generate_sharing_model import *

parser = argparse.ArgumentParser()
parser.add_argument("--source_model_path", type=str, required=True)
args = parser.parse_args()

source_model = torch.jit.load(args.source_model_path).cpu()
source_dict = source_model.state_dict()

input_channel_num = 42
board_size = 9
policy_channel_num = 27
channel_num = 256

for block_num in range(2, 21, 2):
    model = CategoricalNetwork(input_channel_num, block_num, channel_num, policy_channel_num, board_size)
    model.load_state_dict(source_dict)

    input_data = torch.ones([2, input_channel_num, board_size, board_size])
    script_model = torch.jit.trace(model, input_data)
    # script_model = torch.jit.script(model)
    model_path = f"./shogi_cat_bl{block_num}_ch256.model"
    script_model.save(model_path)
    print(f"{model_path}にパラメータを保存")
