import argparse

parser = argparse.ArgumentParser(description='Sketch-based OD')

parser.add_argument('--exp_name', type=str, default='default')

# --------------------
# DataLoader Options
# --------------------

parser.add_argument('--data_dir', type=str, default='/vol/research/sketchscene/sketch-OD/dataset')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--instance_level', action='store_true', default=False)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--bbox_scale', type=float, default=1.0)
parser.add_argument('--bbox_sigma', type=float, default=0.8)
parser.add_argument('--bbox_min_size', type=int, default=50)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Training Params
# ----------------------

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--workers', type=int, default=12)

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=1)

opts = parser.parse_args()
