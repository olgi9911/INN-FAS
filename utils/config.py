import yaml
import argparse
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration Loader")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--device', default = 'cuda:0', help = 'Device to use (cpu/cuda:0/cuda:1)')
    parser.add_argument('--ckpt', type = str, default = None, help = 'Path to checkpoint file')
    parser.add_argument('--op_dir', type = str, default = None, help = 'Directory to save checkpoints')
    parser.add_argument('--source', type = str, default = None, help = 'source dataset')
    parser.add_argument('--target', type = str, default = None, help = 'target dataset')
    parser.add_argument('--save_ckpt', action='store_true', help = 'Whether to save checkpoints during training')
    parser.add_argument('--seed', type = int, default=42, help = 'Random seed for reproducibility (used with --randomize)')

    args = parser.parse_args()
    return args

def load_config():
    args = parse_args()
    config_path = args.config

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['device'] = args.device
    
    if args.ckpt is not None:
        config['train']['ckpt'] = args.ckpt
    if args.op_dir is not None:
        config['train']['output_dir'] = args.op_dir
    if args.source is not None:
        config['dataset']['source'] = args.source
    if args.target is not None:
        config['dataset']['target'] = args.target
    
    config['train']['save_ckpt'] = args.save_ckpt
    config['train']['seed'] = args.seed

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['name'] = f"CNC_{config['dataset']['source']}_to_{config['dataset']['target']}_{timestamp}"
    
    return config