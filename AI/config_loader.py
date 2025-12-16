import os
import yaml
import argparse

def load_config(args=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(base_dir, 'model.yaml')

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"model.yaml not found at {yaml_path}")

    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    default_model_name = yaml_config.get('default_model', 'nano')

    if args is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--model', type=str, default=default_model_name, help='Name of the model configuration to use')
        known_args, _ = parser.parse_known_args()
        model_name = known_args.model
    else:
        model_name = args.model if hasattr(args, 'model') else default_model_name

    if model_name not in yaml_config:
        raise ValueError(f"Model configuration '{model_name}' not found in model.yaml")

    config = yaml_config[model_name]
    config['model_name'] = model_name
    
    return config
