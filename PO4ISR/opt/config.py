import os
import yaml
from opt.parser import parse_args

def init_config():
    config = dict()

    current_path = os.path.dirname(os.path.realpath(__file__))
    basic_init_file = os.path.join(current_path, '../assets/overall.yaml')

    basic_conf = yaml.load(open(basic_init_file), Loader=yaml.loader.SafeLoader)
    config.update(basic_conf)

    args = parse_args()
    model_file = 'openai'
    model_init_file = os.path.join(current_path, f'../assets/{model_file}.yaml')
    model_conf = yaml.load(open(model_init_file), Loader=yaml.loader.SafeLoader)
    config.update(model_conf)

    args_conf = vars(args)

    for k, v in config.items():
        if k in args_conf.keys() and args_conf[k] is not None:
            config[k] = args_conf[k]
        else:
            config[k] = v
    return config
