import yaml

class Config:
    def __init__(self, init_map):
        for k, v in init_map.items():
            if isinstance(v, dict):
                instance = Config(v)
                init_map[k] = instance
        self.__dict__.update(init_map)

def get_config(path):
    with open(path, "r") as config_file:
        config_map = yaml.load(config_file.read(), Loader=yaml.FullLoader)
        return Config(config_map)