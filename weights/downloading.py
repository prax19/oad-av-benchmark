import os
import sys
import yaml
import glob
import subprocess
import time
from pathlib import Path
import shutil

# TODO: add more OAD backbones to `models.yaml`

def _download_reference(reference: str, name: str, retries=3):
    # It sometimes just doesn't work because of certificates.
    # However there is no other "official" way so probably MMAction2 is just limited.
    # Once backbone is downloaded, `mim` wouldn't be triggered anymore.
    py = sys.executable
    cmd = [
        py, '-m', 'mim', 'download', 'mmaction2', 
        '--config', reference,
        '--dest', f'weights/backbones/{name}'
    ]

    for i in range(retries):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if i == retries - 1:
                raise
            print(f'Retries left: {retries}')
            time.sleep(5)

def load_by_key(key: str):
    backbones_root = Path('weights/backbones/')
    with open(Path(backbones_root, 'models.yaml'), 'r') as file:
        models = yaml.safe_load(file)
    if not key in models['backbones'].keys():
        raise FileNotFoundError(f'No backbone named {key}')
    
    backbone_dir = Path.joinpath(backbones_root, key)
    model_data = models['backbones'][key]

    if not backbone_dir.exists():
        os.makedirs(backbone_dir)

    config_glob = glob.glob(f'{backbone_dir}/*.py')
    weights_glob = glob.glob(f'{backbone_dir}/*.pth')
    if (not len(config_glob) == 1) or (not len(weights_glob) == 1):
        shutil.rmtree(backbone_dir, ignore_errors=True)
        _download_reference(model_data['reference'], key)
        config_glob = glob.glob(f'{backbone_dir}/*.py')
        weights_glob = glob.glob(f'{backbone_dir}/*.pth')
    if (len(config_glob) == 1) and (len(weights_glob) == 1):
        config = config_glob[0]
        weights = weights_glob[0]
        return config, weights
    else:
        raise FileNotFoundError(f"There is something wrong with `{backbone_dir}`")

x, y = load_by_key('tsn-kinetics-400')
print(f"{x}, {y}")