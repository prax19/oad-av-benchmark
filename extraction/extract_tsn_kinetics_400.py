import torch
from mmengine import Config
from mmengine.dataset import Compose
from mmaction.apis import init_recognizer
from pathlib import Path
import mmaction

from weights import load_by_key
from utils import torch_scripts

video = r"data/road/videos/2014-06-25-16-45-34_stereo_centre_02.mp4"
device = torch_scripts.get_device()

config_file, checkpoint_file = load_by_key('tsn-kinetics-400')

cfg = Config.fromfile(str(config_file))
model = init_recognizer(cfg, str(checkpoint_file), device=device).eval()

test_pipeline = Compose(cfg.test_dataloader.dataset.pipeline)

data = dict(
    filename=video,
    label=-1,
    start_index=0
)

data = test_pipeline(data)

data_batch = dict(
    inputs=[data['inputs']],
    data_samples=[data['data_samples']]
)

data_batch = model.data_preprocessor(data_batch, training=False)

inputs = data_batch['inputs']
data_samples = data_batch['data_samples']

with torch.no_grad():
    pred = model(inputs, data_samples, mode='predict')

    N, num_segs, C, H, W = inputs.shape
    backbone_input = inputs.view(N * num_segs, C, H, W)
    feats = model.backbone(backbone_input)

if isinstance(feats, (list, tuple)):
    feats = feats[-1]

# global average pooling to spatial
if feats.ndim == 4:              # [N*num_segs, C, H, W]
    feats = feats.mean(dim=[2, 3])   # -> [N*num_segs, C]

feats_video = feats.view(N, num_segs, -1).mean(dim=1)   # [N, C]

print("pred:", pred)
print("feats_video shape:", feats_video.shape)
