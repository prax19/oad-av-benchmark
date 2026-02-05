# Contribution guide
Here you can find information about how you can help with growing this repo or/and use it in your research.

## Adding new content to repository

### Dataset
For any custom dataset, there should be provided a proper `torch.utils.data.Dataset` implementation in `extraction\extractor_datasets.py` (for finding videos and labels) and `core\datasets.py` (reading `.npz` files and labels).

*full instructions soon*

### OAD method
All OAD methods are stored in [`methods`](methods) directory. You can put there any OAD repository/code you wish, as long as it's possible to create proper adapter to build and run model. If the method has its own repository, it's important to include it as *git submodule*.

You should also add your own method adapter in [`adapters.py`](core\adapters.py).

*full instructions soon*

### RGB backbones for extraction
It's possible to add another pre-trained backbone to run pre-extraction for future training. As backbones and weights are from `MMAction2`, you can include reference name of any backbone variant from [there](https://github.com/open-mmlab/mmaction2/tree/main/configs/recognition), as long as it's 2D or 3D backbone or you can provide proper extractor in [`extractors.py`](extraction\extractors.py).

The reference name is just stem of `.py` config files (without `.py` postfix). It should be included in [`models.yaml`](weights\backbones\models.yaml) in order to download weights and configuration.

*full instructions soon*
