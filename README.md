# MUSH: Multi-Scale Hierarchical Feature Extraction for Semantic Image Synthesis
This paper proposes a new network for semantic image synthesis, which contains multi-scale hierarchical architecture for semantic feature extration. It is able to distinguish the relative position of each pixel inside its category area so as to get better results. Here is the code for this paper

## Code Structure
- `data:` defines classes of all datasets.
- `models/pix2pix_model.py:` creates all networks, controls the process of image synthesis and computes the losses.
- `models/networks/:` defines all components of the model and contains implementation of each component.
- `options:` creates option lists.
- `trainers/:` controls the training process.
- `train.py` and `test.py:` the entrance for training and testing.

## Installation
Clone this repository by
```bash
git clone https://github.com/onepunchcc/MUSH.git
```

## Dependencies
Our program requires
```bash
torch>=1.0.0
torchvision
dominate>=2.3.1
dill
scikit-image
```
Please install them by
```bash
pip install -r requirements.txt
```

## Dataset Preparation
The Cityscapes, ADE20K and COCO-Stuff dataset can be downloaded from [Cityscapes](https://www.cityscapes-dataset.com/), [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) and [COCO-Stuff](https://github.com/nightrome/cocostuff). 

## Model Training
You can use the following command to train the model:
```bash
python train.py --name [experiment_name] --dataset_mode [dataset_mode] --dataroot [path_to_dataset] --no_ganFeat_loss
```
`[experiment_name]` is the name that you can define for this experiment. `[dataset_mode]` can be `ade20k`, `cityscapes` or `coco`. `[path_to_dataset]` is the path to the dataset. If you want an encoder for the model, please add `--use_vae` in the command. GAN feature matching loss can be used by deleting `--no_ganFeat_loss`. Information about more options can be shown using `python train.py --help`.

## Model Testing
Similar to model training, the model can be tested by the following command:
```bash
python test.py --name [experiment_name] --dataset_mode [dataset_mode] --dataroot [path_to_dataset] 
```
The loaded checkpoint can be changed using `which_epoch`.
