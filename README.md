# ResDiff: Combining CNN and Diffusion Model for Image Super-Resolution

## Brief
This is an implementation of ResDiff by PyTorch.Thank you for your reading!
Once the paper is accepted, we will refine the code and release it as soon as possible.
(Our code is based on [this](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement))


## Usage

### Environment
```shell
pip install -r requirement.txt
```

### Data Prepare
Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```shell
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

The obtained data is organized as follows:
```
# set the high/low resolution images, bicubic interpolation images path 
dataset/celebahq_16_128/
├── hr_128 # it's same with sr_16_128 directory if you don't have ground-truth images.
├── lr_16 # vinilla low resolution images
└── sr_16_128 # images ready to super resolution
```

Then change the dataset config to your data path and image resolution: 

```json
"datasets" : {
    "train": {
        "dataroot": "[output root] in prepare.py script",
        "l_resolution": "low resolution need to super_resolution",
        "r_resolution": "high resolution",
        "datatype": "lmdb or img, path of img files"
    },
    "val": {
        "dataroot": "[output root] in prepare.py script"
    }
},
```

### Pre-train CNN and generate predicted images

Modify the parameters in several files in the /pretrain_CNN directory, and then run the following script directly.

```shell
python pretrain_CNN/train.py
```

The CNN predictions will be written to the specified path, 
note that the path needs to be specified as the previously generated **dataset/xxx/sr_[lr]_[hr]**.

### Training/Resume Training

```shell
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
# config file: e.g. config/sr_resdiff_32_128.json
python sr.py -p train -c [config file]
```

### Test/Evaluation

```shell
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c [config file]

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```shell
# run the script
python infer.py -c [config file]
```







