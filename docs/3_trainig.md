# Training DCGAN
Here, we provide guides for training a DCGAN model.

### 1. Configuration Preparation
To train a DCGAN model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu                 # You can DDP training with multiple gpus. e.g. gpu: [0], [0,1], [1,2,3], cpu: cpu, mac: mps

# project config
project: outputs/DCGAN      # Project directory
name: CelebA                # Trained model-related data are saved at ${project}/${name} folde

# image setting config
img_size: 64                # Image size will be set to (img_size x img_size)
color_channel: 1            # [1, 3], you can choose one
convert2grayscale: False    # if True and color_channel is 3, you can train color image with grayscaled image

# data config
workers: 0                  # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
CelebA_train: True           # if True, CelebA will be loaded automatically.
CelebA:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 128
epochs: 100
lr: 0.0002
beta1: 0.5
g_dim: 64
d_dim: 64
noise_init_size: 128

# logging config
common: ['train_loss_d', 'train_loss_g', 'validation_loss_d', 'validation_loss_g', 'd_x', 'd_g1', 'd_g2']
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`loss`, `last`].
    * `loss` (default): Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```

When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.