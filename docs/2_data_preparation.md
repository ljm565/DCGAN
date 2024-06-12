# Data Preparation
Here, we will proceed with a GAN model training tutorial using the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset by default.
Please refer to the following instructions to utilize custom datasets.


### 1. CelebA
If you want to train on the CelebA dataset, simply set the `CelebA_train` value in the `config/config.yaml` file to `True` as follows.
```yaml
CelebA_train: True
CelebA:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
If you want to train your custom dataset, set the `CelebA_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
```yaml
CelebA_train: False
CelebA:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
