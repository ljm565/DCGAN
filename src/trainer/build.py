import os

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, distributed, random_split

from models import Generator, Discriminator
from utils import RANK, LOGGER, colorstr
from utils.data_utils import DLoader, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    return generator, discriminator


def build_dataset(config, modes):
    dataset_dict = {}
    if config.CelebA_train:
        # set to CelebA size
        config.img_size = 64

        # init train, validation, test sets
        os.makedirs(config.CelebA.path, exist_ok=True)
        data_path = os.path.join(config.CelebA.path, 'CelebA')

        if not os.path.isdir(data_path):
            try:
                LOGGER.info(colorstr('Downloading...'))
                wget_path = os.path.join(config.CelebA.path, 'CelebA.zip')
                exec = 'wget -O ' + wget_path + ' https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip && unzip ' + wget_path + ' -d ' + data_path + ' && rm ' + wget_path
                os.system(exec)
            except Exception as e:
                LOGGER.info(colorstr('red', 'Downloading failed...'))
                LOGGER.info(colorstr('red', 'Command not found. Please check mkdir, wget, unzip, rm command'))
                raise e
        
        trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.Resize(config.img_size),
                                transforms.CenterCrop(config.img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))]) if config.convert2grayscale else \
                transforms.Compose([transforms.Resize(config.img_size),
                            transforms.CenterCrop(config.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dsets.ImageFolder(root=data_path, transform=trans)
        trainset, valset, testset = random_split(dataset, [len(dataset)-2000, 1000, 1000])
        tmp_dsets = {'train': trainset, 'validation': valset, 'test': testset}
        for mode in modes:
            dataset_dict[mode] = tmp_dsets[mode]
    else:
        for mode in modes:
            dataset_dict[mode] = DLoader(config.CUSTOM.get(f'{mode}_data_path'))
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, modes, is_ddp=False):
    datasets = build_dataset(config, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders