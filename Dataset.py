import os
import numpy as np
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torch.utils.data.dataset import Dataset


class LIDC(Dataset):
    def __init__(self, train=True, augment=False):
        self.train = train
        self.len = 0
        self.augment = augment

        if self.train:
            self.data_folder = 'path/to/train'
        else:
            self.data_folder = 'path/to/test'

        if (train and (not os.path.exists('Train-128'))) or (not train and (not os.path.exists('Test-128'))):
            print('Getting Data')
            extract_archive(self.data_folder, './', False)
        else:
            print('Using existing Data')

        if train:
            self.data_folder = './Train-128/'
        else:
            self.data_folder = './Test-128/'

        self.len = len([name for name in os.listdir(self.data_folder) if name.endswith('.npz')])

    def __getitem__(self, index):
        path = self.data_folder + f'{index}.npz'
        image = np.load(path)['arr_0']

        image = transforms.ToTensor()(image)

        if self.augment:
            image = transforms.RandomAffine(
                degrees=(-1, 1),
                scale=(1, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR
            )(image)
        return image.float()

    def __len__(self):
        return self.len
