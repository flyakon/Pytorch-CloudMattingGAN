import torch
import torch.utils.data as data_utils
import os
import numpy as np
from PIL import Image
import glob
from CloudMatting.utils import path_utils
from CloudMatting.transform.builder import build_transforms
from CloudMatting.transform.matting_transforms import Compose
from CloudMatting.utils import cloud_utils
import tqdm

class CloudSingleDataset(data_utils.Dataset):

    def __init__(self, data_path,
                 data_format='*.jpg', img_size=(512, 512),with_name=False, **kwargs):

        self.data_folder = data_path
        self.data_files = glob.glob(os.path.join(self.data_folder, data_format))
        self.with_name=with_name
        if not isinstance(img_size, list):
            self.img_size = (img_size, img_size)

        transforms = []
        if 'transforms_cfg' not in kwargs.keys():
            raise Exception('transforms_cfg not in parameters')
        transforms_cfg = kwargs['transforms_cfg']
        for param in transforms_cfg.values():
            transforms.append(build_transforms(**param))
        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.data_files)

    def get_data(self, img_file):
        item_result = []
        img = Image.open(img_file)

        img = self.transforms(img)
        assert img.shape[0]==3
        item_result.append(img)
        return item_result

    def __getitem__(self, item):
        item_result = []
        img_file = self.data_files[item]
        item_result.extend(self.get_data(img_file))
        if self.with_name:
            file_name = path_utils.get_filename(img_file, is_suffix=False)
            item_result.append(file_name)
        return item_result

class CloudInMemoryleDataset(data_utils.Dataset):
    def __init__(self, data_path,
                 data_format='*.jpg', img_size=(512, 512),with_name=False, **kwargs):

        self.data_folder = data_path
        self.data_files = glob.glob(os.path.join(self.data_folder, data_format))
        self.with_name=with_name
        if not isinstance(img_size, list):
            self.img_size = (img_size, img_size)

        transforms = []
        if 'transforms_cfg' not in kwargs.keys():
            raise Exception('transforms_cfg not in parameters')
        transforms_cfg = kwargs['transforms_cfg']
        for param in transforms_cfg.values():
            transforms.append(build_transforms(**param))
        self.transforms = Compose(transforms)

        self.load_data()


    def load_data(self):
        self.data_list = []
        for img_file in tqdm.tqdm(self.data_files):
            img = Image.open(img_file)
            self.data_list.append(img)

    def __len__(self):
        return len(self.data_list)

    def get_data(self, index):
        item_result = []
        img = self.data_list[index]

        img = self.transforms(img)
        assert img.shape[0]==3
        item_result.append(img)
        return item_result

    def __getitem__(self, item):
        item_result = []
        img_file = self.data_files[item]
        item_result.extend(self.get_data(item))
        if self.with_name:
            file_name = path_utils.get_filename(img_file, is_suffix=False)
            item_result.append(file_name)
        return item_result

class CloudMattingDataset(data_utils.Dataset):

    def __init__(self,bg_data,thickCloud_data,thinCloud_data,in_memory=True,**kwargs):
        if not in_memory:
            self.bg_dataset=CloudSingleDataset(**bg_data)
            self.thick_dataset=CloudSingleDataset(**thickCloud_data)
            self.thin_dataset=CloudSingleDataset(**thinCloud_data)
        else:
            self.bg_dataset = CloudInMemoryleDataset(**bg_data)
            self.thick_dataset = CloudInMemoryleDataset(**thickCloud_data)
            self.thin_dataset = CloudInMemoryleDataset(**thinCloud_data)

    def __len__(self):
        return min(len(self.bg_dataset),len(self.thick_dataset),len(self.thin_dataset))

    def __getitem__(self, item):
        idx=np.random.randint(0,len(self.bg_dataset),1,)[0]
        bg_img=self.bg_dataset.__getitem__(idx)[0]

        if np.random.randint(0, 2, 1)[0] == 0:
            idx = np.random.randint(0, len(self.thick_dataset), 1)[0]
            thick_img = self.thick_dataset.__getitem__(idx)[0]
        else:
            idx = np.random.randint(0, len(self.thin_dataset), 1)[0]
            thick_img = self.thin_dataset.__getitem__(idx)[0]

        random_id=np.random.randint(0,2,1)[0]
        if random_id==0:
            idx = np.random.randint(0, len(self.thick_dataset), 1)[0]
            cloud_img = self.thick_dataset.__getitem__(idx)[0]
        else:
            idx = np.random.randint(0, len(self.thin_dataset), 1)[0]
            cloud_img = self.thin_dataset.__getitem__(idx)[0]

        return bg_img, thick_img, cloud_img