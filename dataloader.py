import os
import torch

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset

from utils import make_classes, make_DET_dataset, make_CLS_LOC_dataset


class MIR_FLICKR_Dataset(Dataset):
    """MIR-Flickr-25K dataset."""

    def __init__(self, img_name, img_labels, root_dir, transform=None):
        """
        Args:
            img_name (ndarray): Images' name.
            img_labels (ndarray): List of the images' tags.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_name = img_name
        self.img_labels = img_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                  self.img_name[idx])
        image = Image.open(image_name).convert('RGB')
        image_labels = self.img_labels[idx]
        sample = {'image': image, 'labels': image_labels}

        if self.transform:
            sample = {'image': self.transform(sample['image']), 'labels': image_labels}

        return sample


class ImageNet_Dataset(Dataset):
    def __init__(self, root, loader, transform=None, dataset_type='train'):
        map_dir = os.path.join(root, 'devkit/data')
        if dataset_type == 'train' or dataset_type == 'test':
            CLS_LOC_image_dir = os.path.join(root, 'Data/CLS-LOC/train')
            CLS_LOC_annotation_dir = os.path.join(root, 'Annotations/CLS-LOC/train')
            DET_image_dir = os.path.join(root, 'Data/DET/train')
            DET_annotation_dir = os.path.join(root, 'Annotations/DET/train')
        else:
            CLS_LOC_image_dir = os.path.join(root, 'Data/CLS-LOC/val')
            CLS_LOC_annotation_dir = os.path.join(root, 'Annotations/CLS-LOC/val')
            DET_image_dir = os.path.join(root, 'Data/DET/val')
            DET_annotation_dir = os.path.join(root, 'Annotations/DET/val')

        classes, class_to_idx = make_classes(map_dir)
        CLS_LOC_samples = make_CLS_LOC_dataset(CLS_LOC_image_dir, CLS_LOC_annotation_dir, class_to_idx, dataset_type)
        DET_samples     = make_DET_dataset(DET_image_dir, DET_annotation_dir, class_to_idx, dataset_type)

        if len(CLS_LOC_samples) == 0 or len(DET_samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))

        self.root = root
        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.inx_to_class = {v: k for k, v in class_to_idx.items()}

        self.samples = CLS_LOC_samples + DET_samples

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            print(path)
        if self.transform is not None:
            sample = self.transform(sample)

        targets_tensor = torch.zeros(len(self.classes))
        for i in range(target.shape[0]):
            targets_tensor[target[i]] = 1
        return sample, targets_tensor

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
