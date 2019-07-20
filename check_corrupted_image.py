import torch
from torchvision.datasets.folder import default_loader

from dataloader import ImageNet_Dataset
from torchvision.transforms import transforms

if __name__ == "__main__":
    root_dir = './ILSVRC'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageNet_Dataset(
        root_dir,
        loader=default_loader,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        dataset_type='train')

    for path, target in train_dataset.samples:
        try:
            sample = train_dataset.loader(path)
            if train_dataset.transform is not None:
                sample = train_dataset.transform(sample)

            targets_tensor = torch.zeros(len(train_dataset.classes))
            for i in range(target.shape[0]):
                targets_tensor[target[i]] = 1

        except Exception as e:
            print(path)
            print(e)