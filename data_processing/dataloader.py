from __future__ import print_function, division
import os
from numpy import array
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MIR_FLICKR_Dataset(Dataset):
    """MIR-Flickr-25K dataset."""

    def __init__(self, img_name, img_labels, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
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
        image = Image.open(image_name)
        image_labels = self.img_labels[idx]
        sample = {'image': image, 'labels': image_labels}

        if self.transform:
            sample = {'image': self.transform(sample['image']), 'labels': image_labels}

        return sample


def show_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch, labels_batch = sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.title('Batch from dataloader')


if __name__ == "__main__":
    with open('./taxonomy.txt') as all_tags:
        tag_list = [item.rstrip() for item in all_tags.readlines()]

    number_of_images = 25000
    number_of_tags = len(tag_list)
    img_labels = array([[0 for i in range(number_of_tags)] for j in range(number_of_images)])

    with open('./mirflickr_labels.txt') as info:
        lines = info.readlines()
        img_name = array([img_info.split()[0] for img_info in lines])
        for i, img_label in enumerate(lines):
            img_tags = img_label.split()[1:]
            tag_index = [tag_list.index(tag) for tag in img_tags]
            for index in tag_index:
                img_labels[i][index] = 1

    mir_flickr_dataset = MIR_FLICKR_Dataset(img_name, img_labels,
                                            root_dir='/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/mirflickr',
                                            transform=transforms.Compose([transforms.Resize((333, 500)),
                                                                          transforms.ToTensor()]))

    dataloader = DataLoader(mir_flickr_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['labels'].size())

        if i_batch == 3:
            plt.figure()
            show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break