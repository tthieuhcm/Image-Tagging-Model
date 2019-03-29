import matplotlib.pyplot as plt
import numpy
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms

from dataloader import ImageNet_Dataset

import operator
from nltk.corpus import wordnet


def get_name(wnid):
    return wordnet.synset_from_pos_and_offset('n', int(wnid[1:])).lemmas()[0].name()


def make_class_weights(data_folder):
    classes_and_number_of_images, class_to_idx = tag_statistic(data_folder)
    min_number_of_images = min(classes_and_number_of_images.items(), key=lambda x: x[1])[1]
    class_weights = {class_to_idx[key]: min_number_of_images/val for key, val in classes_and_number_of_images.items()}
    weight_array = numpy.zeros(len(class_weights))
    for key, val in class_weights.items():
        weight_array[key] = val
    return weight_array


def tag_statistic(data_folder):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageNet_Dataset(
        data_folder,
        loader=default_loader,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        dataset_type='train')

    tag = iter(train_dataset.classes)
    init_count_of_tags = numpy.zeros(len(train_dataset.classes))
    stat = dict(zip(tag, init_count_of_tags))

    for item in train_dataset.samples:
        for target in item[1].numpy():
            stat[train_dataset.inx_to_class[target]] += 1

    return stat, train_dataset.class_to_idx


if __name__ == "__main__":
    data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/'

    # classes_and_number_of_images, class_to_idx = tag_statistic(data_folder)
    make_class_weights(data_folder)
    print("a")
    # sorted_b = sorted(b.items(), key=operator.itemgetter(1), reverse=True)
    # dictOfWords = {i[0]: i[1] for i in a[::150]}
    #
    # names = list(dictOfWords.keys())
    # values = list(dictOfWords.values())
    # fig, ax = plt.subplots()
    # for i, v in enumerate(values):
    #     ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    # width = 0.75  # the width of the bars
    # ind = numpy.arange(len(values))  # the x locations for the groups
    # ax.barh(ind, values, width, color="blue")
    # ax.set_yticks(ind + width / 2)
    # ax.set_yticklabels(names, minor=False)
    # # plt.title('title')
    # plt.xlabel('Number of images')
    # plt.ylabel('Tag name')
    # plt.show()
    #
    # names = list(dictOfWords.keys())
    # values = list(dictOfWords.values())
    #
    # # N1, N2 = len(df[df['Expo'] == 1]), len(df[df['Expo'] == 2])
    # # plt.bar([1, 2], [N1, N2], color="blue")
    #
    # plt.bar(range(len(dictOfWords)), values, tick_label=names)
    # plt.savefig('bar.png')
    # plt.show()
