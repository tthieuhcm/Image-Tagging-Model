import os
import numpy as np
import matplotlib
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
    class_weights = {class_to_idx[key]: min_number_of_images / val for key, val in classes_and_number_of_images.items()}
    weight_array = numpy.zeros(len(class_weights))
    for key, val in class_weights.items():
        weight_array[key] = val
    return weight_array


def tag_statistic(data_folder):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    train_dataset = ImageNet_Dataset(
        data_folder,
        loader=default_loader,
        # transform=transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ]),
        dataset_type='train')

    tag = iter(train_dataset.classes)
    init_count_of_tags = numpy.zeros(len(train_dataset.classes))
    stat = dict(zip(tag, init_count_of_tags))

    for item in train_dataset.samples:
        for target in item[1].numpy():
            stat[train_dataset.inx_to_class[target]] += 1

    return stat, train_dataset.class_to_idx


if __name__ == "__main__":
    # data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/'
    # classes = list()
    # with open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/devkit/data/map_clsloc.txt',
    #           'r') as cls_loc:
    #     cls_loc_lines = cls_loc.readlines()
    #     for line in cls_loc_lines:
    #         classes.append(line.split()[0])
    # with open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/devkit/data/map_det.txt',
    #           'r') as det:
    #     det_lines = det.readlines()
    #     for line in det_lines:
    #         classes.append(line.split()[0])
    # # for subfolder in sorted(os.listdir('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/Data/DET/train/ILSVRC2013_train')):
    # #     classes.append(subfolder)
    # #
    # classes_and_number_of_images, class_to_idx = tag_statistic(data_folder)
    # CLS_LOC_classes_and_number_of_images = dict()
    # for _class in classes:
    #     CLS_LOC_classes_and_number_of_images[_class] = classes_and_number_of_images[_class]
    # sorted_CLS_LOC_classes_and_number_of_images = sorted(classes_and_number_of_images.items(),
    #                                                      key=lambda kv: kv[1])  # , reverse=True)
    # reversed_sorted_CLS_LOC_classes_and_number_of_images = sorted(classes_and_number_of_images.items(),
    #                                                               key=lambda kv: kv[1], reverse=True)

    # make_class_weights(data_folder)
    # print("a")
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
    # 'n02089078', 732.0
    # 'n02095889', 936.0
    # 'n03866082', 1005.0
    # 'n03041632', 1067.0
    # 'n07730033', 1125.0
    # 'n03899768', 1152.0
    # 'n02794156', 1199.0
    # 'n03400231', 1222.0
    # 'n04127249', 1259.0
    # 'n03791053', 1300.0
    # 'n02084071', 67513
    # 'n01503061', 33509
    # 'n00007846', 29700
    # 'n01726692', 8591
    # 'n02484322', 7707
    # 'n04379243', 6711
    # 'n02274259', 4270
    # 'n04356056', 2003
    # 'n02165456', 1003
    # 'n03062245', 461
    # 'n00021939', 796018
    # 'n00015388', 691740
    # 'n01471682', 597128
    # 'n01466257', 597128
    # 'n03489162', 10035
    # 'n02121808', 9762
    # 'n02127808', 9531
    # 'n02088839', 732
    # 'n02091635', 738
    # 'n02089973', 754
    # 'n00015388', 495545 animal
    # 'n01861778', 224917 mammal
    # 'n02075296', 143429 carnivore
    # 'n02329401', 10538 rodent
    # 'n02876657', 10443 bottle
    # 'n04437953', 10384 timepiece
    # 'n02121808', 9762 domestic cat
    # 'n02402010', 297 bovine
    # 'n02402425', 297 cattle
    # 'n02062744', 385 whale
    plt.rcParams.update({'font.size': 17})

    # objects = ['Coonhound', 'Sealyham', 'Overskirt', 'Cleaver', 'Cardoon', 'Patio', 'Barometer', 'Frypan',
    #            'Safety pin', 'Scooter'][::-1]
    # objects = ['Dog', 'Bird', 'Person', 'Snake', 'Monkey', 'Table', 'Butterfly', 'Sunglasses', 'Ladybug',
    #            'Cocktail \nshaker']
    objects = ['Artifact', 'Animal', 'Vertebrate', 'Chordate', 'Hand tool', 'Domestic \ncat', 'Big cat', 'Coonhound', 'Otterhound',
               'English \nfoxhound']
    # objects = ['Animal', 'Mammal', 'Carnivore', 'Rodent', 'Bottle', 'Timepiece', 'Domestic \ncat', 'Whale', 'Bovine', 'Cattle']
    y_pos = np.arange(len(objects))
    # performance = [732, 936, 1005, 1067, 1125, 1152, 1199, 1222, 1259, 1300][::-1]
    # performance = [67513, 33509, 29700, 8591, 7707, 6711, 4270, 2003, 1003, 461]
    performance = [796018, 691740, 597128, 597128, 10035, 9762, 9531, 732, 732, 738]
    # performance = [495545, 224917, 143429, 10538, 10443, 10384, 9762, 385, 297, 297]
    fig, ax = plt.subplots()
    ax.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of images')
    # plt.title('Number of images for each class')
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.show()
