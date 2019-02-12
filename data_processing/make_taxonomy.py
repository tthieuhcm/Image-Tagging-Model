# Author: Trung Hieu Tran
# Date  : 12/1/2019

"""
Combine all the tags into taxonomy file

Short description:
We'll use many different datasets and combine them to make our one.
Therefore we'll have one tags file each dataset.
This module will combine all the tags into one taxonomy file.
"""

import re
from ImageNet import wnid_to_tags


def mirflickr_processing():
    fname = "mirflickr_labels.txt"
    fhand = open(fname)
    all_tags = list()
    tag_list = list()

    regex = re.compile('(.*)?.jpg')

    for line in fhand:
        words = line.split()
        filtered = [i for i in words if not regex.search(i)]
        all_tags.extend(filtered)

    all_tags.sort()

    for word in all_tags:
        if word not in tag_list:
            tag_list.append(word)

    tag_file = open("taxonomy.txt", "w")
    for item in tag_list:
        tag_file.write("%s\n" % item)


def imagenet_processing():
    clsloc_path = "../ILSVRC/devkit/data/map_clsloc.txt"
    det_path = "../ILSVRC/devkit/data/map_det.txt"
    clsloc_file = open(clsloc_path)
    det_file = open(det_path)

    all_tags = list()

    for line in clsloc_file:
        words = line.split()
        all_tags.extend(wnid_to_tags(int(words[0][1:])))

    for line in det_file:
        words = line.split()
        all_tags.extend(wnid_to_tags(int(words[0][1:])))

    all_tags = list(set(all_tags))
    all_tags.sort()

    tag_file = open("taxonomy.txt", "w")
    for item in all_tags:
        tag_file.write("%s\n" % item)


if __name__ == "__main__":
    # imagenet_processing()
    mirflickr_processing()