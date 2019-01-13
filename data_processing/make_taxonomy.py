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
