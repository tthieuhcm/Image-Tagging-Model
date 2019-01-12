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
AllWords = list()
ResultList = list()

regex = re.compile('(.*)?.jpg')

for line in fhand:
    line.rstrip()
    words = line.split()
    filtered = filter(lambda i: not regex.search(i), words)
    filtered = [i for i in words if not regex.search(i)]
    AllWords.extend(filtered)

AllWords.sort()

for word in AllWords:
    if word not in ResultList:
        ResultList.append(word)

tag_file = open("taxonomy.txt", "w")
for item in ResultList:
    tag_file.write("%s\n" % item)
