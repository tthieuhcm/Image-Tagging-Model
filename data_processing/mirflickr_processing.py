# Author: Trung Hieu Tran
# Date  : 12/1/2019

"""
For processing Mirflickr25k dataset only

Short description:
Mirflickr25k dataset has 25000 images with 24 tags and 14 tags of them are ground truth.
Therefore this dataset has 24 files as 24 tags and 14 files in addition as ground truth.
Those ground truth files has substring "_r1" in their names. But we'll ignore those files because we don't need them for this project.
Each tag file has many lines. Each line has 1 number as the image name.
We'll combine those files into 1 file: mirflickr_labels.txt
This file has 25000 lines as 25000 images, each line has image name with their tags

Example: im1.jpg structures female portrait sky people
Which is:
- "im1.jpg" is the image name
- "structures", "female", "portrait", "sky", "people" are the image tags

Note: in label files, they write 1 tag wrong (wat3er)
"""

import os

number_of_image = 25000
img_array = [[] for _ in range(number_of_image)]

for file in os.listdir("./mirflickr_annotations"):
    if not file.endswith("_r1.txt") and not file.endswith("README.txt"):
        tag = file.replace(".txt", "")
        with open(os.path.join("./mirflickr_annotations/", file)) as f:
            lines = [line.rstrip('\n') for line in f]
            for line in lines:
                img_array[int(line)-1].append(tag)

tag_file = open("mirflickr_labels.txt", "w")
for i, tag_list in enumerate(img_array):
    tag_file.write("im%s.jpg " % str(i+1))
    for item in tag_list:
        tag_file.write("%s " % item)
    tag_file.write('\n')

tag_file.close()
