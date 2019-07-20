import os
import xml
import torch
from nltk.corpus import wordnet


def wnid_to_tags(wnid):
    """
    :param wnid: WordNet ID. Examples: n07555863, n03642806, n03770439
    :return: List of WordNet IDs which are related to the main WordNet ID
    """
    list_of_tag = list()

    tag = wordnet.synset_from_pos_and_offset('n', int(wnid[1:]))
    list_of_tag.append('n'+str(tag.offset()).zfill(8))

    while tag.lemma_names()[0] != 'plant' \
            and tag.lemma_names()[0] != 'geological_formation' \
            and tag.lemma_names()[0] != 'natural_object' \
            and tag.lemma_names()[0] != 'sport' \
            and tag.lemma_names()[0] != 'artifact' \
            and tag.lemma_names()[0] != 'fungus' \
            and tag.lemma_names()[0] != 'person' \
            and tag.lemma_names()[0] != 'animal' \
            and tag.lemma_names()[0] != 'communication' \
            and tag.lemma_names()[0] != 'toilet_tissue' \
            and tag.lemma_names()[0] != 'sphere' \
            and tag.lemma_names()[0] != 'food':
        tag = tag.hypernyms()[0]
        list_of_tag.append('n'+str(tag.offset()).zfill(8))
    return list_of_tag


def find_hypernyms_from_wnid(class_id, classes):
    """
    :param class_id: the main WordNet ID
    :param classes: list of all WordNet IDs
    :return: list of all WordNet IDs and the WordNet IDs which are related to the main one
    """
    tag = wordnet.synset_from_pos_and_offset('n', int(class_id[1:]))
    tag_name = tag.lemma_names()[0]
    while tag_name != 'plant' \
            and tag_name != 'geological_formation' \
            and tag_name != 'natural_object' \
            and tag_name != 'sport' \
            and tag_name != 'artifact' \
            and tag_name != 'fungus' \
            and tag_name != 'person' \
            and tag_name != 'animal' \
            and tag_name != 'communication' \
            and tag_name != 'toilet_tissue' \
            and tag_name != 'sphere' \
            and tag_name != 'food':
        tag = tag.hypernyms()[0]
        tag_name = tag.lemma_names()[0]
        classes.append('n'+str(tag.offset()).zfill(8))
    return classes


def find_hypernyms_names_from_wnid(class_id):
    """
    :param class_id: the main WordNet ID
    :return: All parents' name from the main WordNet IDs
    """
    classes_name = list()
    tag = wordnet.synset_from_pos_and_offset('n', int(class_id[1:]))
    tag_name = tag.lemma_names()[0]
    while tag_name != 'plant' \
            and tag_name != 'geological_formation' \
            and tag_name != 'natural_object' \
            and tag_name != 'sport' \
            and tag_name != 'artifact' \
            and tag_name != 'fungus' \
            and tag_name != 'person' \
            and tag_name != 'animal' \
            and tag_name != 'communication' \
            and tag_name != 'toilet_tissue' \
            and tag_name != 'sphere' \
            and tag_name != 'food':
        tag = tag.hypernyms()[0]
        tag_name = tag.lemma_names()[0]
        classes_name.append(tag_name)
    return classes_name


def make_classes(map_dir):
    """
    :param map_dir: folder has map files
    :return: list of WordNet IDs and their indexes
    """
    with open(map_dir+'/map_clsloc.txt', 'r') as cls_loc, \
            open(map_dir+'/map_det.txt', 'r') as det:
        # classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes = list()
        cls_loc_lines = cls_loc.readlines()
        det_lines = det.readlines()
        for line in cls_loc_lines:
            classes.append(line.split()[0])
        for line in det_lines:
            classes.append(line.split()[0])

    classes = list(set(classes))
    classes.sort()

    for class_id in classes:
        find_hypernyms_from_wnid(class_id, classes)

    classes.append('n00017222')    # n00017222 : plant
    classes.append('n03956922')    # n03956922 : plant
    classes.append('n09287968')    # n09287968 : geological_formation
    classes.append('n00019128')    # n00019128 : natural_object
    classes.append('n00021939')    # n00021939 : artifact
    classes.append('n12992868')    # n12992868 : fungus
    classes.append('n00007846')    # n00007846 : person
    classes.append('n00015388')    # n00015388 : animal
    classes.append('n00033020')    # n00033020 : communication
    classes.append('n15075141')    # n15075141 : toilet_tissue
    classes.append('n13899200')    # n13899200 : sphere
    classes.append('n00021265')    # n00021265 : food
    classes.append('n07555863')    # n07555863 : food

    classes = list(set(classes))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_CLS_LOC_dataset(CLS_LOC_data_dir, CLS_LOC_annotation_dir, class_to_idx, dataset_type):
    images = []
    CLS_LOC_data_dir = os.path.expanduser(CLS_LOC_data_dir)
    for folder in sorted(os.listdir(CLS_LOC_data_dir)):
        subdir = os.path.join(CLS_LOC_data_dir, folder)
        if not os.path.isdir(subdir):
            continue

        list_target = wnid_to_tags(folder)

        for root, _, fnames in sorted(os.walk(subdir)):
            sorted(fnames)
            # if dataset_type == 'train':
            #     fnames = fnames[100:]
            # elif dataset_type == 'test':
            #     fnames = fnames[:100]

            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                list_converted_target = list()

                for target in list_target:
                    list_converted_target.append(class_to_idx[target])

                pt_tensor_from_list = torch.LongTensor(list_converted_target)

                item = (path, pt_tensor_from_list)
                images.append(item)

    return images


def make_DET_dataset(DET_data_dir, DET_annotation_dir, class_to_idx, dataset_type):
    images = []
    annotation_dir = os.path.expanduser(DET_annotation_dir)
    _0_object = 0
    _1_object = 0
    _2_object = 0
    _3_object = 0
    _4_object = 0
    _5_object = 0
    _6_object = 0
    _7_object = 0
    _8_object = 0
    _9_object = 0
    _10_object = 0
    person = 0
    if dataset_type == 'val':
        for root, _, fnames in sorted(os.walk(annotation_dir)):
            sorted(fnames)

            for fname in sorted(fnames):
                image_path = os.path.join(DET_data_dir, fname[:-3] + 'JPEG')
                annotation_path = os.path.join(annotation_dir, fname)
                annotation = xml.etree.ElementTree.parse(annotation_path).getroot()
                list_target = list()

                for object in annotation.findall('object'):
                    list_target.append(object.find('name').text)

                list_target = list(set(list_target))
                # if 'n00007846' in list_target:
                #     person += 1
                # if len(list_target) == 0:
                #     _0_object += 1
                # elif len(list_target) == 1:
                #     _1_object += 1
                # elif len(list_target) == 2:
                #     _2_object += 1
                # elif len(list_target) == 3:
                #     _3_object += 1
                # elif len(list_target) == 4:
                #     _4_object += 1
                # elif len(list_target) == 5:
                #     _5_object += 1
                # elif len(list_target) == 6:
                #     _6_object += 1
                # elif len(list_target) == 7:
                #     _7_object += 1
                # elif len(list_target) == 8:
                #     _8_object += 1
                # elif len(list_target) == 9:
                #     _9_object += 1
                # else:
                #     _10_object += 1
                extended_list_target = list()
                for target in list_target:
                    extended_list_target = extended_list_target + wnid_to_tags(target)

                extended_list_target = list(set(extended_list_target))

                list_converted_target = list()
                for target in extended_list_target:
                    list_converted_target.append(class_to_idx[target])

                pt_tensor_from_list = torch.LongTensor(list_converted_target)

                item = (image_path, pt_tensor_from_list)
                images.append(item)
    else:
        for folder in sorted(os.listdir(annotation_dir)):
            ann_dir = os.path.join(annotation_dir, folder)
            image_dir = os.path.join(DET_data_dir, folder)
            if not os.path.isdir(ann_dir):
                continue
            if folder == 'ILSVRC2013_train':
                for subfolder in sorted(os.listdir(os.path.join(annotation_dir, folder))):
                    ann_dir = os.path.join(annotation_dir, folder, subfolder)
                    image_dir = os.path.join(DET_data_dir, folder, subfolder)

                    for root, _, fnames in sorted(os.walk(ann_dir)):
                        sorted(fnames)
                        # if dataset_type == 'train':
                        #     fnames = fnames[50:]
                        # elif dataset_type == 'test':
                        #     fnames = fnames[:50]

                        for fname in sorted(fnames):
                            image_path = os.path.join(image_dir, fname[:-3] + 'JPEG')
                            annotation_path = os.path.join(ann_dir, fname)
                            annotation = xml.etree.ElementTree.parse(annotation_path).getroot()
                            list_target = list()

                            for object in annotation.findall('object'):
                                list_target.append(object.find('name').text)

                            list_target = list(set(list_target))
                            # if 'n00007846' in list_target:
                            #     person += 1
                            # if len(list_target) == 0:
                            #     _0_object += 1
                            # elif len(list_target) == 1:
                            #     _1_object += 1
                            # elif len(list_target) == 2:
                            #     _2_object += 1
                            # elif len(list_target) == 3:
                            #     _3_object += 1
                            # elif len(list_target) == 4:
                            #     _4_object += 1
                            # elif len(list_target) == 5:
                            #     _5_object += 1
                            # elif len(list_target) == 6:
                            #     _6_object += 1
                            # elif len(list_target) == 7:
                            #     _7_object += 1
                            # elif len(list_target) == 8:
                            #     _8_object += 1
                            # elif len(list_target) == 9:
                            #     _9_object += 1
                            # else:
                            #     _10_object += 1
                            extended_list_target = list()
                            for target in list_target:
                                extended_list_target = extended_list_target + wnid_to_tags(target)

                            extended_list_target = list(set(extended_list_target))

                            list_converted_target = list()
                            for target in extended_list_target:
                                list_converted_target.append(class_to_idx[target])

                            pt_tensor_from_list = torch.LongTensor(list_converted_target)

                            item = (image_path, pt_tensor_from_list)
                            images.append(item)
            else:
                for root, _, fnames in sorted(os.walk(ann_dir)):
                    sorted(fnames)
                    # if dataset_type == 'train':
                    #     fnames = fnames[50:]
                    # elif dataset_type == 'test':
                    #     fnames = fnames[:50]

                    for fname in sorted(fnames):
                        image_path = os.path.join(image_dir, fname[:-3]+'JPEG')
                        annotation_path = os.path.join(ann_dir, fname)
                        annotation = xml.etree.ElementTree.parse(annotation_path).getroot()
                        list_target = list()

                        for object in annotation.findall('object'):
                            list_target.append(object.find('name').text)

                        list_target = list(set(list_target))
                        # if 'n00007846' in list_target:
                        #     person += 1
                        # if len(list_target) == 0:
                        #     _0_object += 1
                        # elif len(list_target) == 1:
                        #     _1_object += 1
                        # elif len(list_target) == 2:
                        #     _2_object += 1
                        # elif len(list_target) == 3:
                        #     _3_object += 1
                        # elif len(list_target) == 4:
                        #     _4_object += 1
                        # elif len(list_target) == 5:
                        #     _5_object += 1
                        # elif len(list_target) == 6:
                        #     _6_object += 1
                        # elif len(list_target) == 7:
                        #     _7_object += 1
                        # elif len(list_target) == 8:
                        #     _8_object += 1
                        # elif len(list_target) == 9:
                        #     _9_object += 1
                        # else:
                        #     _10_object += 1
                        extended_list_target = list()
                        for target in list_target:
                            extended_list_target = extended_list_target + wnid_to_tags(target)

                        extended_list_target = list(set(extended_list_target))

                        list_converted_target = list()
                        for target in extended_list_target:
                            list_converted_target.append(class_to_idx[target])

                        pt_tensor_from_list = torch.LongTensor(list_converted_target)

                        item = (image_path, pt_tensor_from_list)
                        images.append(item)
    # print(person)
    # print(_0_object)
    # print(_1_object)
    # print(_2_object)
    # print(_3_object)
    # print(_4_object)
    # print(_5_object)
    # print(_6_object)
    # print(_7_object)
    # print(_8_object)
    # print(_9_object)
    # print(_10_object)
    return images


if __name__ == "__main__":
    # root = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'
    # map_dir = os.path.join(root, 'devkit/data')
    # classes, class_to_idx = make_classes(map_dir)
    #
    # DET_image_dir = os.path.join(root, 'Data/DET/val')
    # DET_annotation_dir = os.path.join(root, 'Annotations/DET/val')
    # DET_samples = make_DET_dataset(DET_image_dir, DET_annotation_dir, class_to_idx, 'val')

    for tag in wnid_to_tags('n03769881'):
        print(tag)
        print(wordnet.synset_from_pos_and_offset('n', int(tag[1:])).lemmas()[0].name())