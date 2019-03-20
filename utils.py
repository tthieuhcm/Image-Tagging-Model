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

    classes.append('n00017222')
    classes.append('n03956922')
    classes.append('n09287968')
    classes.append('n00019128')
    classes.append('n00021939')
    classes.append('n12992868')
    classes.append('n00007846')
    classes.append('n00015388')
    classes.append('n00033020')
    classes.append('n15075141')
    classes.append('n13899200')
    classes.append('n00021265')
    classes.append('n07555863')

    # n00017222 : plant
    # n03956922 : plant
    # n09287968 : geological_formation
    # n00019128 : natural_object
    # n00021939 : artifact
    # n12992868 : fungus
    # n00007846 : person
    # n00015388 : animal
    # n00033020 : communication
    # n15075141 : toilet_tissue
    # n13899200 : sphere
    # n00021265 : food
    # n07555863 : food

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

        for root, _, fnames in sorted(os.walk(subdir)):
            sorted(fnames)
            if dataset_type == 'train':
                fnames = fnames[100:]
            elif dataset_type == 'test':
                fnames = fnames[:100]

            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                list_target = wnid_to_tags(folder)

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
                        if dataset_type == 'train':
                            fnames = fnames[50:]
                        elif dataset_type == 'test':
                            fnames = fnames[:50]

                        for fname in sorted(fnames):
                            image_path = os.path.join(image_dir, fname[:-3] + 'JPEG')
                            annotation_path = os.path.join(ann_dir, fname)
                            annotation = xml.etree.ElementTree.parse(annotation_path).getroot()
                            list_target = list()

                            for object in annotation.findall('object'):
                                list_target.append(object.find('name').text)

                            list_target = list(set(list_target))
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
                    if dataset_type == 'train':
                        fnames = fnames[50:]
                    elif dataset_type == 'test':
                        fnames = fnames[:50]

                    for fname in sorted(fnames):
                        image_path = os.path.join(image_dir, fname[:-3]+'JPEG')
                        annotation_path = os.path.join(ann_dir, fname)
                        annotation = xml.etree.ElementTree.parse(annotation_path).getroot()
                        list_target = list()

                        for object in annotation.findall('object'):
                            list_target.append(object.find('name').text)

                        list_target = list(set(list_target))
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

    return images


if __name__ == "__main__":
    root = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'
    map_dir = os.path.join(root, 'devkit/data')
    classes, class_to_idx = make_classes(map_dir)

    DET_image_dir = os.path.join(root, 'Data/DET/val')
    DET_annotation_dir = os.path.join(root, 'Annotations/DET/val')
    DET_samples = make_DET_dataset(DET_image_dir, DET_annotation_dir, class_to_idx, 'val')

    print(wnid_to_tags('n07555863'))