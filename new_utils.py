import os
import xml
import torch
from nltk.corpus import wordnet


def wnid_to_tags(wnid):
    """
    :param wnid: WordNet ID. Examples: n07555863, n03642806, n03770439
    :return: List of WordNet IDs which are related to the main WordNet ID
    """
    useless_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/useless_tag_IDs.txt'
    # useless_tags_file_path = ''
    useless_tags_list = list()
    list_of_tag = list()

    if useless_tags_file_path:
        useless_tags_list = [line.rstrip() for line in open(useless_tags_file_path)]

    tag = wordnet.synset_from_pos_and_offset('n', int(wnid[1:]))
    list_of_tag.append(wnid)

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
        tagID = 'n'+str(tag.offset()).zfill(8)
        if tagID not in useless_tags_list:
            list_of_tag.append(tagID)

    return list_of_tag


def find_hypernym_IDs_from_wnid(main_tag_ID, list_of_tag_IDs):
    """
    :param main_tag_ID: the main tag ID
    :param list_of_tag_IDs: list of all relevant tag IDs
    :return: list of all tag IDs and the relevant tag IDs
    """
    noise_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/noise_tag_IDs.txt'
    useless_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/useless_tag_IDs.txt'
    # noise_tags_file_path = ''
    # useless_tags_file_path = ''
    useless_tags_list = list()
    noise_tags_list = list()

    if useless_tags_file_path:
        useless_tags_list = [line.rstrip() for line in open(useless_tags_file_path)]

    if noise_tags_file_path:
        noise_tags_list = [line.rstrip() for line in open(noise_tags_file_path)]

    tag = wordnet.synset_from_pos_and_offset('n', int(main_tag_ID[1:]))

    if main_tag_ID in noise_tags_list:
        return list_of_tag_IDs

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
        tag_ID = 'n'+str(tag.offset()).zfill(8)

        if tag_ID not in useless_tags_list:
            list_of_tag_IDs.append(tag_ID)

    return list_of_tag_IDs


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
    noise_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/noise_tag_IDs.txt'
    useless_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/useless_tag_IDs.txt'
    # noise_tags_file_path = ''
    # useless_tags_file_path = ''
    useless_tags_list = list()
    noise_tags_list = list()

    if useless_tags_file_path:
        useless_tags_list = [line.rstrip() for line in open(useless_tags_file_path)]

    if noise_tags_file_path:
        noise_tags_list = [line.rstrip() for line in open(noise_tags_file_path)]

    with open(map_dir+'/map_clsloc.txt', 'r') as cls_loc, \
            open(map_dir+'/map_det.txt', 'r') as det:
        # classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes = list()
        cls_loc_lines = cls_loc.readlines()
        det_lines = det.readlines()
        for line in cls_loc_lines:
            tag_ID = line.split()[0]
            if tag_ID not in useless_tags_list and tag_ID not in noise_tags_list:
                classes.append(tag_ID)
        for line in det_lines:
            tag_ID = line.split()[0]
            if tag_ID not in useless_tags_list and tag_ID not in noise_tags_list:
                classes.append(tag_ID)

    classes = list(set(classes))
    classes.sort()

    for class_id in classes:
        find_hypernym_IDs_from_wnid(class_id, classes)

    classes.append('n00017222')    # n00017222 : plant
    classes.append('n03956922')    # n03956922 : plant
    classes.append('n09287968')    # n09287968 : geological_formation
    classes.append('n00019128')    # n00019128 : natural_object
    # classes.append('n00021939')    # n00021939 : artifact
    classes.append('n12992868')    # n12992868 : fungus
    classes.append('n00007846')    # n00007846 : person
    classes.append('n00015388')    # n00015388 : animal
    classes.append('n00033020')    # n00033020 : communication
    classes.append('n15075141')    # n15075141 : toilet_tissue
    # classes.append('n13899200')    # n13899200 : sphere
    classes.append('n00021265')    # n00021265 : food
    classes.append('n07555863')    # n07555863 : food

    classes = list(set(classes))
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_CLS_LOC_dataset(CLS_LOC_data_dir, CLS_LOC_annotation_dir, class_to_idx, dataset_type):
    noise_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/noise_tag_IDs.txt'
    # noise_tags_file_path = ''
    noise_tags_list = list()

    images = []
    CLS_LOC_data_dir = os.path.expanduser(CLS_LOC_data_dir)

    if noise_tags_file_path:
        noise_tags_list = [line.rstrip() for line in open(noise_tags_file_path)]

    for folder in sorted(os.listdir(CLS_LOC_data_dir)):
        if folder in noise_tags_list:
            continue

        subdir = os.path.join(CLS_LOC_data_dir, folder)
        if not os.path.isdir(subdir):
            continue

        for root, _, fnames in sorted(os.walk(subdir)):
            sorted(fnames)
            # if dataset_type == 'train':
            #     fnames = fnames[100:]
            # elif dataset_type == 'test':
            #     fnames = fnames[:100]

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
    noise_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/noise_tag_IDs.txt'
    useless_tags_file_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/useless_tag_IDs.txt'
    # noise_tags_file_path = ''
    # useless_tags_file_path = ''
    noise_tags_list = list()
    useless_tags_list = list()

    images = []
    annotation_dir = os.path.expanduser(DET_annotation_dir)

    if noise_tags_file_path:
        noise_tags_list = [line.rstrip() for line in open(noise_tags_file_path)]
    if useless_tags_file_path:
        useless_tags_list = [line.rstrip() for line in open(useless_tags_file_path)]

    if dataset_type == 'val':
        for root, _, fnames in sorted(os.walk(annotation_dir)):
            sorted(fnames)

            for fname in sorted(fnames):
                image_path = os.path.join(DET_data_dir, fname[:-3] + 'JPEG')
                annotation_path = os.path.join(annotation_dir, fname)
                annotation = xml.etree.ElementTree.parse(annotation_path).getroot()
                list_target = list()

                for object in annotation.findall('object'):
                    tagID = object.find('name').text
                    if tagID not in noise_tags_list and tagID not in useless_tags_list:
                        list_target.append(tagID)

                list_target = list(set(list_target))

                if len(list_target) == 0:
                    continue

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
                    if subfolder in noise_tags_list:
                        continue

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
                                tagID = object.find('name').text
                                if tagID not in noise_tags_list and tagID not in useless_tags_list:
                                    list_target.append(tagID)

                            list_target = list(set(list_target))

                            if len(list_target) == 0:
                                continue

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
                            tagID = object.find('name').text
                            if tagID not in noise_tags_list and tagID not in useless_tags_list:
                                list_target.append(tagID)

                        list_target = list(set(list_target))

                        if len(list_target) == 0:
                            continue

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


def list_all_tag_names(map_dir):
    tagIDs, tagIDs_to_idx = make_classes(map_dir)
    list_tag_names = list()

    for tagID in tagIDs:
        tag = wordnet.synset_from_pos_and_offset('n', int(tagID[1:]))
        list_tag_names.append(tag.lemma_names()[0])# + ': ' + tagID)
    return list_tag_names


def list_sub_tag_names(map_dir):
    with open(map_dir+'/map_clsloc.txt', 'r') as cls_loc, \
            open(map_dir+'/map_det.txt', 'r') as det:
        # classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        child_tag_IDs = list()
        cls_loc_lines = cls_loc.readlines()
        det_lines = det.readlines()
        for line in cls_loc_lines:
            child_tag_IDs.append(line.split()[0])
        for line in det_lines:
            child_tag_IDs.append(line.split()[0])

    all_tag_IDs = list(set(child_tag_IDs))
    all_tag_IDs.sort()

    for tag_ID in all_tag_IDs:
        find_hypernym_IDs_from_wnid(tag_ID, all_tag_IDs)

    all_tag_IDs.append('n09287968')    # n09287968 : geological_formation
    all_tag_IDs.append('n00019128')    # n00019128 : natural_object
    all_tag_IDs.append('n00021939')    # n00021939 : artifact
    all_tag_IDs.append('n00033020')    # n00033020 : communication

    all_sub_tag_IDs = list(set(all_tag_IDs)-set(child_tag_IDs))
    all_sub_tag_IDs.sort()

    sub_tag_names_list = list()

    for tagID in all_sub_tag_IDs:
        tag = wordnet.synset_from_pos_and_offset('n', int(tagID[1:]))
        sub_tag_names_list.append(tag.lemma_names()[0])
    return sub_tag_names_list


if __name__ == "__main__":
    # root = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'
    # map_dir = os.path.join(root, 'devkit/data')
    # classes, class_to_idx = make_classes(map_dir)
    # tag_names_list = list_all_tag_names(map_dir)
    # sub_tag_names_list = list_sub_tag_names(map_dir)
    tag_names_list = [line.rstrip() for line in open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/all_tag_names.txt')]
    Microsoft_tags_list = [line.rstrip() for line in open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Microsoft_Vision_tags.txt')]
    similar_tag_list = [w for w in set(tag_names_list) if w in Microsoft_tags_list]
    # noise_tags_list = [wordnet.synset_from_pos_and_offset('n', int(line.rstrip()[1:])).lemma_names()[0] for line in open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/noise_tag_IDs.txt')]
    # useless_tags_list = [wordnet.synset_from_pos_and_offset('n', int(line.rstrip()[1:])).lemma_names()[0] for line in open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/useless_tag_IDs.txt')]

    # similar_sub_tag_list = [w for w in set(sub_tag_names_list) if w in Microsoft_tags_list]
    # rest_of_sub_tag_list = list(set(sub_tag_names_list) - set(similar_sub_tag_list))
    similar_tag_list.sort()
    # rest_of_sub_tag_list.sort()
    # similar_sub_tag_list.sort()
    # noise_tags_list.sort()
    print(similar_tag_list)
    # print(similar_sub_tag_list)
    # print(rest_of_sub_tag_list)
    # print(noise_tags_list)
    # print(useless_tags_list)
    # DET_image_dir = os.path.join(root, 'Data/DET/val')
    # DET_annotation_dir = os.path.join(root, 'Annotations/DET/val')
    # DET_samples = make_DET_dataset(DET_image_dir, DET_annotation_dir, class_to_idx, 'val')
    #
    # print(wnid_to_tags('n07555863'))