# from scipy.io import loadmat
# x = loadmat('ILSVRC/devkit/data/ILSVRC2015_det_validation_ground_truth.mat')
# print(x)

from nltk.corpus import wordnet


def wnid_to_tags(wnid):
    list_of_tag = list()

    tag = wordnet.synset_from_pos_and_offset('n', wnid)
    list_of_tag.append(tag.lemma_names()[0])

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
        list_of_tag.append(tag.lemma_names()[0])
    return list_of_tag


if __name__ == "__main__":
    print(wnid_to_tags(2672831))