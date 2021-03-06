from graphviz import Digraph
from nltk.corpus import wordnet

from tag_statistic import tag_statistic
from utils import make_classes, find_hypernyms_names_from_wnid
import os


def visualize_tree(root_name, data_folder, tag_stats):
    map_dir = os.path.join(data_folder, 'devkit/data')

    tags, tag_to_idx = make_classes(map_dir)

    g = Digraph('G', filename=root_name)

    for tag in tags:
        parents_names = find_hypernyms_names_from_wnid(tag)
        tag_name = wordnet.synset_from_pos_and_offset('n', int(tag[1:])).lemma_names()[0]
        if root_name in parents_names or tag_name == root_name:
            tag_count = tag_stats[tag]
            g.node(tag_name + ' (' + tag + ') ('+str(int(tag_count))+')')

    for tag in tags:
        parents_names = find_hypernyms_names_from_wnid(tag)
        if root_name in parents_names:
            tag_synset = wordnet.synset_from_pos_and_offset('n', int(tag[1:]))
            tag_name = tag_synset.lemma_names()[0]
            tag_count = tag_stats[tag]

            if tag_name != root_name:
                parent_tag = 'n' + str(tag_synset.hypernyms()[0].offset()).zfill(8)
                parent_tag_name = wordnet.synset_from_pos_and_offset('n', int(parent_tag[1:])).lemma_names()[0]
                parent_tag_count = tag_stats[parent_tag]

                g.edge(parent_tag_name + ' (' + parent_tag + ') ('+str(int(parent_tag_count))+')',
                       tag_name + ' (' + tag + ') ('+str(int(tag_count))+')')

    g.view()


if __name__ == "__main__":
    data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/'
    tag_stats = tag_statistic(data_folder)
    # visualize_tree('animal', data_folder, tag_stats)
    visualize_tree('artifact', data_folder, tag_stats)
    # visualize_tree('communication', data_folder, tag_stats)
    # visualize_tree('food', data_folder, tag_stats)
    # visualize_tree('fungus', data_folder, tag_stats)
    # visualize_tree('geological_formation', data_folder, tag_stats)
    # visualize_tree('natural_object', data_folder, tag_stats)
    # visualize_tree('person', data_folder, tag_stats)
    # visualize_tree('plant', data_folder, tag_stats)
    # visualize_tree('sphere', data_folder, tag_stats)
    # visualize_tree('toilet_tissue', data_folder, tag_stats)
