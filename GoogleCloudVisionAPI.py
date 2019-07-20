# export API_KEY=AIzaSyBQ64kGvVJkLqp-yPb6iRkF8SjKeA5cKI4
# export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"


import io
from google.cloud import vision
from google.cloud.vision import types
import os
import torch
from torch.autograd import Variable
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader

import torch.nn as nn
from nltk.corpus import wordnet


def make_Google_Vision_tags():
    CLS_LOC_data_dir = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/Data/CLS-LOC/train/'
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/tthieuhcm/PycharmProjects/Image-Tagging-Model/NeuralImageCaption-63a40bb0cc2a.json"
    client = vision.ImageAnnotatorClient()

    list_folder = os.listdir(CLS_LOC_data_dir)
    sorted(list_folder)
    for i in range(1000):
        image_folder = CLS_LOC_data_dir + list_folder[i]
        print(list_folder[i])
        for root, _, file_names in sorted(os.walk(image_folder)):
            sorted(file_names)
            file_names = file_names[:5]
            for file_name in sorted(file_names):
                try:
                    image_path = image_folder + '/' + file_name
                    # Loads the image into memory
                    with io.open(image_path, 'rb') as image_file:
                        content = image_file.read()

                    image = types.Image(content=content)

                    # Performs label detection on the image file
                    response = client.label_detection(image=image)
                    labels = response.label_annotations

                    file = open("/home/tthieuhcm/Desktop/google_results.txt", "a+")
                    for label in labels:
                        file.write(label.description + "\n")
                except Exception as e:
                    print(e)
                    print(file_name)
                    continue


def make_common_tag_set():
    google_tag_set = set([line.rstrip().lower() for line in
                open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Google_Cloud_Vision_tags.txt')])
    # our_tag_set = set([line.rstrip().replace("_", " ").lower() for line in
    #                     open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/all_tag_names.txt')])

    from new_utils import make_classes
    from tag_statistic import get_name

    data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/'
    map_dir = os.path.join(data_folder, 'devkit/data')
    classes, class_to_idx = make_classes(map_dir)

    our_refined_tag_set = set([get_name(wnid).replace("_", " ").lower() for wnid in classes])

    common_tag_set = sorted(google_tag_set.intersection(our_refined_tag_set))
    for item in common_tag_set:
        print(item)


def get_test_result():
    test_folder = '/home/tthieuhcm/Desktop/test/'
    os.environ[
        "GOOGLE_APPLICATION_CREDENTIALS"] = "/home/tthieuhcm/PycharmProjects/Image-Tagging-Model/NeuralImageCaption-63a40bb0cc2a.json"
    client = vision.ImageAnnotatorClient()

    similar_tag_list = [line.rstrip() for line in
                        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Google_refined_similar_tag_list.txt')]

    for root, _, file_names in sorted(os.walk(test_folder)):
        # for file_name in sorted(file_names):
        #     try:
        #         image_path = test_folder + file_name
        #         # Loads the image into memory
        #         with io.open(image_path, 'rb') as image_file:
        #             content = image_file.read()
        #
        #         image = types.Image(content=content)
        #
        #         # Performs label detection on the image file
        #         response = client.label_detection(image=image)
        #         labels = response.label_annotations
        #
        #         file = open("/home/tthieuhcm/Desktop/Google_refined_test_results.txt", "a+")
        #         # file.write("File name: " + file_name + "\n")
        #         for label in labels:
        #             if label.description.replace("_", " ").lower() in similar_tag_list:
        #                 file.write(label.description.lower()+", ")
        #         file.write("\n")
        #
        #     except Exception as e:
        #         print(e)
        #         print(file_name)
        #         continue

        # from utils import make_classes
        from new_utils import make_classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/'

        # number_of_tags = 1788
        number_of_tags = 1161

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
        model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)
        # model.load_state_dict(torch.load("models/densenet121-weightedBCE-47.ckpt"))
        model.load_state_dict(torch.load("models/densenet121-DenoisedWeightedBCE-51.ckpt"))
        was_training = model.training
        model.eval()
        map_dir = os.path.join(data_folder, 'devkit/data')
        classes, class_to_idx = make_classes(map_dir)
        inx_to_class = {v: k for k, v in class_to_idx.items()}
        with torch.no_grad():
            threshold = 0.05
            for file_name in sorted(file_names):
                image_path = test_folder + '/' + file_name

                image = default_loader(image_path)
                image = transform(image)

                input = image.cuda().unsqueeze(0)
                outputs = model(input)
                t = Variable(torch.Tensor([threshold])).to(device)
                preds = (outputs > t).float() * 1
                preds = preds.type(torch.IntTensor).cpu().numpy()

                file = open("/home/tthieuhcm/Desktop/our_refined_test_results_vs_Google.txt", "a+")
                for num_item, item in enumerate(preds[0]):
                    if item != 0:
                        tag_ID = inx_to_class[num_item]
                        tag_name = wordnet.synset_from_pos_and_offset('n', int(tag_ID[1:])).lemmas()[0].name().replace("_", " ").lower()
                        if tag_name in similar_tag_list:
                            file.write(tag_name + ", ")
                file.write("\n")

            model.train(mode=was_training)


def calculate_Fscore_for_test_image():
    test_folder = '/home/tthieuhcm/Desktop/test/'
    similar_tag_list  = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Google_similar_tag_list.txt')]
    confused_tag_list = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Google_confused_tag_list.txt')]
    our_result        = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/our_test_results_vs_Google.txt')]
    Google_result     = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Google_test_results.txt')]
    ground_truth      = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/ground_truth_for_Google.txt')]

    similar_tag_list  = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Google_refined_similar_tag_list.txt')]
    our_result        = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/our_refined_test_results_vs_Google.txt')]

    similar_tag_list = list(set(similar_tag_list) - set(confused_tag_list))

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for root, _, file_names in sorted(os.walk(test_folder)):
        for i, file_name in enumerate(sorted(file_names)):
            result = list(set(our_result[i].split(', ')))
            ground_truth_ith_image = ground_truth[i].split(', ')
            for tag in ground_truth_ith_image:
                if tag not in similar_tag_list:
                    ground_truth_ith_image.remove(tag)

            if len(ground_truth_ith_image) == 0:
                print(file_name)
                continue

            for tag in result:
                if tag is not '':
                    if tag in similar_tag_list:
                        if tag in ground_truth_ith_image:
                            true_positives += 1
                            ground_truth_ith_image.remove(tag)
                        else:
                            false_positives += 1

            for tag in ground_truth_ith_image:
                if tag in similar_tag_list:
                    false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)

    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('F1: ' + str(F1))


if __name__ == "__main__":
    # Step 1
    # make_Google_Vision_tags()

    # Step 2
    # make_common_tag_set()

    # Step 3
    # get_test_result()

    # Step 4
    calculate_Fscore_for_test_image()