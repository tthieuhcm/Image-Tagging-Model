import os
import time
import torch
from torch.autograd import Variable
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader

import torch.nn as nn
import requests
from utils import make_classes
from nltk.corpus import wordnet


def make_Microsoft_Vision_tags():
    CLS_LOC_data_dir = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/Data/CLS-LOC/train/'
    subscription_key = "1999d41f41134b56bd901e5c5346b4c2"
    assert subscription_key
    vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

    list_folder = os.listdir(CLS_LOC_data_dir)
    sorted(list_folder)
    for iter in range(32):
        time.sleep(90)
        print('iter: ', iter + 218)
        for i in range(4):
            image_folder = CLS_LOC_data_dir + list_folder[(iter + 218)*4 + i]
            print(list_folder[(iter + 218)*4 + i])
            for root, _, file_names in sorted(os.walk(image_folder)):
                sorted(file_names)
                file_names = file_names[:5]
                for file_name in sorted(file_names):
                    try:
                        analyze_url = vision_base_url + "analyze"
                        # Set image_path to the local path of an image that you want to analyze.
                        # image_path = "/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/Data/DET/train/ILSVRC2013_train_extra0/ILSVRC2013_train_000009" + str(
                        #     90 + i) + ".JPEG"
                        image_path = image_folder + '/' + file_name

                        # Read the image into a byte array
                        image_data = open(image_path, "rb").read()
                        headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                                   'Content-Type': 'application/octet-stream'}
                        params = {'visualFeatures': 'Description'}
                        response = requests.post(
                            analyze_url, headers=headers, params=params, data=image_data)
                        response.raise_for_status()

                        # The 'analysis' object contains various fields that describe the image. The most
                        # relevant caption for the image is obtained from the 'description' property.
                        analysis = response.json()
                        file = open("/home/tthieuhcm/Desktop/microsoft_results.txt", "a+")
                        for item in analysis["description"]['tags']:
                            file.write(item + "\n")
                        # image_caption = analysis["description"]["captions"][0]["text"].capitalize()
                        #
                        # # Display the image and overlay it with the caption.
                        # image = Image.open(BytesIO(image_data))
                        # plt.imshow(image)
                        # plt.axis("off")
                        # _ = plt.title(image_caption, size="x-large", y=-0.1)
                    except:
                        print(image_path)
                        continue


def get_test_result():
    test_folder = '/home/tthieuhcm/Desktop/test/'
    similar_tag_list = [line.rstrip() for line in
                        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/similar_tags_old.txt')]
    # similar_tag_list = [line.rstrip() for line in
    #                     open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/similar_tags_new.txt')]
    confused_tag_list = [line.rstrip() for line in
                        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/confused_tag.txt')]
    similar_tag_list = list(set(similar_tag_list) - set(confused_tag_list))
    for root, _, file_names in sorted(os.walk(test_folder)):
        sorted(file_names)
        # time.sleep(4)
        # try:
        #     subscription_key = "c320fb27cd9d4b24828ec90e5b684fbc"
        #     assert subscription_key
        #     vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"
        #     analyze_url = vision_base_url + "analyze"
        #
        #     # Set image_path to the local path of an image that you want to analyze.
        #     file_name = 'hummingbird-bird-animal.jpg'
        #
        #     # Read the image into a byte array
        #     image_data = open(image_path, "rb").read()
        #     headers = {'Ocp-Apim-Subscription-Key': subscription_key,
        #                'Content-Type': 'application/octet-stream'}
        #     params = {'visualFeatures': 'Description'}
        #     response = requests.post(
        #         analyze_url, headers=headers, params=params, data=image_data)
        #     response.raise_for_status()
        #
        #     # The 'analysis' object contains various fields that describe the image. The most
        #     # relevant caption for the image is obtained from the 'description' property.
        #     analysis = response.json()
        #     print("File name: ", file_name)
        #     for item in analysis["description"]['tags']:
        #         if item in similar_tag_list:
        #             print(item)
        #
        # except:
        #     print("Can't load file: ", file_name)
        #     continue
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/'

        number_of_tags = 1788
        # number_of_tags = 1161

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
        model.load_state_dict(torch.load("models/densenet121-weightedBCE-47.ckpt"))
        was_training = model.training
        model.eval()
        map_dir = os.path.join(data_folder, 'devkit/data')
        classes, class_to_idx = make_classes(map_dir)
        inx_to_class = {v: k for k, v in class_to_idx.items()}
        with torch.no_grad():
            threshold = 0.05
            for file_name in sorted(file_names):
                print("File name: ", file_name)
                image_path = test_folder + '/' + file_name

                image = default_loader(image_path)
                image = transform(image)

                input = image.cuda().unsqueeze(0)
                outputs = model(input)
                t = Variable(torch.Tensor([threshold])).to(device)
                preds = (outputs > t).float() * 1
                preds = preds.type(torch.IntTensor).cpu().numpy()

                for num_item, item in enumerate(preds[0]):
                    if item != 0:
                        tag_ID = inx_to_class[num_item]
                        tag_name = wordnet.synset_from_pos_and_offset('n', int(tag_ID[1:])).lemmas()[0].name()
                        if tag_name in similar_tag_list:
                            # print('tag: {}, confident: {}'.format(tag_name,
                            #                                       "%.4f" % outputs[0][num_item].item()))
                            print('{}'.format(tag_name))

            model.train(mode=was_training)


def calculate_Fscore_for_test_image():
    test_folder = '/home/tthieuhcm/Desktop/test/'

    similar_tag_list = [line.rstrip() for line in
                        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Microsoft_similar_tag_list.txt')]
    confused_tag_list = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/confused_tag.txt')]
    our_result = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/our_test_result_vs_Microsoft.txt')]
    Microsoft_Vision_result = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Microsoft_Vision_test_result.txt')]

    similar_tag_list = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/Microsoft_refined_similar_tag_list.txt')]
    our_result = [line.rstrip() for line in
        open('/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Desktop/our_refined_test_result_vs_Microsoft.txt')]

    similar_tag_list = list(set(similar_tag_list) - set(confused_tag_list))
    similar_tag_list.sort()
    # Macro average
    # tag_to_pos = dict(zip(similar_tag_list, range(len(similar_tag_list))))
    # true_positives = np.zeros(len(similar_tag_list))
    # false_positives = np.zeros(len(similar_tag_list))
    # false_negatives = np.zeros(len(similar_tag_list))
    #
    # for root, _, file_names in sorted(os.walk(test_folder)):
    #     sorted(file_names)
    #     for i, file_name in enumerate(sorted(file_names)):
    #         result = Microsoft_Vision_result[i].split(', ')
    #         ground_truth = os.path.splitext(file_name)[0].split('-')
    #         for tag in result:
    #             if tag is not '':
    #                 if tag in ground_truth:
    #                     true_positives[tag_to_pos[tag]] += 1
    #                     ground_truth.remove(tag)
    #                 else:
    #                     false_positives[tag_to_pos[tag]] += 1
    #         for tag in ground_truth:
    #             false_negatives[tag_to_pos[tag]] += 1
    #
    # epsilon = np.array([1e-10] * len(similar_tag_list))
    # precision = true_positives / (true_positives + false_positives + epsilon)
    # recall = true_positives / (true_positives + false_negatives + epsilon)
    # F1 = 2 * precision * recall / (precision + recall + epsilon)
    #
    # print('precision: {}'.format(np.sum(precision)/len(similar_tag_list)))
    # print('recall: {}'.format(np.sum(recall)/len(similar_tag_list)))
    # print('F1: {}'.format(np.sum(F1)/len(similar_tag_list)))

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for root, _, file_names in sorted(os.walk(test_folder)):
        sorted(file_names)
        for i, file_name in enumerate(sorted(file_names)):
            result = list(set(Microsoft_Vision_result[i].split(', ')))
            ground_truth = os.path.splitext(file_name)[0].split('-')

            for tag in ground_truth:
                if tag not in similar_tag_list:
                    ground_truth.remove(tag)

            if len(ground_truth) == 0:
                print(file_name)
                continue

            for tag in result:
                if tag is not '':
                    if tag in similar_tag_list:
                        if tag in ground_truth:
                            true_positives += 1
                            ground_truth.remove(tag)
                        else:
                            false_positives += 1

            for tag in ground_truth:
                if tag in similar_tag_list:
                    false_negatives += 1

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)

    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print('F1: ' + str(F1))


if __name__ == "__main__":
    # Step 3
    # get_test_result()

    # Step 4
    calculate_Fscore_for_test_image()