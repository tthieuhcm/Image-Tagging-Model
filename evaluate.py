import time

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

from train import MIR_FLICKR_Dataset, visualize_model, test_image
import matplotlib.pyplot as plt
import torch.nn as nn


def F_score(ground_truth, predicted):
    true_positives = set(ground_truth).intersection(set(predicted))
    false_positives = set(predicted) - set(true_positives)
    false_negatives = set(ground_truth) - set(true_positives)
    return len(true_positives), len(false_positives), len(false_negatives)

# start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('./data_processing/taxonomy.txt') as all_tags:
    tag_list = [item.rstrip() for item in all_tags.readlines()]

number_of_images = 25000
number_of_tags = len(tag_list)
# model = torch.load("./data_processing/densenet161+sigmoid+BCE.pt") # 0.5
# model = torch.load("./data_processing/densenet161.pt") # 0.6
# model = torch.load("./data_processing/dense.pt") # 0.6
# model = torch.load("./data_processing/inception.pt") # 0.7
model_ft = models.densenet121(pretrained=True)
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)

model.load_state_dict(torch.load("models/densenet121+conv0+norm0+dl16conv2+dlnorm2+FC+Sigmoid+BCE-34.ckpt"))
model.eval()

img_labels = np.array([[0 for i in range(number_of_tags)] for j in range(number_of_images)])

with open('./data_processing/mirflickr_labels.txt') as info:
    lines = info.readlines()
    img_name = np.array([img_info.split()[0] for img_info in lines])
    for i, img_label in enumerate(lines):
        img_tags = img_label.split()[1:]
        tag_index = [tag_list.index(tag) for tag in img_tags]
        for index in tag_index:
            img_labels[i][index] = 1


train_split = .7
validation_split = .15

train = int(np.floor(train_split * number_of_images))
validation = int(np.floor(validation_split * number_of_images))

mir_flickr_test = MIR_FLICKR_Dataset(img_name[train + validation:], img_labels[train + validation:],
                                     root_dir='/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/mirflickr',
                                     transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                   transforms.ToTensor()]))

test_loader = DataLoader(mir_flickr_test, batch_size=64, shuffle=False, num_workers=4)

with torch.no_grad():
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    list_threshold = np.arange(0, 1, 0.1)
    list_F_score = list()

    for threshold in list_threshold:
        for sample_batched in tqdm(test_loader):
            inputs, labels = sample_batched['image'].to(device), sample_batched['labels']

            outputs = model(inputs)
            t = Variable(torch.Tensor([threshold])).to(device)
            preds = (outputs > t).float() * 1
            preds = preds.type(torch.IntTensor).cpu().numpy()

            for j in range(inputs.size()[0]):
                pred_list = list()
                ground_truth_list = list()
                for num_item, item in enumerate(preds[j]):
                    if item != 0:
                        pred_list.append(tag_list[item * num_item])
                for num_item, item in enumerate(labels[j]):
                    if item != 0:
                        ground_truth_list.append(tag_list[item * num_item])

                true_positives, false_positives, false_negatives = F_score(ground_truth_list, pred_list)

                all_true_positives += true_positives
                all_false_positives += false_positives
                all_false_negatives += false_negatives

        precision = all_true_positives/(all_true_positives+all_false_positives+1e-10)
        recall = all_true_positives/(all_true_positives+all_false_negatives+1e-10)
        F1 = 2*precision*recall/(precision+recall+1e-10)
        list_F_score.append(F1)
        print(precision)
        print(recall)
        print(F1)

    # plt.plot(list_threshold, list_F_score)
    plt.plot(list_threshold, list_F_score, 'or')
    plt.show()

# visualize_model(device, model, tag_list, test_loader, num_images=4)

# test_image(device, model, tag_list, '/home/tthieuhcm/Downloads/image1.png')
# print(time.time()-start)