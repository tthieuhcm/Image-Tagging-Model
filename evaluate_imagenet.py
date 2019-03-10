import os

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn as nn

from dataloader import ImageNet_Dataset


def F_score(ground_truth, predicted):
    true_positives = set(ground_truth).intersection(set(predicted))
    false_positives = set(predicted) - set(true_positives)
    false_negatives = set(ground_truth) - set(true_positives)
    return len(true_positives), len(false_positives), len(false_negatives)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/Data/CLS-LOC/'
# data_folder = './ILSVRC/Data/CLS-LOC/'

number_of_tags = 1788
batch_size = 64
num_worker = 4

valdir = os.path.join(data_folder, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

model_ft = models.densenet121(pretrained=True)
num_ftrs = model_ft.classifier.in_features
model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)
model.load_state_dict(torch.load("models/densenet121-BCE-30.ckpt"))
model.eval()

val_dataset = ImageNet_Dataset(valdir,
                     loader=default_loader,
                     transform=transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         normalize,
                     ]),
                     dataset_type='val')
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_worker,
    pin_memory=False)

with torch.no_grad():
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    list_threshold = np.arange(0, 1, 0.1)

    list_precisions = list()
    list_recalls = list()
    list_F_score = list()

    for threshold in list_threshold:
        for input, target in tqdm(val_loader):
            input = input.cuda().unsqueeze(0)
            target = target.cuda()

            outputs = model(input)
            t = Variable(torch.Tensor([threshold])).to(device)
            preds = (outputs > t).float() * 1
            preds = preds.type(torch.IntTensor).cpu().numpy()

            for j in range(input.size()[0]):
                pred_list = list()
                ground_truth_list = list()
                for num_item, item in enumerate(preds[j]):
                    if item != 0:
                        pred_list.append(val_dataset.inx_to_class[num_item])
                for num_item, item in enumerate(target[j]):
                    if item != 0:
                        ground_truth_list.append(val_dataset.inx_to_class[num_item])

                true_positives, false_positives, false_negatives = F_score(ground_truth_list, pred_list)

                all_true_positives += true_positives
                all_false_positives += false_positives
                all_false_negatives += false_negatives

        precision = all_true_positives/(all_true_positives+all_false_positives+1e-10)
        recall = all_true_positives/(all_true_positives+all_false_negatives+1e-10)
        F1 = 2*precision*recall/(precision+recall+1e-10)
        list_F_score.append(F1)
        list_precisions.append(precision)
        list_recalls.append(recall)

    plt.plot(list_threshold, list_F_score, 'or')
    plt.show()
    print(list_precisions)
    print(list_recalls)
