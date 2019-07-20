import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

import torch.nn as nn
import numpy as np
from dataloader import ImageNet_Dataset


def F_score(ground_truth, predicted):
    true_positives = set(ground_truth).intersection(set(predicted))
    false_positives = set(predicted) - set(true_positives)
    false_negatives = set(ground_truth) - set(true_positives)
    return len(true_positives), len(false_positives), len(false_negatives)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'
    batch_size = 1
    num_worker = 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = ImageNet_Dataset(data_folder,
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
    number_of_tags = len(val_dataset.classes)

    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
    model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)
    model.load_state_dict(torch.load("models/densenet121-DenoisedWeightedBCE-51.ckpt"))
    model.eval()
    for param in model[0].parameters():
        param.requires_grad = False

    with torch.no_grad():
        all_true_positives = [0] * 11
        all_false_positives = [0] * 11
        all_false_negatives = [0] * 11
        list_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

        for input, target in tqdm(val_loader):
            input = input.cuda()
            target = target.cuda()
            outputs = model(input)

            for i, threshold in enumerate(list_threshold):
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

                    all_true_positives[i] += true_positives
                    all_false_positives[i] += false_positives
                    all_false_negatives[i] += false_negatives

        for i, threshold in enumerate(list_threshold):
            precision = all_true_positives[i] / (all_true_positives[i] + all_false_positives[i] + 1e-10)
            recall = all_true_positives[i] / (all_true_positives[i] + all_false_negatives[i] + 1e-10)
            F1 = 2 * precision * recall / (precision + recall + 1e-10)
            print('threshold: ', threshold)
            print('precision: ', precision)
            print('recall: ', recall)
            print('F1: ', F1)

#
# def F_score(ground_truth, predicted):
#     true_positives = np.logical_and(ground_truth, predicted)
#     false_positives = np.logical_xor(predicted, true_positives)
#     false_negatives = np.logical_xor(ground_truth, true_positives)
#     return true_positives, false_positives, false_negatives
#
#
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'
#     batch_size = 1
#     num_worker = 1
#
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#
#     val_dataset = ImageNet_Dataset(data_folder,
#                                    loader=default_loader,
#                                    transform=transforms.Compose([
#                                        transforms.Resize(256),
#                                        transforms.CenterCrop(224),
#                                        transforms.ToTensor(),
#                                        normalize,
#                                    ]),
#                                    dataset_type='val')
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_worker,
#         pin_memory=False)
#     number_of_tags = len(val_dataset.classes)
#
#     model_ft = models.densenet121(pretrained=True)
#     num_ftrs = model_ft.classifier.in_features
#     model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
#     model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)
#     model.load_state_dict(torch.load("models/densenet121-weightedBCE-47.ckpt"))
#     model.eval()
#     for param in model[0].parameters():
#         param.requires_grad = False
#
#     with torch.no_grad():
#         class_true_positives = np.zeros((10, number_of_tags))
#         class_false_positives = np.zeros((10, number_of_tags))
#         class_false_negatives = np.zeros((10, number_of_tags))
#         list_threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
#         for input, target in tqdm(val_loader):
#             input = input.cuda()
#             target = target.cuda()
#             outputs = model(input)
#
#             for i, threshold in enumerate(list_threshold):
#                 t = Variable(torch.Tensor([threshold])).to(device)
#                 preds = (outputs > t).float() * 1
#                 preds = preds.type(torch.IntTensor).cpu().numpy()
#
#                 sample_true_positives, sample_false_positives, sample_false_negatives = F_score(target.cpu().numpy(), preds)
#
#                 class_true_positives [i, :] += sample_true_positives.flatten()
#                 class_false_positives[i, :] += sample_false_positives.flatten()
#                 class_false_negatives[i, :] += sample_false_negatives.flatten()
#
#         for i, threshold in enumerate(list_threshold):
#             epsilon = np.array([1e-10]*number_of_tags)
#             precision = class_true_positives[i] / (class_true_positives[i] + class_false_positives[i] + epsilon)
#             recall = class_true_positives[i] / (class_true_positives[i] + class_false_negatives[i] + epsilon)
#             F1 = 2 * precision * recall / (precision + recall + epsilon)
#
#             print('threshold: ', threshold)
#             print('precision: ', np.sum(precision)/number_of_tags)
#             print('recall: ', np.sum(recall)/number_of_tags)
#             print('F1: ', np.sum(F1)/number_of_tags)
#
#             file = open("/home/tthieuhcm/Desktop/eval_results.txt", "a+")
#             for i in range(number_of_tags):
#                 file.write('precision: ' + precision[i] + "\n")
#                 file.write('recall: ' + recall[i] + "\n")
#                 file.write('F1: ' + F1[i] + "\n")
