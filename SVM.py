import time

from libsvm.python.svmutil import *
import os
import torch
import numpy as np
from PIL import ImageFile

from utils import wnid_to_tags

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.models as models
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def make_CLS_LOC_dataset_for_SVM(CLS_LOC_data_dir, class_name, dataset_type):
    images = []
    CLS_LOC_data_dir = os.path.expanduser(CLS_LOC_data_dir)
    for folder in sorted(os.listdir(CLS_LOC_data_dir)):
        subdir = os.path.join(CLS_LOC_data_dir, folder)
        if not os.path.isdir(subdir):
            continue
        list_target = wnid_to_tags(folder)
        inClass = 0

        for target in list_target:
            if target == class_name:
                # print(list_target)
                inClass = 1

        for root, _, fnames in sorted(os.walk(subdir)):
            sorted(fnames)
            if dataset_type == 'train' or dataset_type == 'val':
                fnames = fnames[0:10]
            elif dataset_type == 'test':
                fnames = fnames[10:20]

            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, inClass)
                images.append(item)

    return images


def ImageNet_Dataset_for_SVM(root, class_name, dataset_type='train'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset_type == 'train' or dataset_type == 'test':
        CLS_LOC_image_dir = os.path.join(root, 'Data/CLS-LOC/train')
    else:
        CLS_LOC_image_dir = os.path.join(root, 'Data/CLS-LOC/val')

    CLS_LOC_samples = make_CLS_LOC_dataset_for_SVM(CLS_LOC_image_dir, class_name, dataset_type)

    pretrained_model = models.densenet121(pretrained=True)
    feature_extractor = pretrained_model.features.to(device)
    for params in feature_extractor.parameters():
        params.requires_grad = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    list_target = []
    list_features = []
    with torch.no_grad():
        for index in range(len(CLS_LOC_samples)):
            path, target = CLS_LOC_samples[index]
            try:
                image = transform(default_loader(path)).unsqueeze(0)
                input = Variable(image.to(device))
                features = feature_extractor(input)
                out = F.relu(features, inplace=True)
                out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1).cpu().numpy()
                list_features.append(np.squeeze(out))
                list_target.append(target)
            except:
                print("There is problem when loading image: ", path)
                continue
    return list_target, list_features


def example_for_SVM(image_path, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = svm_load_model(model_path)

    pretrained_model = models.densenet121(pretrained=True)
    feature_extractor = pretrained_model.features.to(device)
    for params in feature_extractor.parameters():
        params.requires_grad = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    with torch.no_grad():
        try:
            image = transform(default_loader(image_path)).unsqueeze(0)
            input = Variable(image.to(device))
            features = feature_extractor(input)
            out = F.relu(features, inplace=True)
            out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1).cpu().numpy()
            print(out)
            p_label, p_acc, p_val = svm_predict([1], out, model, '-b 1')
            print(p_val)
            image = mpimg.imread(image_path)
            plt.imshow(image)
            if int(p_label[0]) == 1:
                plt.title("Animal")
            else:
                plt.title("Not Animal")
            plt.axis('off')
            plt.show()
        except:
            print("There is problem when loading image: ", image_path)


def Precision_and_Recall_for_SVM(predicted_list, ground_truth_list):
    if len(predicted_list) != len(ground_truth_list):
        raise ValueError("len(predicted_list) must be equal to len(ground_truth_list)")
    total_correct   = 0
    true_positives  = 0
    false_positives = 0
    false_negatives = 0

    length = len(predicted_list)

    for predicted, ground_truth in zip(predicted_list, ground_truth_list):
        if predicted == ground_truth:
            total_correct += 1
            if predicted == 1:
                true_positives += 1
        elif predicted == 1:
            false_positives += 1
        else:
            false_negatives += 1
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)

    print('Accuracy  =', 100.0 * total_correct / length)
    print('Precision =', 100.0 * precision)
    print('Recall    =', 100.0 * recall)
    print('F1        =', 100.0 * F1)


if __name__ == "__main__":
    # data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'
    # map_dir = os.path.join(data_folder, 'devkit/data')
    # classes, class_to_idx = make_classes(map_dir)
    # for class_name in classes:
    #     start = time.time()
    #     y_train, x_train = ImageNet_Dataset_for_SVM(data_folder, class_name, dataset_type='train')
        # y_val, x_val = ImageNet_Dataset_for_SVM(data_folder, class_name, dataset_type='val')

        # model_path = 'models/nu_SVC_image_tagging_kernel0_'+class_name+'.model'
        # prob = svm_problem(y_train, x_train)  # , isKernel=True)
        # param = svm_parameter('-s 0 -t 0 -c 4 -b 1')  # -d 4 -h 0
        # model = svm_train(prob, param)
        # svm_save_model(model_path, model)
        # print(time.time() - start)

        # p_label, p_acc, p_val = svm_predict(y_val, x_val, model, '-b 1')
        # Precision_and_Recall_for_SVM(p_label, y_val)

    # for i in range(4):
    #     model_path = 'models/C_SVC_image_tagging_kernel' + str(i) + '_animal.model'
    #     prob = svm_problem(y_train, x_train)  # , isKernel=True)
    #     param = svm_parameter('-s 0 -t '+str(i)+' -c 4 -b 1')  # -d 4 -h 0
    #     model = svm_train(prob, param)
    #     svm_save_model(model_path, model)
    #     p_label, p_acc, p_val = svm_predict(y_val, x_val, model, '-b 1')
    #     Precision_and_Recall_for_SVM(p_label, y_val)

    # model_path = 'models/nu_SVC_image_tagging_kernel0_animal.model'
    # list_threshold = [0, 1]
    # model = svm_load_model(model_path)
    # for threshold in list_threshold:
    #     p_label, p_acc, p_val = svm_predict(y_val, x_val, model, '-b '+str(threshold))
    #     Precision_and_Recall_for_SVM(p_label, y_val)
    # print(time.time() - start)
    np.random.seed()
    # image_path = '/home/tthieuhcm/Downloads/my_image/6EA050699974C7FBE2B26A44194D415BA1AC421F.jpg'  # F
    image_path = '/home/tthieuhcm/Downloads/my_image/300px-Elephant_at_Indianapolis_Zoo.jpg'  # F
    # image_path = '/home/tthieuhcm/Downloads/my_image/1200px-Jelly_Monterey.jpg'  # F
    # image_path = '/home/tthieuhcm/Downloads/my_image/5793360-picture.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/39417693314_0f34b8c76f_m.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/blue-and-gold-macaw-5319e4fcefa42.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/Brown_Water_Snake.jpg'  # F
    # image_path = '/home/tthieuhcm/Downloads/my_image/ep271.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/european-fire-salamander-salamandra-salamandra-red-morph-with-stripes-EBNCJA.jpg'  # F
    # image_path = '/home/tthieuhcm/Downloads/my_image/gettyimages-858073064-612x612.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/i-am-a-killer-of-scorpions-0.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/images.jpeg'  # F
    # image_path = '/home/tthieuhcm/Downloads/my_image/images1.jpeg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/images2.jpeg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/North-Watersnake.jpg'  # F
    # image_path = '/home/tthieuhcm/Downloads/my_image/pexels-photo-128756.jpeg'  # F
    # image_path = '/home/tthieuhcm/Downloads/my_image/puppy-dog.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/Scorpion_Photograph_By_Shantanu_Kuveskar.jpg'  # T
    # image_path = '/home/tthieuhcm/Downloads/my_image/uploads%2Fcard%2Fimage%2F887009%2F49d251da-d190-49d8-80ad-5f12221574b4.jpg%2F950x534__filters%3Aquality%2890%29.jpg'  # T
    model_path = 'models/nu_SVC_image_tagging_kernel0_animal.model'

    example_for_SVM(image_path, model_path)
    example_for_SVM(image_path, model_path)
