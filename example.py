import os

import torch
from torch.autograd import Variable
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader

import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nltk.corpus import wordnet

from utils import make_classes

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_folder = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/'

    number_of_tags = 1788

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms = transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             normalize,
                         ])
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
    model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)
    model.load_state_dict(torch.load("models/densenet121-BCE-30.ckpt"))
    was_training = model.training
    model.eval()
    map_dir = os.path.join(data_folder, 'devkit/data')
    classes, class_to_idx = make_classes(map_dir)
    inx_to_class = {v: k for k, v in class_to_idx.items()}

    with torch.no_grad():
        fig = plt.figure()

        threshold = 0.5
        path = '/home/tthieuhcm/Downloads/my_image/300px-Elephant_at_Indianapolis_Zoo.jpg'
        image = default_loader(path)
        image = transforms(image)

        input = image.cuda().unsqueeze(0)
        outputs = model(input)
        t = Variable(torch.Tensor([threshold])).to(device)
        preds = (outputs > t).float() * 1
        preds = preds.type(torch.IntTensor).cpu().numpy()
        pred_list = list()

        for num_item, item in enumerate(preds[0]):
            if item != 0:
                pred_list.append(inx_to_class[num_item])

        for tag in pred_list:
            tag_name = wordnet.synset_from_pos_and_offset('n', int(tag[1:]))
            print(tag_name.lemmas()[0].name())

        image = mpimg.imread(path)
        plt.imshow(image)

        plt.axis('off')
        plt.show()
        model.train(mode=was_training)
