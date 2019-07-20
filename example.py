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
    # number_of_tags = 1161

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
    model.load_state_dict(torch.load("models/densenet121-FocalLoss-33-gamma2-alpha1.ckpt"))
    model.load_state_dict(torch.load("models/densenet121-FocalLoss-42-gamma5-alpha025.ckpt"))
    model.load_state_dict(torch.load("models/densenet121-DET-CLS-FocalLoss-36-gamma2-alpha1.ckpt"))
    model.load_state_dict(torch.load("models/densenet121-weightedBCE-47.ckpt"))
    # model.load_state_dict(torch.load("models/densenet121-DenoisedWeightedBCE-51.ckpt"))

    was_training = model.training
    model.eval()
    map_dir = os.path.join(data_folder, 'devkit/data')
    classes, class_to_idx = make_classes(map_dir)
    inx_to_class = {v: k for k, v in class_to_idx.items()}

    with torch.no_grad():
        fig = plt.figure()

        threshold = 0.05
        image_path = '/home/tthieuhcm/Downloads/my_image/_103462292_8ea7a778-b2ea-45ec-836e-9f00fea37278.jpg'
        image_path = '/home/tthieuhcm/Downloads/my_image/1200px-Jelly_Monterey.jpg'
        image_path = '/home/tthieuhcm/Downloads/my_image/22730317_1307316912713596_5177287517156284158_n.jpg'
        image_path = '/home/tthieuhcm/Downloads/my_image/37692573_1117350788412869_3391902473001107456_n.jpg'
        # image_path = '/home/tthieuhcm/Downloads/my_image/pexels-photo-937481.jpeg'
        # image_path = '/home/tthieuhcm/Downloads/my_image/prism-kites-stowaway-diamond-p1-flying-sky.jpg'

        # image_path = '/home/tthieuhcm/Downloads/my_image/300px-Elephant_at_Indianapolis_Zoo.jpg'
        # image_path = '/home/tthieuhcm/Downloads/my_image/220px-Amanita_muscaria_(fly_agaric).JPG'
        # image_path = '/home/tthieuhcm/Downloads/my_image/47495857_566104483857232_8201193975302848512_n.jpg'
        # image_path = '/home/tthieuhcm/Downloads/my_image/a.jpg'
        # image_path = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC/Data/DET/train/ILSVRC2013_train/n03599486/n03599486_11467.JPEG'
        image = default_loader(image_path)
        image = transforms(image)

        input = image.cuda().unsqueeze(0)
        outputs = model(input)
        t = Variable(torch.Tensor([threshold])).to(device)
        preds = (outputs > t).float() * 1
        preds = preds.type(torch.IntTensor).cpu().numpy()
        pred_list = list()

        result = dict()
        for num_item, item in enumerate(preds[0]):
            if item != 0:
                tag_ID = inx_to_class[num_item]
                tag_name = wordnet.synset_from_pos_and_offset('n', int(tag_ID[1:]))
                result[tag_name.lemmas()[0].name()] = outputs[0][num_item].item()
                # print('{}: {}'.format(tag_name.lemmas()[0].name(), "%.4f" % outputs[0][num_item].item()))

        for tag in sorted(result, key=result.get, reverse=True):
            print('{} & {}\\\\'.format(tag, "%.4f" % result[tag]))


        # for tag in pred_list:
        #     tag_name = wordnet.synset_from_pos_and_offset('n', int(tag[1:]))
        #     print(tag_name.lemmas()[0].name())

        # image = mpimg.imread(image_path)
        # plt.imshow(image)
        #
        # plt.axis('off')
        # plt.show()
        # model.train(mode=was_training)
