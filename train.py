# Author: Trung Hieu Tran
# Date  : 19/1/2019

from __future__ import print_function, division

import copy
import os
import time

from numpy import array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.autograd import Variable

from dataloader import MIR_FLICKR_Dataset


def show_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch, labels_batch = sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.title('Batch from dataloader')


def show_image(img):
    """Show image for a batch of samples."""
    plt.imshow(img.numpy().transpose((1, 2, 0)))


def train_model(model, train, val, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_loss = 100.0
    train_error_list = list()
    val_error_list = list()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for sample_batched in tqdm(dataloader):
                inputs, labels = sample_batched['image'].to(device), sample_batched['labels'].type(
                    torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * 100

            epoch_loss = running_loss / len(dataloader)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                train_error_list.append(epoch_loss)
            else:
                val_error_list.append(epoch_loss)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), './models/densenet121+conv0+norm0+dl16conv2+dlnorm2+FC+Sigmoid+BCE-{}.ckpt'.format(epoch + 1))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))
    print('Train_error_list:', train_error_list)
    print('Val_error_list:', val_error_list)

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model


def visualize_model(device, model, tag_list, dataloader, num_images=4):
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched['image'].to(device), sample_batched['labels']

            outputs = model(inputs)
            threshold = 0.3
            t = Variable(torch.Tensor([threshold])).to(device)
            preds = (outputs > t).float() * 1
            preds = preds.type(torch.IntTensor).cpu().numpy()
            images_so_far = 0

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(4 // 2, 2, images_so_far)
                ax.axis('off')
                pred_list = list()
                ground_truth_list = list()
                for num_item, item in enumerate(preds[j]):
                    if item != 0:
                        pred_list.append(tag_list[item * num_item])
                for num_item, item in enumerate(labels[j]):
                    if item != 0:
                        ground_truth_list.append(tag_list[item * num_item])
                ax.set_title('predicted: {}\nground truth: {}'.format(pred_list, ground_truth_list))
                show_image(sample_batched['image'][j])

                # imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    plt.axis('off')
                    plt.show()
                    model.train(mode=was_training)
                    # return
        model.train(mode=was_training)


def test_image(device, model, tag_list, image_path):
    was_training = model.training
    model.eval()
    fig = plt.figure()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)

    with torch.no_grad():
        outputs = model(image.to(device).unsqueeze(0))
        threshold = 0.3
        t = Variable(torch.Tensor([threshold])).to(device)
        preds = (outputs > t).float() * 1
        preds = preds.type(torch.IntTensor).cpu().numpy()
        ax = plt.subplot(4 // 2, 2, 1)
        ax.axis('off')
        pred_list = list()
        for num_item, item in enumerate(preds.squeeze(0)):
                if item != 0:
                    pred_list.append(tag_list[item * num_item])
        ax.set_title('predicted: {}'.format(pred_list))
        show_image(image)
        plt.axis('off')
        plt.show()
        model.train(mode=was_training)


if __name__ == "__main__":
    with open('./data_processing/taxonomy.txt') as all_tags:
        tag_list = [item.rstrip() for item in all_tags.readlines()]
        print("Classes: ", tag_list)

    number_of_images = 25000
    number_of_tags = len(tag_list)
    img_labels = array([[0 for i in range(number_of_tags)] for j in range(number_of_images)])

    with open('./data_processing/mirflickr_labels.txt') as info:
        lines = info.readlines()
        img_name = array([img_info.split()[0] for img_info in lines])
        for i, img_label in enumerate(lines):
            img_tags = img_label.split()[1:]
            tag_index = [tag_list.index(tag) for tag in img_tags]
            for index in tag_index:
                img_labels[i][index] = 1

    train_split = .7
    validation_split = .15

    train = int(np.floor(train_split * number_of_images))
    validation = int(np.floor(validation_split * number_of_images))

    mir_flickr_val = MIR_FLICKR_Dataset(img_name[train:train + validation], img_labels[:validation],
                                        root_dir='/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/mirflickr',
                                        transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                      # transform=transforms.Compose([transforms.Resize((256, 256)),
                                                                      transforms.ToTensor()]))

    mir_flickr_train = MIR_FLICKR_Dataset(img_name[:train], img_labels[validation:validation+train],
                                          root_dir='/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/mirflickr',
                                          transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                        # transform=transforms.Compose([transforms.Resize((256, 256)),

                                                                        transforms.ToTensor()]))

    train_loader = DataLoader(mir_flickr_train, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(mir_flickr_val, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.densenet161(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
    model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)

    # model.load_state_dict(torch.load("models/densenet121+conv0+norm0+dl16conv2+dlnorm2+FC+Sigmoid+BCE-44.ckpt"))  # 0.7
    for param in model[0].parameters():
        param.requires_grad = False
    # for param in model[0].features.conv0.parameters():
    #     param.requires_grad = True
    # for param in model[0].features.norm0.parameters():
    #     param.requires_grad = True
    for param in model[0].features.denseblock4.denselayer16.conv2.parameters():
        param.requires_grad = True
    for param in model[0].features.denseblock4.denselayer16.norm2.parameters():
        param.requires_grad = True
    for param in model[0].classifier.parameters():
        param.requires_grad = True

    criterion = nn.BCELoss()

    optimizer_ft = optim.Adam(model.parameters(), lr=0.0003)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    train_model(model, train_loader, val_loader, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)
