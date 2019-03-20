import time

import torch
from torch.optim import lr_scheduler
from torchvision.datasets.folder import default_loader

from torchvision.transforms import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from dataloader import ImageNet_Dataset
from FocalLoss import FocalLoss
from torch.autograd import Variable


def train_model(model, train, val, criterion, optimizer, scheduler, num_epochs=60):

    best_loss = float("inf")
    train_error_list = list()
    val_error_list = list()

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input, target in dataloader:
                input = Variable(input.cuda())
                target = Variable(target.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(input)
                    loss = criterion(output, target)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / (len(dataloader)*batch_size)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                train_error_list.append(epoch_loss)
            else:
                val_error_list.append(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), './models/densenet121-FocalLoss-{}.ckpt'.format(epoch + 1))
        time_elapsed = time.time() - since
        print('Training 1 epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_loss))
    print('Train_error_list:', train_error_list)
    print('Val_error_list:', val_error_list)


if __name__ == "__main__":
    batch_size = 64
    num_worker = 4
    root_dir = './ILSVRC'
    root_dir = '/media/tthieuhcm/6EAEFFD5AEFF93B5/Users/Administrator/Downloads/ILSVRC'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageNet_Dataset(
        root_dir,
        loader=default_loader,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        dataset_type='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True)

    number_of_tags = len(train_dataset.classes)

    val_loader = torch.utils.data.DataLoader(
        ImageNet_Dataset(root_dir,
                         loader=default_loader,
                         transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                         ]),
                         dataset_type='val'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
    model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)

    for param in model[0].parameters():
        param.requires_grad = False
    for param in model[0].features.denseblock3.parameters():
        param.requires_grad = True
    for param in model[0].features.transition3.parameters():
        param.requires_grad = True
    for param in model[0].features.denseblock4.parameters():
        param.requires_grad = True
    for param in model[0].classifier.parameters():
        param.requires_grad = True

    criterion = FocalLoss(gamma=2, alpha=1, reduction='sum')
    # criterion = nn.BCELoss(reduction='sum')

    optimizer_ft = optim.SGD(model.parameters(), lr=0.02)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    train_model(model, train_loader, val_loader, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=60)

    torch.save(model.state_dict(), './models/densenet121-FocalLoss-Final.ckpt')

