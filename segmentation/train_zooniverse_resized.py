import wandb
wandb.init(project='clouds-segmentation', entity='raysamram')
config = wandb.config
from PIL import Image
import datetime
import re

from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from architecture import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from collections import OrderedDict
import pandas as pd
import os
import gc
import argparse, sys


parser=argparse.ArgumentParser()

parser.add_argument('--lr', help='Learning Rate, default = 0.0001', type=int, default=0.0001)
parser.add_argument('--epochs', help='Number of epochs, default = 100', type=int, default=100)
parser.add_argument('--load_trained', help='Load existing model', type=bool, default=False)
parser.add_argument('--test_percentage', help='Percentage of training dataset, default=0.3', type=int, default=0.3)
parser.add_argument('--path_image_folder', help='Path to the dataset (ImageFolder scheme).', type=str, default="../../DATASETS/CLASSIF_RESIZED/")

args=parser.parse_args()




EPOCHS = args.epochs
LEARNING_RATE = args.lr
PERCENTAGE_TEST = args.test_percentage
IMAGE_FOLDER = args.path_image_folder
config.learning_rate = LEARNING_RATE
config.dataset = "ZOONIVERSE RESIZED"
config.epochs = EPOCHS
config.percentage_test = PERCENTAGE_TEST
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
transformer = transforms.Compose([
    transforms.Resize([768, 512]),
    # you can add other transformations in this list
    transforms.ToTensor(),
    normalize,
])
dataset = ImageFolder(root=IMAGE_FOLDER, transform=transformer)
n = len(dataset)
n_test = int(0.1 * n)  # take ~10% for test
train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-n_test, n_test], generator=torch.Generator().manual_seed(42))


config.number_train_images = len(train_set)
config.number_test_images = len(test_set)


train_dataloader  = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1) 
test_dataloader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)


pretrained_model = models.resnet18(pretrained=True)
for param in pretrained_model.parameters():
    param.requires_grad = False

pretrained_model.fc = nn.Sequential(
    nn.Linear(512, 100),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(100, 20),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(20, 4),
    nn.LogSoftmax(dim=1)
)
pretrained_model.to(device)
net = pretrained_model
def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

PATH = "zooniverse-resized-image_net_model.checkpt"
if args.load_trained and os.path.isfile(PATH):
    print("LOADED MODEL") 
    net = torch.load(PATH)

print("NET : "+str(net))
net.train()




#del variables
gc.collect()
torch.cuda.empty_cache()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_func = nn.NLLLoss()  
running_loss = 0
print_every = 5
# Training and Testing
for epoch in range(EPOCHS):
    print("Epoch : "+str(epoch))
    for step, (x, y) in enumerate(train_dataloader):
        
        b_x = Variable(x).to(device)   # batch x (image)
        b_y = Variable(y).to(device)   # batch y (target)
        
        output = net(b_x)#.argmax(dim=1)
        loss = loss_func(output, b_y)   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()
        running_loss += loss.item()
        
        # Test -> this is where I have no clue
        if step % 10 == 0:
            test_loss = 0
            accuracy = 0
            net.eval()
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = net(inputs)
                    batch_loss = loss_func(logps, labels)
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    wandb.log({
                        "epoch":epoch,
                        "train loss":running_loss/print_every,
                        "test loss":test_loss/len(test_dataloader),
                        "test accuracy":accuracy/len(test_dataloader),
                        })
            print(f"Epoch {epoch}.. "
                  f"Step {step}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_dataloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_dataloader):.3f}")
            running_loss = 0
            net.train()
            torch.save(net, "zooniverse-resized-image_net_model.checkpt")
            wandb.save("zooniverse-all-resized_net_model.checkpt")


torch.save(net.state_dict(), "zooniverse-resized-image_state_dict")
torch.save(net, "zooniverse-resized-image_net_model")
