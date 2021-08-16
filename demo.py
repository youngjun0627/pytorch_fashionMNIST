import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

config = {
    'batch_size' : 64,
    'lr':0.001,
    'n_classes' : 10,
    'epochs':5,
    'mean':0.5, 
    'std':0.5,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
}

'''
# 간단한 transform 정의
'''
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((config['mean']), (config['std']))])


'''
# dataset & dataLoader
'''
trainset = datasets.FashionMNIST('.', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)


testset = datasets.FashionMNIST('.', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False)

'''
MODEL 정의
tensorflow 예제에서는 단순 DNN이였다면 
퍼포먼스 향상을 위해 CNN(LeNet)으로 change
'''
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_classes = 1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size = 1) # 28 x 28 이므로 5->1로 변경
        self.conv2 = nn.Conv2d(6,16,kernel_size = 5)
        self.conv3 = nn.Conv2d(16,120,kernel_size = 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, n_classes)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride =2)
        
    def forward(self, x): # tanh -> relu
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1,120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DNN(nn.Module):
    def __init__(self, n_classes = 1):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

'''
train, valdation function
'''
def train(model, dataloader, criterion, optimizer, device):
    running_loss = 0
    for images, labels in tqdm(dataloader, position=0, leave=True):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    running_loss /= len(dataloader)
    return running_loss

def validation(model, dataloader, criterion, device):
    running_loss = 0
    preds = []
    targets = []
    for images, labels in tqdm(dataloader, position=0, leave=True):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        preds += torch.argmax(outputs,1).tolist()
        targets += labels.tolist()
        running_loss += loss.item()
    preds = np.array(preds)
    targets = np.array(targets)
    score = (preds == targets).sum() / len(preds)
    running_loss /= len(dataloader)
    return running_loss, score

'''
CNN 실험해보기
'''
device = config['device']
model = CNN(config['n_classes']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = config['lr'])
epochs = config['epochs']


for epoch in range(1, epochs):
    t_loss = train(model, trainloader, criterion, optimizer, device)
    v_loss, score = validation(model, testloader, criterion, device)
    print('train_loss : {:.4f} \t test_loss : {:.4f} \t score : {:.3f}'.format(t_loss, v_loss, score))
print("Accuracy Score : {}".format(score))
