import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

dataset_path = "dataset"
# check if dataset exists before starting
if not os.path.exists(dataset_path):
    print("dataset folder not found")
    exit()

# setup data augmentation for training to help model generalize
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# testing data shouldn't be augmented, just resized and normalized
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# load the image folders
train_data = datasets.ImageFolder(dataset_path + '/Training', transform=train_transform)
test_data = datasets.ImageFolder(dataset_path + '/Testing', transform=test_transform)

# create data loaders for batching
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# build a simple CNN model for image classification
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution layers to extract features
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # fully connected layers for classification
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 4) # 4 classes: glioma, meningioma, no_tumor, pituitary
        self.relu = nn.ReLU()

    def forward(self, x):
        # pass through first conv block
        x = self.pool(self.relu(self.conv1(x)))
        # pass through second conv block
        x = self.pool(self.relu(self.conv2(x)))
        
        # flatten the tensor before passing to linear layers
        x = x.view(-1, 32 * 32 * 32)
        
        # pass through linear blocks
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# standard loss function and optimizer for multiclass classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 10
# start training loop
for epoch in range(epochs):
    model.train() # set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero out the gradients
        optimizer.zero_grad()
        
        # forward pass -> compute loss -> backward pass -> update weights
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # run evaluation on the test set
    model.eval() # set model to eval mode (disables dropout/batchnorm if any)
    test_correct = 0
    test_total = 0
    with torch.no_grad(): # no need to track gradients during testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # print epoch summary
    print(f"epoch {epoch+1} - train acc: {correct/total:.2f}, test acc: {test_correct/test_total:.2f}")

# save the final model weights
torch.save(model.state_dict(), "model.pth")
print("saved")
