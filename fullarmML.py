import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch as th
import torch.nn as nn
import torch.optim as optim
from cv_bridge import CvBridge
import torchvision.transforms as transforms
bridge = CvBridge()

# Define a custom Dataset that formats that data into a tensor that can be input into the dataset
class ImageAngleDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        #There is a picture saved locally that each path from the csv relates to so it can pull the image for data.
        img_path = self.dataframe.iloc[idx, 0]
        actionNum = int(self.dataframe.iloc[idx, 1])  # Ensure labels are integers
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, th.tensor(actionNum, dtype=th.long)  # Ensure labels are LongTensor


# Define the CNN model
class CNNModel(nn.Module):
    #Nicks Model
    def __init__(self, num_classes, input_shape = (3, 128, 128)):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        self.num_classes = num_classes

        # features
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(16),
        )  # output size is 3x5x7 = 105

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.features(x)
        x = th.flatten(x, 1)
        x = self.classifier(x)
        return x
# Load the data from the csv saved locally
dataset = pd.read_csv("/home/gov_laptop/ArmDemo/FullArmOnlyGrab.csv")  # Ensure CSV contains image paths and multi-labels
dataset['action_num']=dataset['action_num'].shift(-1)
dataset=dataset.drop(len(dataset)-1)
print(dataset)
#This transform gets just sets the variable for the dataset to make sure its in tensor format and resized to 128 by 128
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageAngleDataset(dataframe=dataset, transform=transform)
#Next lines splits the data into training and test with test having 20 percent of the pictures and training with 20 percent
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#Sets the memory and batch size for the train and test dataset.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
num_classes = 11 # Assuming first column is image path and rest are labels the labels are the state names listed in the loading documents
#Calls the model listed at the top of this script
model = CNNModel(num_classes=num_classes)
#Assigns this model as a crossentropy loss that will use the softmax activation layer.
criterion = nn.CrossEntropyLoss()
#Uses the adam optimizer with a learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
#For each epoch it starts training and has a running loss that will print at the end of each epoch
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        #print(outputs,labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Testing loop
model.eval()
with th.no_grad():
    total_loss = 0.0
    correct = 0
    total = 0
    #Takes the images and labels from the testing dataset and then gets the predictions and compares them to 
    for images, labels in test_loader:
        outputs = model(images)
        print(outputs, labels)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        
        # Compute accuracy for multi-class classification
        preds = th.argmax(outputs, dim=1)  # Get the index of the max logit for each class
        correct += (preds == labels).sum().item()
        total += labels.size(0)  # Number of samples in the batch
    
    average_loss = total_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f'Test Loss: {average_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}%')


# Prediction function
def predict_labels(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    model.eval()
    with th.no_grad():
        output = model(image_tensor)
    
    predicted_label = th.argmax(output, dim=1).item()  # Get the index of the max logit
    return predicted_label


# Save the model
def save_model(model, file_path):
    th.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Example usage
save_model(model, '/home/gov_laptop/ArmDemo/FullArmGrabModel.pth')
