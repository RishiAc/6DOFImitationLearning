import pandas as pd  # Import pandas for data manipulation
from PIL import Image  # Import PIL for image processing
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting (though not used in the code)
from torchvision import transforms  # Import transforms for image preprocessing
from torch.utils.data import Dataset, DataLoader, random_split  # Import Dataset, DataLoader, and random_split for dataset handling
import torch as th  # Import PyTorch library as th
import torch.nn as nn  # Import neural network module from PyTorch
import torch.optim as optim  # Import optimization algorithms from PyTorch
from cv_bridge import CvBridge  # Import CvBridge (though not used in the code)
import torchvision.transforms as transforms  # Import transforms (duplicate import)

# Initialize CvBridge (not used in the code but typically for ROS image conversion)
bridge = CvBridge()

# Define a custom Dataset class to handle image and label data
class ImageAngleDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe  # Store the dataframe
        self.transform = transform  # Store any image transformations

    def __len__(self):
        return len(self.dataframe)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]  # Get the image path from the dataframe
        actionNum = int(self.dataframe.iloc[idx, 1])  # Get the action number (label) and ensure it's an integer
        joints = self.dataframe.iloc[idx, 2:].astype(float)  # Get the joint data and ensure it's in float format

        image = Image.open(img_path).convert('RGB')  # Open and convert the image to RGB
        if self.transform:
            image = self.transform(image)  # Apply transformations if any
        
        # Return the image, action number as tensor, and joint data as tensor
        return image, th.tensor(actionNum, dtype=th.long), th.tensor(joints, dtype=th.float)

# Define the CNN model class
class CNNModel(nn.Module):
    def __init__(self, num_classes, input_shape=(3, 128, 128), joints=(5,1)):
        super().__init__()
        self.input_shape = input_shape  # Store input shape
        self.input_channels = input_shape[0]  # Number of input channels (RGB)
        self.num_classes = num_classes  # Number of output classes
        self.joints = joints[0]  # Number of joint features

        # Define the CNN layers
        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 128, kernel_size=4, stride=2, padding=1),  # First convolution layer
            nn.ELU(),  # ELU activation
            nn.BatchNorm2d(128),  # Batch normalization
            nn.MaxPool2d((2, 2)),  # Max pooling
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  # Second convolution layer
            nn.ELU(),  # ELU activation
            nn.BatchNorm2d(64),  # Batch normalization
            nn.MaxPool2d((2, 2)),  # Max pooling
            nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=1),  # Third convolution layer
            nn.ELU(),  # ELU activation
            nn.BatchNorm2d(16),  # Batch normalization
        )

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(261, 261),  # Fully connected layer
            nn.ELU(),  # ELU activation
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(261, 261),  # Another fully connected layer
            nn.ELU(),  # ELU activation
            nn.Linear(261, self.num_classes),  # Output layer
        )

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        x = self.features(x)  # Pass the input through CNN layers
        x = th.flatten(x, 1)  # Flatten CNN output
        y = y.view(y.size(0), -1)  # Flatten joint tensor if needed
        x = th.cat((x, y), dim=1)  # Concatenate CNN output with joint data
        x = self.classifier(x)  # Pass through the classifier
        return x

# Load the data from a CSV file
dataset = pd.read_csv("/home/gov_laptop/ArmDemo/FullArmData.csv")  # Load CSV into dataframe
dataset['action_num'] = dataset['action_num'].shift(-1)  # Shift action numbers to align with images
dataset = dataset.drop(len(dataset)-1)  # Drop the last row (due to shift)

print(dataset)  # Print dataset for verification

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std
])

# Create dataset object with transformations
dataset = ImageAngleDataset(dataframe=dataset, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining 20% for testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Split the dataset

# Create DataLoader objects for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Training DataLoader with shuffling
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Testing DataLoader without shuffling

# Initialize the model, loss function, and optimizer
num_classes = 11  # Number of output classes
model = CNNModel(num_classes=num_classes)  # Instantiate the model
criterion = nn.CrossEntropyLoss()  # Define the loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Define the optimizer with learning rate

# Training loop
num_epochs = 20  # Number of training epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    trainingaccuracy=0
    for images, labels,joints in train_loader:
        optimizer.zero_grad()
        outputs = model(images,joints)
        #print(outputs,labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        # Compute accuracy for multi-classq classification
        preds = th.argmax(outputs, dim=1)  # Get the index of the max logit for each class
        correct += (preds == labels).sum().item()
        total += labels.size(0)  # Number of samples in the batch
    trainingaccuracy = 100 * correct / total
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    print(f'Accuracy: {trainingaccuracy:.2f}%')
    model.eval()
    with th.no_grad():
        total_loss = 0.0
        correct = 0
        total = 0
        #Takes the images and labels from the testing dataset and then gets the predictions and compares them to 
        for images, labels,joints in test_loader:
            outputs = model(images,joints)
            #print(outputs, labels)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            # Compute accuracy for multi-classq classification
            preds = th.argmax(outputs, dim=1)  # Get the index of the max logit for each class
            correct += (preds == labels).sum().item()
            total += labels.size(0)  # Number of samples in the batch
        
        average_loss = total_loss / len(test_loader.dataset)
        accuracy = 100 * correct / total
        print(f'Test Loss: {average_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
    if trainingaccuracy-accuracy>.5:
        break
# Testing loop
model.eval()  # Set model to evaluation mode
with th.no_grad():  # Disable gradient computation
    total_loss = 0.0  # Initialize total loss
    correct = 0  # Initialize correct predictions count
    total = 0  # Initialize total samples count
    for images, labels, joints in test_loader:
        outputs = model(images, joints)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        total_loss += loss.item() * images.size(0)  # Accumulate loss

        preds = th.argmax(outputs, dim=1)  # Get predicted classes
        correct += (preds == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Accumulate total samples
    
    average_loss = total_loss / len(test_loader.dataset)  # Compute average test loss
    accuracy = 100 * correct / total  # Compute accuracy
    print(f'Test Loss: {average_loss:.4f}')  # Print test loss
    print(f'Accuracy: {accuracy:.2f}%')  # Print accuracy

# Function to save the trained model
def save_model(model, file_path):
    th.save(model.state_dict(), file_path)  # Save model parameters
    print(f"Model saved to {file_path}")  # Print confirmation

# Example usage of saving the model
save_model(model, '/home/gov_laptop/ArmDemo/FullDemoModel.pth')  # Save model to specified path
