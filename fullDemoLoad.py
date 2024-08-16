#!/usr/bin/env python  # Shebang line indicating the script should be run using Python

from ikpy.chain import Chain  # Import Chain class for inverse kinematics
from ikpy.link import OriginLink, URDFLink  # Import link classes for robot kinematics
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot  # Import matplotlib for plotting (though not used in the code)
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit (though not used in the code)
import rospy  # Import rospy for ROS (Robot Operating System) communication
from std_msgs.msg import Float64MultiArray, Float64  # Import message types for ROS
import pygame  # Import pygame (though not used in the code)
from sensor_msgs.msg import CompressedImage  # Import ROS message type for compressed images
import pandas as pd  # Import pandas for data manipulation (though not used in the code)
from cv_bridge import CvBridge  # Import CvBridge for converting ROS images to OpenCV format
import cv2  # Import OpenCV for image processing
import math  # Import math for mathematical operations (though not used in the code)
import os  # Import os for operating system interactions (though not used in the code)
import torch  # Import PyTorch for deep learning
import torchvision.transforms as transforms  # Import image transformations for preprocessing
from PIL import Image  # Import PIL for image processing
from std_msgs.msg import Float64  # Duplicate import for Float64 message type
import torch.nn as nn  # Import neural network module from PyTorch
import time  # Import time for time-related functions
from sensor_msgs.msg import JointState  # Import ROS message type for joint states
import message_filters  # Import message_filters for synchronizing ROS messages
import torch as th  # Import PyTorch with alias th
bridge = CvBridge()  # Initialize CvBridge for image conversion

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
            nn.ELU(),  # ELU activation function
            nn.BatchNorm2d(128),  # Batch normalization
            nn.MaxPool2d((2, 2)),  # Max pooling
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1),  # Second convolution layer
            nn.ELU(),  # ELU activation function
            nn.BatchNorm2d(64),  # Batch normalization
            nn.MaxPool2d((2, 2)),  # Max pooling
            nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=1),  # Third convolution layer
            nn.ELU(),  # ELU activation function
            nn.BatchNorm2d(16),  # Batch normalization
        )

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(261, 261),  # Fully connected layer
            nn.ELU(),  # ELU activation function
            nn.Dropout(0.2),  # Dropout for regularization
            nn.Linear(261, 261),  # Another fully connected layer
            nn.ELU(),  # ELU activation function
            nn.Linear(261, self.num_classes),  # Output layer
        )

    def _get_flatten_size(self):
        # Determine the size of the flattened tensor after the CNN layers
        with th.no_grad():
            dummy_input = th.zeros(1, *self.input_shape)  # Create a dummy input tensor
            dummy_output = self.features(dummy_input)  # Pass it through the CNN layers
            return th.flatten(dummy_output, 1).shape[1]  # Flatten and get the size

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        x = self.features(x)  # Pass the input through CNN layers
        x = th.flatten(x, 1)  # Flatten the output from CNN
        if not isinstance(y, th.Tensor):  # Check if y is a tensor
            raise TypeError(f"Expected tensor, got {type(y)}")
        
        y = y.view(y.size(0), -1)  # Flatten joint tensor if needed
        
        # Print shapes for debugging
        print(f"Features tensor shape (x): {x.shape}")
        print(f"Joint tensor shape (y): {y.shape}")
        
        # Concatenate CNN output with joint data
        x = th.cat((x, y), dim=1)
        x = self.classifier(x)  # Pass through the classifier
        return x

# Function to publish joint angles to ROS
def publish_joint_angles(joint_angles):
    msg = Float64MultiArray()  # Create a Float64MultiArray message
    msg.data = joint_angles  # Set the joint angles data
    joint1pub.publish(msg.data[1])  # Publish joint1 angle
    joint2pub.publish(msg.data[3])  # Publish joint2 angle
    joint3pub.publish(msg.data[4])  # Publish joint3 angle
    # joint4pub.publish(msg.data[5])  # Publish joint4 angle (commented out)
    rate.sleep()  # Sleep to maintain the desired frequency

angle = 0  # Initialize gripper angle

# Function to publish actions based on input
def publish_action(action):
    global angle
    if action == 5:  # Move Up
        target_position[0] += .005
        
    elif action == 3:  # Move Left
        target_position[1] -= .005
        
    elif action == 6:  # Move Down
        target_position[0] -= .005
        
    elif action == 4:  # Move Right
        target_position[1] += .005
        
    elif action == 2:  # Move Backward
        target_position[2] -= .005
        
    elif action == 1:  # Move Forward
        target_position[2] += .005
        
    elif action == 7:  # Gripper Up
        angle += .05
        joint4pub.publish(angle)
        
    elif action == 8:  # Gripper Down
        angle -= .05
        joint4pub.publish(angle)
        
    elif action == 10:  # Gripper Closed
        gripper1pub.publish(-.8)
        
    elif action == 9:  # Gripper Open
        gripper1pub.publish(1.2)
    
    joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)
    if action is not None:
        try:
            publish_joint_angles(joint_angles)  # Publish joint angles to move the robot
        except rospy.ROSInterruptException:
            pass

# Load the trained model
def load_model(file_path, num_classes):
    model = CNNModel(num_classes=num_classes)  # Initialize model
    model.load_state_dict(torch.load(file_path))  # Load trained model parameters
    model.eval()  # Set model to evaluation mode
    return model

# Preprocess image and joint data for model input
def preprocess_image(rawimage, joints):
    image = Image.fromarray(cv2.cvtColor(rawimage, cv2.COLOR_BGR2RGB))  # Convert OpenCV image to PIL image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize image to 128x128
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
    ])
    
    joints = np.array(joints, dtype=np.float32)  # Ensure joint data is float32
    joint_tensor = th.tensor(joints)  # Convert joint data to tensor
    joint_tensor = joint_tensor.reshape(1, 5)  # Reshape tensor
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension to image tensor
    return image_tensor, joint_tensor

# Predict labels using the trained model
def predict(model, image_tensor, joint_tensor):
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_tensor, joint_tensor)  # Get model output
        predicted_labels = torch.softmax(output, dim=1).squeeze().tolist()  # Convert logits to probabilities
    return predicted_labels

# Callback function for processing messages
def callback(cammsg, jointmsg):
    joint_states = jointmsg.position[0:5]  # Get the first 5 joint positions
    cv2image = bridge.compressed_imgmsg_to_cv2(cammsg, "bgr8")  # Convert ROS compressed image to OpenCV format
    imgtensor, jointTensor = preprocess_image(cv2image, joint_states)  # Preprocess image and joint data
    labels = predict(model, imgtensor, jointTensor)  # Get predictions from model
    highestProb = 0
    index = 0
    for i in range(len(labels)):
        if labels[i] > highestProb:  # Find the highest probability label
            highestProb = labels[i]
            index = i
    state_name = state_names[index] if index < len(state_names) else "Unknown"
    print(f"Highest Probability: {highestProb}, Label Index: {index}, State: {state_name}")
    if highestProb > .2:
        publish_action(index)  # Publish action based on the predicted label

# Main function
if __name__ == '__main__':
    rospy.init_node("MLTest")  # Initialize the ROS node
    joint1pub = rospy.Publisher('/joint1_controller/command', Float64, queue_size=10)  # Publisher for joint1
    joint2pub = rospy.Publisher('/joint2_controller/command', Float64, queue_size=10)  # Publisher for joint2
    joint3pub = rospy.Publisher('/joint3_controller/command', Float64, queue_size=10)  # Publisher for joint3
    joint4pub = rospy.Publisher('/joint4_controller/command', Float64, queue_size=10)  # Publisher for joint4
    gripper1pub = rospy.Publisher('/r_joint_controller/command', Float64, queue_size=10)  # Publisher for gripper
    rate = rospy.Rate(10)  # Rate for publishing messages
    cv2image = None
    angle = 1
    robot = Chain.from_urdf_file('/home/gov_laptop/ArmDemo/jetautoArm.urdf')  # Load robot kinematics from URDF file
    print(robot)
    
    # Define the target position and orientation
    target_position = [.2, 0, .3]
    target_orientation = [0, 0, 0]
    joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)
    try:
        publish_joint_angles(joint_angles)  # Publish initial joint angles
    except rospy.ROSInterruptException:
        pass

    joint4pub.publish(0)  # Set gripper to initial position
    time.sleep(1)  # Wait for 1 second
    target_position = [.1, 0, .3]  # Update target position
    joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)
    try:
        publish_joint_angles(joint_angles)  # Publish updated joint angles
    except rospy.ROSInterruptException:
        pass

    joint4pub.publish(1)  # Move gripper to position 1
    gripper1pub.publish(1.2)  # Set gripper to open position
    
    state_names = [
        "Nothing", "Up", "Down", "Left", "Right", "Forward",
        "Backward", "GripperDown", "GripperUp", "GripperOpen", "GripperClosed",
    ]
    num_classes = 11  # Number of classes in the model
    device=th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = load_model('/home/gov_laptop/ArmDemo/FullDemoModel.pth', num_classes=num_classes).to(device)  # Load the trained model
    camerasub = message_filters.Subscriber("/usb_cam/image_color/compressed", CompressedImage, queue_size=1, buff_size=52428800)  # Subscriber for compressed images
    jointsub = message_filters.Subscriber("/joint_states", JointState, queue_size=1, buff_size=52428800)  # Subscriber for joint states
    ts = message_filters.ApproximateTimeSynchronizer([camerasub, jointsub], queue_size=1, slop=.5, allow_headerless=False)  # Synchronize messages
    ts.registerCallback(callback)  # Register callback for processing messages
    try:
        rospy.spin()  # Keep the node running
    except KeyboardInterrupt:
        print("Shutting down")  # Handle keyboard interrupt gracefully
