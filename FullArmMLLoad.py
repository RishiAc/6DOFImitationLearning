#!/usr/bin/env python
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
import rospy
from std_msgs.msg import Float64MultiArray, Float64
import pygame
from sensor_msgs.msg import CompressedImage
import pandas as pd
from cv_bridge import CvBridge
import cv2
import math
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
import rospy
import torch.nn as nn
from cv_bridge import CvBridge
import cv2
import math
import time
bridge = CvBridge()
# Define the model class
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        input_shape = (3,128,128)
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

    def forward(self, x):
        # Define forward pass here
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
def publish_joint_angles(joint_angles):
    msg = Float64MultiArray()
    msg.data = joint_angles
    joint1pub.publish(msg.data[1])
    joint2pub.publish(msg.data[3])
    joint3pub.publish(msg.data[4])
    rate.sleep()
angle=0
def publish_action(action):
    global angle
    if action==5: #Up
        target_position[0]+=.005
        

    elif action==3: #Left
        target_position[1]-=.005
        
    elif action==6: #Down
        target_position[0]-=.005
        

    elif action==4: #Right
        target_position[1]+=.005
        
    elif action==2: #Backward
        target_position[2]-=.005
        

    elif action==1: #Forward
        target_position[2]+=.005
        
    elif action==7: #GripperUp
        angle +=.05
        joint4pub.publish(angle)
    #print("X axis value: " + str(currentXval))

    elif action==8: #GripperDown
        angle -=.05
        joint4pub.publish(angle)
    #print("Y axis value: " + str(currentYval))

    elif action==10: #GripperClosed
        gripper1pub.publish(-.8)
    elif action==9: #GripperOpen
        gripper1pub.publish(1.2)
    
    joint_angles = robot.inverse_kinematics(target_position=target_position,target_orientation=target_orientation)
    if action!= None:
        try:
            publish_joint_angles(joint_angles)
        except rospy.ROSInterruptException:
            pass
# Load the model
def load_model(file_path, num_classes):
    model = CNNModel(num_classes=num_classes)
    model.load_state_dict(torch.load(file_path))
    model.eval()  # Set to evaluation mode
    return model

# Initialize model
def preprocess_image(rawimage):
    image = Image.fromarray(cv2.cvtColor(rawimage, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        predicted_labels = torch.softmax(output, dim=1).squeeze().tolist()  # Convert logits to probabilities
    return predicted_labels
def callback(msg):
    #print(msg)
    cv2image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    imgtensor = preprocess_image(cv2image)
    labels = predict(model,imgtensor)
    highestProb = 0
    index = 0
    for i in range(len(labels)):
        if labels[i]>highestProb:
            highestProb=labels[i]
            index=i
    state_name = state_names[index] if index < len(state_names) else "Unknown"
    print(f"Highest Probability: {highestProb}, Label Index: {index}, State: {state_name}")
    if highestProb>.4:
        publish_action(index)
   # rospy.sleep(.5)
if __name__ == '__main__':
    rospy.init_node("MLTest")
    joint1pub = rospy.Publisher('/joint1_controller/command', Float64, queue_size=10)
    joint2pub = rospy.Publisher('/joint2_controller/command', Float64, queue_size=10)
    joint3pub = rospy.Publisher('/joint3_controller/command', Float64, queue_size=10)
    joint4pub = rospy.Publisher('/joint4_controller/command', Float64, queue_size=10)
    gripper1pub = rospy.Publisher('/r_joint_controller/command', Float64, queue_size=10)
    rate = rospy.Rate(10)
    cv2image = None
    angle=0
    robot = Chain.from_urdf_file('/home/gov_laptop/ArmDemo/jetautoArm.urdf')
    print(robot)
    # Define the target position and orientation (end-effector pose)
    #Bounds for X are 0.265 far and 0.04 Close
    #Bounds for Y are -0.225 Left and 0.225 Right
    #bounds for Z are tricky but out of robot is 0.14 Down and 0.475 Up
    # Define target position and orientation
    #gripper1pub.publish(1.2)
    target_position = [.2, 0, .3]
    target_orientation = [0, 0, 0]
    joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)

    try:
        publish_joint_angles(joint_angles)
    except rospy.ROSInterruptException:
        pass

    joint4pub.publish(0)

    time.sleep(1)
    target_position = [.1, 0, .3]
    target_orientation = [0, 0, 0]
    joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)

    try:
        publish_joint_angles(joint_angles)
    except rospy.ROSInterruptException:
        pass

    joint4pub.publish(1)
    gripper1pub.publish(1.2)
    state_names = [
    "Nothing","Up", "Down", "Left", "Right", "Forward",
    "Backward", "GripperDown", "GripperUp", "GripperOpen", "GripperClosed", # Ensure this matches the number of classes in your model
]
    num_classes = 11  # Adjust based on your dataset
    model = load_model('/home/gov_laptop/ArmDemo/FullArmGrabModel.pth', num_classes=num_classes)
    camerasub = rospy.Subscriber("/usb_cam/image_color/compressed", CompressedImage, callback,queue_size=1,buff_size=52428800)

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")
