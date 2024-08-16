from ikpy.chain import Chain
import rospy
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import CompressedImage
import pygame
import pandas as pd
from cv_bridge import CvBridge
import cv2
import os
import time
from sensor_msgs.msg import JointState
import message_filters

# Initialize the CvBridge
bridge = CvBridge()

# Initialize pygame
pygame.init()
joysticks = []
clock = pygame.time.Clock()
keepPlaying = True
currentZval = 0
currentXbackward = 0
currentYval = 0
currentXforward = 0
gripperaxismover = 0
angle = 1
isgripperclosed = False

lastYval = 0 
lastZval = 0
lastXforward = 0
lastXbackward = 0   
imageNum = 0

# Load dataset
df = pd.read_csv("/home/gov_laptop/ArmDemo/FullArmData.csv")  # Ensure CSV contains image paths and multi-labels

#publishing the joint angles derive from inv kinematics function 
def publish_joint_angles(joint_angles):
    #Inv Kinematics outputs an array of each joint angle as [BaseJoint,Joint1,servoJoint,Joint2,Joint3,Joint4,GripperJoint]
    msg = Float64MultiArray()
    msg.data = joint_angles
    joint1pub.publish(msg.data[1]) #Joint 1 controls rotation
    joint2pub.publish(msg.data[3]) #Joint 2 controls lowest visible joint
    joint3pub.publish(msg.data[4]) #Joint 3 is the middle joint
    #joint4pub.publish(msg.data[3]) #Joint4 is the Gripper angle joint
    rate.sleep()

def teleop(): 
    global keepPlaying, currentZval, currentXbackward, currentYval, currentXforward
    global gripperaxismover, angle, isgripperclosed
    global lastYval, lastZval, lastXforward, lastXbackward
    
    buttonpressed = False #For "A" on controller, keeps track of whether gripper is open or closed
    while keepPlaying: #Lets teleop loop only run/publish once if keepPlaying is false
        #keepPlaying = not onlyOneAction
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 1: #event axis 1 and 0 are left joystick inputs, controlling up/down and left/right on robot when facing robot head-on
                    currentZval = event.value
                elif event.axis == 0: 
                    currentYval = event.value
                elif event.axis == 2: #Forward/backward on robot are left and right triggers (events 2 and 5), which is why they're split up
                    currentXbackward = event.value
                elif event.axis == 5:
                    currentXforward = event.value
                elif event.axis == 4: #gripper angle up/down is right joystick up/down (event 4)
                    gripperaxismover = event.value
            if event.type == pygame.JOYBUTTONDOWN: #alternates the gripper open/close
                if event.button == 0:
                    buttonpressed = True
                    isgripperclosed = not isgripperclosed
                elif event.button == 1:
                    keepPlaying = False
                elif event.button == 2:
                    return 0
        
        # Debounce axis values, sets current coordinate values to the last ones if it didn't move. Ensures its constantly updating current positions.
        #holding in the same place will stop the gamepad from updating, so needs to be constantly updated to previous value.
        currentZval = lastZval if currentZval == lastZval else currentZval
        currentYval = lastYval if currentYval == lastYval else currentYval
        currentXbackward = lastXbackward if currentXbackward == lastXbackward else currentXbackward
        currentXforward = lastXforward if currentXforward == lastXforward else currentXforward

        # Determine action based on joystick input

        action = None #action is used to log the movement done on the controller.
        #target_position is the x,y,z coordinates the inv kinematics will try to reach. 
        if currentXforward > 0.5:
            action = 5 
            target_position[0] += 0.005 #Index 0 is x, moving it forward
        elif currentYval < -0.7:
            action = 3
            target_position[1] -= 0.005 #index 1 is y, moving it left
        elif currentXbackward > 0.5:
            action = 6
            target_position[0] -= 0.005 #moving backward
        elif currentYval > 0.7:
            action = 4
            target_position[1] += 0.005 #moving right
        elif currentZval > 0.7:
            action = 2
            target_position[2] -= 0.005 #Index 2 is z, moving arm down 
        elif currentZval < -0.7:
            action = 1
            target_position[2] += 0.005 #moving arm up
        elif gripperaxismover > 0.7: #publishing gripper axis seperate of inv kinematics
            action = 7
            angle += 0.05
            joint4pub.publish(angle)
        elif gripperaxismover < -0.7:
            action = 8
            angle -= 0.05
            joint4pub.publish(angle)
        elif buttonpressed: #changing gripper 
            buttonpressed = False
            if isgripperclosed:
                gripper1pub.publish(-0.8)
                action = 10
            else:
                gripper1pub.publish(1.2)
                action = 9
        
        # Compute joint angles and publish
        joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)
        if action is not None: #only publish if we are actually moving (action recorded)
            try:
                publish_joint_angles(joint_angles)
            except rospy.ROSInterruptException:
                pass
        
        # Update last values
        lastXforward = currentXforward
        lastXbackward = currentXbackward
        lastZval = currentZval
        lastYval = currentYval 
        # if onlyOneAction:
        #     keepPlaying=False
        return action


def callback(cammsg, jointmsg):
    global imageNum, df
    joint_states= jointmsg.position #Get only joint states array from larger message
    cv2image = bridge.compressed_imgmsg_to_cv2(cammsg, "bgr8") #make image usable
    # cv2.imshow('Image', cv2image)
    # # Wait for a key event
    # while True:
    #     key = cv2.waitKey(0)  # Wait indefinitely for a key press
    #     if key == ord('q'):  # Check if 'q' key is pressed
    #         break
    
    action = None
    while action is None: #Stays stuck in the while loop while we aren't moving, so nothing is published or written to df
        action = teleop() 
    #once we move, we break out of while loop to write image, states, and action
        
    #Saving image, writing to df, saving to csv
    cv2.imwrite(f"/home/gov_laptop/ArmDemo/FullArmPics/pic{imageNum}.jpeg", cv2image)
    print("Image Saved")
    new_row_df = pd.DataFrame([{"image_path": f"/home/gov_laptop/ArmDemo/FullArmPics/pic{imageNum}.jpeg", "action_num": str(action), "Joint_1": str(joint_states[0]),"Joint_2": str(joint_states[1]),"Joint_3": str(joint_states[2]),"Joint_4": str(joint_states[3]),"r_joint": str(joint_states[4])}])
    df = pd.concat([df, new_row_df], ignore_index=True)
    df.to_csv("/home/gov_laptop/ArmDemo/FullArmData.csv", index=False, encoding='utf-8')
    print(imageNum)
    imageNum += 1

if __name__ == '__main__':
    pygame.init()
    joysticks = []
    clock = pygame.time.Clock()
    keepPlaying = True

    # for all the connected joysticks
    for i in range(0, pygame.joystick.get_count()):
        # create an Joystick object in our list
        joysticks.append(pygame.joystick.Joystick(i))
        # initialize the appended joystick (-1 means last array item)
        joysticks[-1].init()
        # print a statement telling what the name of the controller is
        print ("Detected joystick "),joysticks[-1].get_name(),"'"
    rospy.init_node('joint_angle_publisher', anonymous=True)
    #Setting up publishers. the /command topics are the only ones allowing to pubish to the joint.
    joint1pub = rospy.Publisher('/joint1_controller/command', Float64, queue_size=10)
    joint2pub = rospy.Publisher('/joint2_controller/command', Float64, queue_size=10)
    joint3pub = rospy.Publisher('/joint3_controller/command', Float64, queue_size=10)
    joint4pub = rospy.Publisher('/joint4_controller/command', Float64, queue_size=10)
    gripper1pub = rospy.Publisher('/r_joint_controller/command', Float64, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz
    
    # Load the URDF file
    robot = Chain.from_urdf_file('/home/gov_laptop/ArmDemo/jetautoArm.urdf')
    print(robot)
    #Bounds for X are 0.265 far and 0.04 Close
    #Bounds for Y are -0.225 Left and 0.225 Right
    #bounds for Z are tricky but out of robot is 0.14 Down and 0.475 Up
    # Define target position and orientation
    #gripper1pub.publish(1.2)

    #Setting up initial starting position, gets robot unstuck
    target_position = [.2, 0, .3]
    target_orientation = [0, 0, 0]
    joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)

    try:
        publish_joint_angles(joint_angles)
    except rospy.ROSInterruptException:
        pass

    joint4pub.publish(0)

    time.sleep(1)

    #Setting up initial position where camera sees top-down view
    target_position = [.1, 0, .3]
    target_orientation = [0, 0, 0]
    joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)

    try:
        publish_joint_angles(joint_angles)
    except rospy.ROSInterruptException:
        pass

    joint4pub.publish(1)
    gripper1pub.publish(1.2)
    
    #Gets the number of the last image in the folder, so it can continue writing trials to same folder with different pic name.
    #and makes sure the csv has the most updated picture name
    imgs = os.listdir("/home/gov_laptop/ArmDemo/FullArmPics")

    highest_number = 0
    for img in imgs:
        img1 = img.split('c')
        img2 = img1[1].split('.')
        if highest_number < int(img2[0]):
            highest_number = int(img2[0])
    
    imageNum = highest_number+1
    print(imageNum)
    time.sleep(1)

    #Setting up subscribers. Using compressedImage because raw was too big of a file.
    camerasub = message_filters.Subscriber("/usb_cam/image_color/compressed", CompressedImage,queue_size=1,buff_size=52428800)
    jointsub = message_filters.Subscriber("/joint_states", JointState, queue_size=1,buff_size=52428800)
    #Synchronizing joint subscriber with camera and putting both messages in same callback
    ts = message_filters.ApproximateTimeSynchronizer([camerasub, jointsub],queue_size=1, slop =.5, allow_headerless=False)
    ts.registerCallback(callback)
    #time.sleep(1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        time.sleep(2)
