#!/usr/bin/env python
import rospy
import pygame
import os
import sys
import signal
import time
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

bridge = CvBridge()


#
# REFER TO JOINT_STATES_SAVE.py FOR ANY COMMENTS ON THIS CODE
#


# Initialize Pygame
pygame.init()
joysticks = []
clock = pygame.time.Clock()
keepPlaying = True
currentZval = 0
currentXbackward = 0
currentYval = 0
currentXforward = 0
gripperaxismover = 0
angle = 0
isgripperclosed = False

lastYval = 0 
lastZval = 0
lastXforward = 0
lastXbackward = 0   
imageNum = 0

def publish_joint_angles(joint_angles):
    msg = Float64MultiArray()
    msg.data = joint_angles
    joint1pub.publish(msg.data[1])
    joint2pub.publish(msg.data[3])
    joint3pub.publish(msg.data[4])
    rate.sleep()

def teleop():
    global keepPlaying, currentZval, currentXbackward, currentYval, currentXforward, gripperaxismover
    global angle, isgripperclosed, lastYval, lastZval, lastXforward, lastXbackward
    buttonpressed = False

    while keepPlaying:
        try:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    if event.axis == 1:
                        currentZval = event.value
                    elif event.axis == 0:
                        currentYval = event.value
                    elif event.axis == 2:
                        currentXbackward = event.value
                    elif event.axis == 5:
                        currentXforward = event.value
                    elif event.axis == 4:
                        gripperaxismover = event.value
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:
                        buttonpressed = True
                        isgripperclosed = not isgripperclosed
                    elif event.button == 2:
                        action = 0

            if lastZval == currentZval:
                currentZval = lastZval
            if lastYval == currentYval:
                currentYval = lastYval
            if lastXbackward == currentXbackward:
                currentXbackward = lastXbackward
            if lastXforward == currentXforward:
                currentXforward = lastXforward

            action = None
            if currentXforward > .5:
                action = 5
                target_position[0] += .005
            elif currentYval < -.7:
                action = 3
                target_position[1] -= .005
            elif currentXbackward > .5:
                action = 6
                target_position[0] -= .005
            elif currentYval > .7:
                action = 4
                target_position[1] += .005
            elif currentZval > .7:
                action = 2
                target_position[2] -= .005
            elif currentZval < -.7:
                action = 1
                target_position[2] += .005
            elif gripperaxismover > .7:
                action = 7
                angle += .05
                joint4pub.publish(angle)
            elif gripperaxismover < -.7:
                action = 8
                angle -= .05
                joint4pub.publish(angle)
            elif buttonpressed:
                buttonpressed = False
                if isgripperclosed:
                    gripper1pub.publish(-.8)
                    action = 10
                else:
                    gripper1pub.publish(1.2)
                    action = 9

            state_names = [
                "Nothing", "Up", "Down", "Left", "Right", "Forward",
                "Backward", "GripperDown", "GripperUp", "GripperOpen", "GripperClosed",
            ]
            state_name = state_names[action] if action is not None and action < len(state_names) else "Unknown"
            print(f"Label Index: {action}, State: {state_name}")

            joint_angles = robot.inverse_kinematics(target_position=target_position, target_orientation=target_orientation)
            if action is not None:
                try:
                    publish_joint_angles(joint_angles)
                except rospy.ROSInterruptException:
                    pass

            lastXforward = currentXforward
            lastXbackward = currentXbackward
            lastZval = currentZval
            lastYval = currentYval

        except KeyboardInterrupt:
            print("KeyboardInterrupt caught, stopping the program...")
            break

def signal_handler(sig, frame):
    print("Signal handler called with signal:", sig)
    global keepPlaying
    keepPlaying = False
    pygame.quit()
    sys.exit(0)

if __name__ == '__main__':
    pygame.init()
    joysticks = []
    clock = pygame.time.Clock()
    keepPlaying = True

    # for al the connected joysticks
    for i in range(0, pygame.joystick.get_count()):
        # create an Joystick object in our list
        joysticks.append(pygame.joystick.Joystick(i))
        # initialize the appended joystick (-1 means last array item)
        joysticks[-1].init()
        # print a statement telling what the name of the controller is
        print ("Detected joystick "),joysticks[-1].get_name(),"'"
    rospy.init_node('joint_angle_publisher', anonymous=True)
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    rospy.init_node('joint_angle_publisher', anonymous=True)
    joint1pub = rospy.Publisher('/joint1_controller/command', Float64, queue_size=10)
    joint2pub = rospy.Publisher('/joint2_controller/command', Float64, queue_size=10)
    joint3pub = rospy.Publisher('/joint3_controller/command', Float64, queue_size=10)
    joint4pub = rospy.Publisher('/joint4_controller/command', Float64, queue_size=10)
    gripper1pub = rospy.Publisher('/r_joint_controller/command', Float64, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz

    robot = Chain.from_urdf_file('/home/gov_laptop/ArmDemo/jetautoArm.urdf')
    print(robot)

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
    imgs = os.listdir("/home/gov_laptop/ArmDemo/FullArmPics")
    highest_number = 0
    for img in imgs:
        img1 = img.split('c')
        img2 = img1[1].split('.')
        if highest_number < int(img2[0]):
            highest_number = int(img2[0])

    imageNum = highest_number
    print(imageNum)

    try:
        teleop()
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        pygame.quit()  # Clean up Pygame resources
        sys.exit(0)    # Exit with a success status
