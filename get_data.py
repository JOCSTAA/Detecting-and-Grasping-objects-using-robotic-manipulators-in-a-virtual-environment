import pybullet as p
import time
from random import *
import math
import random
import numpy as np
import pybullet_data
import random
import glob
#import png
import os
import numpngw
from datetime import datetime


######################################################### Simulation Setup ############################################################################

clid = p.connect(p.GUI)
if (clid < 0):
    p.connect(p.GUI)

#p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_path = 'Data/plane/plane.urdf'
planeId = p.loadURDF("plane.urdf", [0, 0, -1])

# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# the path of robot urdf, should be under the folder ../Data/sawyer_robot/sawyer_description/urdf/sawyer.urdf
robot_path = 'Data/sawyer_robot/sawyer_description/urdf/sawyer.urdf'
sawyerId = p.loadURDF(robot_path, [0, 0, 0],
                      [0, 0, 0, 3])
# the path of the table, should be under the folder ../Data/table/table.urdf
table_path = 'Data/table/table.urdf'
tableId = p.loadURDF(table_path, [1.4, 0, -1], p.getQuaternionFromEuler([0, 0, 1.56]))  # load table


######################################################### Load Object Here!!!!#############################################################################

# load the assigned objects, change file name to load different objects
# p.loadURDF(finlename, position([X,Y,Z]), orientation([a,b,c,d]))
# center of table is at [1.4,0, -1], adjust postion of object to put it on the table

# the path of the objects, should be under Data/random_urdfs/000/000.urdf
# Example
possiblex = [1.0, 1.1, 1.2]
possibleyz = [-0.1, 0, 0.1]
possibleangle = [0,1,2,3,4,5]

possition = [0,0,0,0,0,0]
explored = []
path = []
text = []
imgpath = []

path.append('Data/random_urdfs/155/155.urdf')
path.append('Data/random_urdfs/156/156.urdf')
path.append('Data/random_urdfs/157/157.urdf')
path.append('Data/random_urdfs/158/158.urdf')
path.append('Data/random_urdfs/159/159.urdf')

text.append('Data/dtd/images/banded/banded_0002.jpg')
text.append('Data/dtd/images/blotchy/blotchy_0003.jpg')
text.append('Data/dtd/images/braided/braided_0002.jpg')
text.append('Data/dtd/images/bubbly/bubbly_0038.jpg')
text.append('Data/dtd/images/bumpy/bumpy_0014.jpg')

imgpath.append('images/ob1/')
imgpath.append('images/ob2/')
imgpath.append('images/ob3/')
imgpath.append('images/ob4/')
imgpath.append('images/ob5/')

ii = 0
while ii <= 4:
    print(ii)
    object_1_path = path[ii]
    cnt = 0
    while cnt < 50:
        print('cnt'+str(cnt))
        #makes a random pos/ori
        posx = (sample(possiblex, 1))[0]
        posy = (sample(possibleyz, 1))[0]
        posz = (sample(possibleyz, 1))[0]
        angx = (sample(possibleangle, 1))[0]
        angy = (sample(possibleangle, 1))[0]
        angz = (sample(possibleangle, 1))[0]


        possition[0] = posx
        possition[1] = posy
        possition[2] = posz
        possition[3] = angx
        possition[4] = angy
        possition[5] = angz


        #loads obj at random pos/ori
        object_1_path = path[ii]
        objectId = p.loadURDF(object_1_path, [posx, posy, posz], p.getQuaternionFromEuler([angx, angy, angz]))



        # apply texture to objects
        # apply randomly textures from the dtd dataset to each object, each object's texture should be the different.

        # texture_paths = Data/dtd/images/banded/banded_0002.jpg
        # example:

        text_paths = text[ii]
        text_id = p.loadTexture(text_paths)
        p.changeVisualShape(objectId,-1,textureUniqueId=text_id)



        ########################################################Insert Camera####################################################################################
        # Using the inserted camera to caputure data for training. Save the captured numpy array as image files for later training process.

        width = 256
        height = 256

        fov = 60
        aspect = width / height
        near = 0.02
        far = 1

        # the view_matrix should contain three arguments, the first one is the [X,Y,Z] for camera location
        #												  the second one is the [X,Y,Z] for target location
        #											      the third one is the  [X,Y,Z] for the top of the camera
        # Example:
        # viewMatrix = pb.computeViewMatrix(
        #     cameraEyePosition=[0, 0, 3],
        #     cameraTargetPosition=[0, 0, 0],
        #     cameraUpVector=[0, 1, 0])
        view_matrix = p.computeViewMatrix([1.05,-0.05,0.4], [1.1, 0, 0], [1, 0, 0])

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


        # Get depth values using the OpenGL renderer
        images = p.getCameraImage(width,
                                  height,
                                  view_matrix,
                                  projection_matrix,
                                  shadow=True,
                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
        depth_buffer_opengl = np.reshape(images[3], [width, height])
        depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
        seg_opengl = np.reshape(images[4], [width, height]) * 1. / 255.
        time.sleep(1)

        y = depth_opengl
        z = (65535*((y - y.min())/y.ptp())).astype(np.uint16)
        numpngw.write_png(imgpath[ii]+'depth/' + str(cnt) +'.png', z)

        y = rgb_opengl
        z = (65535 * ((y - y.min()) / y.ptp())).astype(np.uint16)
        numpngw.write_png(imgpath[ii]+'rgb/' + str(cnt) + '.png', z)

        y = seg_opengl
        z = (65535 * ((y - y.min()) / y.ptp())).astype(np.uint16)
        numpngw.write_png(imgpath[ii]+'seg/' + str(cnt) + '.png', z)
        print(path[ii], text[ii], imgpath[ii])

        p.resetBasePositionAndOrientation(objectId, [2, 2, 2], [0, 0, 0, 1])
        cnt = cnt + 1
    ii = ii + 1

possiblemulti = [[1.2, -0.1], [1.2, 0.1], [1.1, 0], [1, -0.1], [1, 0.1]]
object_1_path = 'Data/random_urdfs/155/155.urdf'
object_2_path = 'Data/random_urdfs/156/156.urdf'
object_3_path = 'Data/random_urdfs/157/157.urdf'
object_4_path = 'Data/random_urdfs/158/158.urdf'
object_5_path = 'Data/random_urdfs/159/159.urdf'


cnt = 0
while cnt < 100:
    print(cnt)
    explored = []
    zebras = 0
    while zebras < 5:
        posx = (sample(possiblemulti, 1))[0]
        posz = (sample(possibleyz, 1))[0]
        pasta = 0

        for val in explored:
            if posx == val:
                pasta = 1

        if pasta == 0:
            print(zebras)
            explored.append(posx)
            test = [posx[0], posx[1], posz]
            print(test)
            if zebras == 0:
                objectId = p.loadURDF(object_1_path, test, p.getQuaternionFromEuler([((sample(possibleangle, 1))[0]), ((sample(possibleangle, 1))[0]), ((sample(possibleangle, 1))[0])]))
            elif zebras == 1:
                objectId2 = p.loadURDF(object_2_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            elif zebras == 2:
                objectId3 = p.loadURDF(object_3_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            elif zebras == 3:
                objectId4 = p.loadURDF(object_4_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            elif zebras == 4:
                objectId5 = p.loadURDF(object_5_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            zebras = zebras + 1
        # apply texture to objects
        # apply randomly textures from the dtd dataset to each object, each object's texture should be the different.

        # texture_paths = Data/dtd/images/banded/banded_0002.jpg
        # example:

    text_paths = 'C:/Users/josho/Desktop/SCHOOL WORK/SPRING 20/AI in robotics/final project/Project 2/Data/dtd/images/banded/banded_0002.jpg'
    text_id = p.loadTexture(text_paths)
    p.changeVisualShape(objectId, -1, textureUniqueId=text_id)
    text_paths2 = 'C:/Users/josho/Desktop/SCHOOL WORK/SPRING 20/AI in robotics/final project/Project 2/Data/dtd/images/blotchy/blotchy_0003.jpg'
    text_id2 = p.loadTexture(text_paths2)
    p.changeVisualShape(objectId2, -1, textureUniqueId=text_id2)
    text_paths3 = 'C:/Users/josho/Desktop/SCHOOL WORK/SPRING 20/AI in robotics/final project/Project 2/Data/dtd/images/braided/braided_0002.jpg'
    text_id3 = p.loadTexture(text_paths3)
    p.changeVisualShape(objectId3, -1, textureUniqueId=text_id3)
    text_paths4 = 'C:/Users/josho/Desktop/SCHOOL WORK/SPRING 20/AI in robotics/final project/Project 2/Data/dtd/images/bubbly/bubbly_0038.jpg'
    text_id4 = p.loadTexture(text_paths4)
    p.changeVisualShape(objectId4, -1, textureUniqueId=text_id4)
    text_paths5 = 'C:/Users/josho/Desktop/SCHOOL WORK/SPRING 20/AI in robotics/final project/Project 2/Data/dtd/images/bumpy/bumpy_0014.jpg'
    text_id5 = p.loadTexture(text_paths5)
    p.changeVisualShape(objectId5, -1, textureUniqueId=text_id5)
    ########################################################Insert Camera####################################################################################
    # Using the inserted camera to caputure data for training. Save the captured numpy array as image files for later training process.

    width = 256
    height = 256

    fov = 60
    aspect = width / height
    near = 0.02
    far = 1

    # the view_matrix should contain three arguments, the first one is the [X,Y,Z] for camera location
    #												  the second one is the [X,Y,Z] for target location
    #											      the third one is the  [X,Y,Z] for the top of the camera
    # Example:
    # viewMatrix = pb.computeViewMatrix(
    #     cameraEyePosition=[0, 0, 3],
    #     cameraTargetPosition=[0, 0, 0],
    #     cameraUpVector=[0, 1, 0])
    view_matrix = p.computeViewMatrix([1.05,-0.05,0.4], [1.1, 0, 0], [1, 0, 0])

    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)


    # Get depth values using the OpenGL renderer
    images = p.getCameraImage(width,
                              height,
                              view_matrix,
                              projection_matrix,
                              shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
    depth_buffer_opengl = np.reshape(images[3], [width, height])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    seg_opengl = np.reshape(images[4], [width, height]) * 1. / 255.
    time.sleep(1)

    y = depth_opengl
    z = (65535*((y - y.min())/y.ptp())).astype(np.uint16)
    numpngw.write_png('images/multi/depth/' + str(cnt) +'.png', z)

    y = rgb_opengl
    z = (65535 * ((y - y.min()) / y.ptp())).astype(np.uint16)
    numpngw.write_png('images/multi/rgb/' + str(cnt) + '.png', z)

    y = seg_opengl
    z = (65535 * ((y - y.min()) / y.ptp())).astype(np.uint16)
    numpngw.write_png('images/multi/seg/' + str(cnt) + '.png', z)

    p.resetBasePositionAndOrientation(objectId, [2, 2, 2], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(objectId2, [2, 2, 2], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(objectId3, [2, 2, 2], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(objectId4, [2, 2, 2], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(objectId5, [2, 2, 2], [0, 0, 0, 1])


    cnt = cnt + 1

########################################################################################################################################################

# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.resetBasePositionAndOrientation(sawyerId, [0, 0, 0], [0, 0, 0, 1])

# bad, get it from name! sawyerEndEffectorIndex = 18
sawyerEndEffectorIndex = 16
numJoints = p.getNumJoints(sawyerId)  # 65 with ar10 hand
# print(p.getJointInfo(sawyerId, 3))
# useRealTimeSimulation = 0
# p.setRealTimeSimulation(useRealTimeSimulation)
# p.stepSimulation()
# all R joints in robot
js = [3, 4, 8, 9, 10, 11, 13, 16, 21, 22, 23, 26, 27, 28, 30, 31, 32, 35, 36, 37, 39, 40, 41, 44, 45, 46, 48, 49, 50,
      53, 54, 55, 58, 61, 64]
# lower limits for null space
ll = [-3.0503, -5.1477, -3.8183, -3.0514, -3.0514, -2.9842, -2.9842, -4.7104, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17,
      0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17, 0.85, 0.34,
      0.17]
# upper limits for null space
ul = [3.0503, 0.9559, 2.2824, 3.0514, 3.0514, 2.9842, 2.9842, 4.7104, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57,
      0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 1.57, 1.57, 0.17, 2.15, 1.5, 1.5]
# joint ranges for null space
jr = [0, 0, 0, 0, 0, 0, 0, 0, 1.4, 1.4, 1.4, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4, 0, 1.4, 1.4,
      0, 1.4, 1.4, 0, 1.3, 1.16, 1.33]
# restposes for null space
rp = [0] * 35
# joint damping coefficents
jd = [1.1] * 35

i = 0
while 1:
    i += 1
    p.stepSimulation()
    # 0.03 sec/frame
    time.sleep(0.03)
    # increase i to increase the simulation time
    if (i == 20000000):
        break


