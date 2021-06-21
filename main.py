import pybullet as p
import time
from random import *
import pybullet_data
import numpngw
import math
#=======================================================================================================================
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

from object_detection.utils import ops as utils_ops

from utils import label_map_util
from utils import visualization_utils as vis_util
matplotlib.use('qt5agg')
######################################################### Simulation Setup #############################################
# To setup the simulation evrionment
# The simulation envrionment should be same with
clid = p.connect(p.GUI)
if (clid < 0):
    p.connect(p.GUI)

# p.setGravity(0,0,-9.8)
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

#########################################################Load Objects############################################################
possibleangle = [0,1,2,3,4,5]
possiblemulti = [[1.2, -0.1], [1.2, 0.1], [1.1, 0], [1, -0.1], [1, 0.1]]
possibleyz = [-0.1, 0, 0.1]
object_1_path = 'Data/random_urdfs/155/155.urdf'
object_2_path = 'Data/random_urdfs/156/156.urdf'
object_3_path = 'Data/random_urdfs/157/157.urdf'
object_4_path = 'Data/random_urdfs/158/158.urdf'
object_5_path = 'Data/random_urdfs/159/159.urdf'

zebras = 0
explored = []
while zebras < 5:
    posx = (sample(possiblemulti, 1))[0]
    posz = (sample(possibleyz, 1))[0]
    pasta = 0

    for val in explored:
        if posx == val:
            pasta = 1

    if pasta == 0:
        explored.append(posx)
        test = [posx[0], posx[1], posz]
        if zebras == 0:
            objectId = p.loadURDF(object_1_path, test, p.getQuaternionFromEuler([((sample(possibleangle, 1))[0]), ((sample(possibleangle, 1))[0]), ((sample(possibleangle, 1))[0])]))
            depthz1 = posz
        elif zebras == 1:
            objectId2 = p.loadURDF(object_2_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            depthz2 = posz
        elif zebras == 2:
            objectId3 = p.loadURDF(object_3_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            depthz3 = posz
        elif zebras == 3:
            objectId4 = p.loadURDF(object_4_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            depthz4 = posz
        elif zebras == 4:
            objectId5 = p.loadURDF(object_5_path, test, p.getQuaternionFromEuler([(sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0], (sample(possibleangle, 1))[0]]))
            depthz5 = posz
        zebras = zebras + 1


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
numpngw.write_png('images/multi/depth/' + str(test) +'.png', z)

y = rgb_opengl
z = (65535 * ((y - y.min()) / y.ptp())).astype(np.uint16)
numpngw.write_png('test_images/image1.png', z)

fill_color = 0
image = Image.open('test_images/image1.png')
background = Image.new(image.mode[:-1], image.size, fill_color)
background.paste(image, image.split()[-1])
image = background
image.save('test_images/image1.jpg')

y = seg_opengl
z = (65535 * ((y - y.min()) / y.ptp())).astype(np.uint16)
numpngw.write_png('images/multi/seg/' + str(test) + '.png', z)


########################################################Import the trained recognition model##############################################
##=====================================================================================================================
MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'
#======================================================================================================================

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

##=====================================================================================================================
obj_to_grasp = 3
iii=0
while iii<5:
    if obj_to_grasp == 1:
        depthz = depthz1
    elif obj_to_grasp == 2:
        depthz = depthz2
    elif obj_to_grasp == 3:
        depthz = depthz3
    elif obj_to_grasp == 4:
        depthz = depthz4
    elif obj_to_grasp == 5:
        depthz = depthz5
    iii= iii+1
PATH_TO_TEST_IMAGES_DIR = 'test_images/image1.jpg'
##=====================================================================================================================
# Size, in inches, of the output images.


IMAGE_SIZE = (12, 8)



def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


image = Image.open(PATH_TO_TEST_IMAGES_DIR)
# the array based representation of the image will be used later in order to prepare the
# result image with boxes and labels on it.
image_np = load_image_into_numpy_array(image)
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
image_np_expanded = np.expand_dims(image_np, axis=0)
# Actual detection.
output_dict = run_inference_for_single_image(image_np, detection_graph)
# Visualization of the results of a detection.
vis_util.visualize_boxes_and_labels_on_image_array(
  image_np,
  output_dict['detection_boxes'],
  output_dict['detection_classes'],
  output_dict['detection_scores'],
  category_index,
  instance_masks=output_dict.get('detection_masks'),
  use_normalized_coordinates=True,
  line_thickness=3)
#line thickness was 8
plt.figure(figsize=IMAGE_SIZE)
plt.imshow(image_np)
plt.show()



# This is the way I'm getting my coordinates
boxes = output_dict['detection_boxes']
# get all boxes from an array
max_boxes_to_draw = boxes.shape[0]
# get scores to get a threshold
scores = output_dict['detection_scores']
# this is set as a default but feel free to adjust it to your needs
min_score_thresh = .5
# iterate over all objects found
for jjj in range(min(max_boxes_to_draw, boxes.shape[0])):
  #
  if scores is None or scores[jjj] > min_score_thresh:
      #the number in the next if statement tells which object you wanna get cordinate of; in this case 2 asin obj 2
      if output_dict['detection_classes'][jjj] == obj_to_grasp:
          cordinates = 256 * boxes[jjj]
          print('xmin:' + str(cordinates[0]))
          print('ymin:' + str(cordinates[1]))
          print('xmax:' + str(cordinates[2]))
          print('ymax:' + str(cordinates[3]))

xmin = cordinates[0]
ymin = cordinates[1]
xmax = cordinates[2]
ymax = cordinates[3]


########################################################Based on the recognition, get the depth region of for the same region#############################################
def get_coord(u,v,d):
    """
    xman, xmin, ymax, and ymin you sould be able to from the detected depth region of the object in the scene.
    u = (xmax-xmin)/2
    y = (ymax-ymin)/2
    :param u: image coordination of the point of the object in x-axis
    :param v: image coordination of the point of the object in y-axis
    :param d: depth information of the point
    :return: world coordinate of the object.
    """
    # K is the intrinsic matrix
    fx = 1/(math.tan(60/2))
    fy = 1/(math.tan(60/2))
    px = 0
    py = 0
    K = [[fx,0,px],[0,600.688,py],[0,0,1]]
    C = [u, v, d]
    cam = [C[0]*C[2],C[1]*C[2],C[2]]
    kinv = np.linalg.inv(K)
    coordinate = np.matmul(kinv,cam)
    return coordinate

wx = (xmax-xmin)/2
wy = (ymax-ymin)/2
world_cord = get_coord(wx,wy,depthz)
print("world cordinates are :")
print(world_cord)


########################################################Use the project_joint_control.py file to get informtion#############################################
"""
The results of coodination from previous are the coodination of objects. Please consider the volume of objects 
to modify it to update into the palmP function in the project_joint_control.py 

You should get these information from the porject_joint_control.py funciton
            print("Thumb Position: ", p.getLinkState(sawyerId, 62)[0])  # get position of thumb tip
            print("Thumb Orientation: ", p.getLinkState(sawyerId, 62)[1])  # get orientation of thumb tip
            print("Index Position: ", p.getLinkState(sawyerId, 51)[0])
            print("Index Orientation: ", p.getLinkState(sawyerId, 51)[1])
            print("Mid: Position: ", p.getLinkState(sawyerId, 42)[0])
            print("Mid: Orientation: ", p.getLinkState(sawyerId, 42)[1])
            print("Ring Position: ", p.getLinkState(sawyerId, 33)[0])
            print("Ring Orientation: ", p.getLinkState(sawyerId, 33)[1])
            print("Pinky Position: ", p.getLinkState(sawyerId, 24)[0])
            print("Pinky Orientation: ", p.getLinkState(sawyerId, 24)[1])
            print("Palm Position: ", p.getLinkState(sawyerId, 20)[0])
            print("Palm Orientation: ", p.getLinkState(sawyerId, 20)[1])
"""
########################################################Use the porject_IK.py file to get informtion#############################################

"""
Upate parameters you have got from previous funcitons. 
Use the project_IK.py to grasp target object 
"""