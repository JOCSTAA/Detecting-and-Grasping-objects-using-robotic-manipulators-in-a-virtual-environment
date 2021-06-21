#so detection has happened and you've got output_dict as a
# result of your inference

# then assume you've got this in your inference.py in order to draw rectangles
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks'),
    use_normalized_coordinates=True,
    line_thickness=8)

# This is the way I'm getting my coordinates
boxes = output_dict['detection_boxes']
# get all boxes from an array
max_boxes_to_draw = boxes.shape[0]
# get scores to get a threshold
scores = output_dict['detection_scores']
# this is set as a default but feel free to adjust it to your needs
min_score_thresh=.5
# iterate over all objects found
for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    #
    if scores is None or scores[i] > min_score_thresh:
        # boxes[i] is the box which will be drawn
        class_name = category_index[output_dict['detection_classes'][i]]['name']
        print ("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])