import numpy as np
import cv2 as cv
from IPython.core.debugger import set_trace

def postprocess(frame, outs, classes, yolo):
    # Increase the confidence threshold
    confThreshold = 0.6  # Adjust the threshold to a higher value
    nmsThreshold = 0.4
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    def drawPred(classId, conf, left, top, right, bottom, bbox_color):
        # Increase border thickness
        border_thickness = 6  # Thick border for high visibility on a high-resolution image

        # Use bbox_color for the bounding box
        cv.rectangle(frame, (left, top), (right, bottom), bbox_color, thickness=border_thickness)

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)
        # Adjust font size and thickness for high-resolution image
        font_scale = 2  # Larger font scale for readability
        label_thickness = 2  # Thicker text for better visibility

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
        top = max(top, labelSize[1])

        # Define a different color for the tag
        tag_color = (255, 255, 255)  # White color for the tag

        # Draw the label's background
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), tag_color, cv.FILLED)

        # Choose a contrasting color for the text
        text_color = (0, 0, 0)  # Black color for the text

        # Draw the text
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness=label_thickness)



    layerNames = yolo.getLayerNames()
    lastLayerId = yolo.getLayerId(layerNames[-1])
    lastLayer = yolo.getLayer(lastLayerId)
    
    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'DetectionOutput':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for out in outs:
            for detection in out[0, 0]:
                confidence = detection[2]
                if confidence > confThreshold:
                    left = int(detection[3])
                    top = int(detection[4])
                    right = int(detection[5])
                    bottom = int(detection[6])
                    width = right - left + 1
                    height = bottom - top + 1
                    if width <= 2 or height <= 2:
                        left = int(detection[3] * frameWidth)
                        top = int(detection[4] * frameHeight)
                        right = int(detection[5] * frameWidth)
                        bottom = int(detection[6] * frameHeight)
                        width = right - left + 1
                        height = bottom - top + 1
                    classIds.append(int(detection[1]) - 1)  # Skip background label
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    elif lastLayer.type == 'Region':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    outNames = yolo.getUnconnectedOutLayersNames()

    if len(outNames) > 1 or lastLayer.type == 'Region' and args.backend != cv.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            nms_indices = nms_indices[:] if len(nms_indices) else [] # was nms_indices = nms_indices[:, 0] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))
    
    # Set color
    color_palette = [
        (255, 0, 0),    # Color for class 0
        (0, 255, 0),    # Color for class 1
        (0, 0, 255),    # Color for class 2
        (255, 255, 0),  # Color for class 3
        (128, 0, 128),  # Color for class 4
        ]
    # Set counter
    color_dict = {}
    count_dict = {}
    
    for i in indices:
        classId = classIds[i]
        if classId not in color_dict:
            color_dict[classId] = color_palette[len(color_dict) % len(color_palette)]
            count_dict[classes[classId]] = 1
        else:
            count_dict[classes[classId]] += 1
        # Now get the color for the current class_id
        bbox_color = color_dict[classId]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classId, confidences[i], left, top, left + width, top + height, bbox_color)
        
    return count_dict