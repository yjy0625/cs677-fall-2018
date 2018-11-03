''' utils.py '''
import os
import cv2
import re

''' Safe mkdir that checks directory before creation. '''
def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

''' Display an image in a window with given name. '''
def show_img(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

''' Extract bounding boxes from a specified XML file. '''
def extract_bboxes_from_xml(filename):
    bboxes = []

    with open(filename, 'r') as f:
        content = ''.join(f.readlines())
        content = re.sub(r'\s+', '', content)
        bbox_re = re.compile('<bndbox>'
            + '<xmin>[0-9]*</xmin>'
            + '<ymin>[0-9]*</ymin>'
            + '<xmax>[0-9]*</xmax>'
            + '<ymax>[0-9]*</ymax>'
            + '</bndbox>')
        bbox_strings = bbox_re.findall(content)

        for bbox_str in bbox_strings:
            number_re = re.compile('>([0-9]+)<')
            bbox = [int(x) for x in number_re.findall(bbox_str)]
            bboxes.append(bbox)

    return bboxes

''' Converts bounding box format from xywh to x1y1x2y2 format. '''
def bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

''' Get area of a bounding box in xyxy format. '''
def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

''' Get overlap area of two bounding boxes, returns 0 if no overlap. '''
def get_overlap_area(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[2], bbox2[2])
    y1 = max(bbox1[1], bbox2[1])
    y2 = min(bbox1[3], bbox2[3])
    if not (x1 < x2 and y1 < y2):
        return 0 # no overlap
    return (x2 - x1) * (y2 - y1)

''' Evaluate predicted bounding boxes using IOU.

Args:
    pred_bboxes: bounding boxes to be evaluated.
    gt_bboxes: ground truth bounding boxes.

Returns:
    precision: precision of the predicted bboxes.
    recall: recall value of the predicted bboxes.
    correct_bboxes: a list of correct bboxes (with iou > 0.5).
'''
def eval_iou(pred_bboxes, gt_bboxes):
    tp = 0
    correct_bboxes = []
    for gt_bbox in gt_bboxes:
        for pred_bbox in pred_bboxes:
            area_pred_bbox = get_area(pred_bbox)
            area_gt_bbox = get_area(gt_bbox)
            intersect_area = get_overlap_area(pred_bbox, gt_bbox)
            union_area = area_pred_bbox + area_gt_bbox - intersect_area
            iou = float(intersect_area) / union_area
            if iou > 0.5:
                tp += 1
                correct_bboxes.append(pred_bbox)
                break
    precision = tp / len(pred_bboxes)
    recall = tp / len(gt_bboxes)
    return precision, recall, correct_bboxes

''' Draw a list of bounding boxes in a given image with a color.

Args:
    bboxes: a list of bounding boxes.
    img: image to be annotated on.
    color: optional parameter specifying bounding box color.

Returns:
    img_with_bboxes: an image with annotated bounding boxes.
'''
def draw_bboxes_on_image(bboxes, img, color='g'):
    if color == 'b':
        color_tuple = (255, 0, 0)
    elif color == 'g':
        color_tuple = (0, 255, 0)
    elif color == 'r':
        color_tuple = (0, 0, 255)
    elif color == 'k':
        color_tuple = (0, 0, 0)

    img_with_bboxes = img.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        img_with_bboxes = cv2.rectangle(img_with_bboxes, (x1, y1), (x2, y2), color_tuple, 1)
    return img_with_bboxes