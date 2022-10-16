def get_IOU(box_pred, box_gt):
    #Calculate area of individual boxes using w x h
    box_pred_area = box_pred[2] * box_pred[3]
    box_gt_area = box_gt[2] * box_gt[3]
    #Calculate sum of areas
    total_area = box_pred_area + box_gt_area
    #Calculate intersection area of boxes
    x_left = max(box_pred[0], box_gt[0])
    y_top = max(box_pred[1], box_gt[1])
    x_right = min(box_pred[0] + box_pred[2] ,box_gt[0] + box_gt[2])
    y_bottom = min(box_pred[1] + box_pred[3], box_gt[1] + box_gt[3])
    #Return 0 if boxes do not overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = total_area - intersection_area
    
    IOU = intersection_area/float(union_area) 
    return IOU


def get_mean_IOU(y_pred, y_truth):
    assert len(y_pred) == len(y_truth)
    n_boxes = len(y_pred)
    size = (2560,1024)
    IOU_sum = 0.0
    for pred_box, gt_box in zip(y_pred, y_truth):
        pred_box = get_scaled_bbox(pred_box, size)
        gt_box = get_scaled_bbox(gt_box, size)
        IOU_sum += get_IOU(pred_box, gt_box)
    mean_IOU = IOU_sum/float(n_boxes)
    return mean_IOU*100

