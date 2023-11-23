import numpy as np
import cv2

def tps_transform(img, pts1, pts2):
    assert len(pts1) == len(pts2)
    num_pts = len(pts1)
    
    matches = [cv2.DMatch(i,i,0) for i in range(num_pts)]
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(pts2.reshape(-1,num_pts,2), pts1.reshape(-1,num_pts,2), matches)
    tps_img = tps.warpImage(img, borderMode=cv2.BORDER_REPLICATE)
    
    return tps_img

def cloth_transform(img, key_pts1, key_pts2, kind):
    key_idxs_dict = {"short-sleeved-shirt": [[0,1,3,5,8,14,15,16,22], [2,6], [2]],
                     "long_sleeved_shirt": [],
                     "short_sleeved_outwear": [],
                     "long_sleeved_outwear": [],
                     "vest": [],
                     "sling": [],
                     "shorts": [],
                     "trousers": [],
                     "skirt": [],
                     "short_sleeved_dress": [],
                     "long_sleeved_dress": [],
                     "vest_dress": [],
                     "sling_dress": []}
    assert kind in key_idxs_dict
    key_idxs = key_idxs_dict[kind][0]
    scale_idxs = key_idxs_dict[kind][1]
    shift_idx = key_idxs_dict[kind][2][0]
    pts1 = key_pts1.pred_instances.keypoints[0][key_idxs]
    pts2 = key_pts2.pred_instances.keypoints[0][key_idxs]
    num_pts = len(pts1)
    h, w = img.shape[:-1]

    # scale
    dist1 = np.linalg.norm(pts1[scale_idxs[0]]-pts1[scale_idxs[1]])
    dist2 = np.linalg.norm(pts2[scale_idxs[0]]-pts2[scale_idxs[1]])
    scale_value = dist2 / dist1
    img = cv2.resize(img, (int(scale_value*w), int(scale_value*h)))
    pts1 *= scale_value

    # shift
    shift_xy = pts1[shift_idx] - pts2[shift_idx]
    pts2 += shift_xy

    # tps transformation
    transformed_img = tps_transform(img, pts1, pts2)

    return transformed_img


