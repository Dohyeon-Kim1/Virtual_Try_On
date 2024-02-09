import cv2
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw


def remove_background(img, seg_map, kind):
    assert kind in ["body", "cloth"] 
    img = np.array(img)
    seg_map = np.array(seg_map)
    if kind == "body":
        mask = (seg_map != 0).astype(np.uint8)
        img = Image.fromarray(img * mask)
    elif kind == "cloth":
        mask = (seg_map == 4).astype(np.uint8)
        img = Image.fromarray(img * mask)
    return img

def resize(img, size, keep_ratio=True):
    if keep_ratio:
        w, h = img.size
        ratio = size[0] / h
        img = img.resize((int(w*ratio),int(h*ratio)))
        if int(w*ratio) > size[1]:
            diff = int(w*ratio) - size[1]
            new_img = np.array(img)[:,diff//2:diff//2+size[1],:]
        else:
            diff = size[1] - int(w*ratio)
            new_img = np.zeros((*size,3), dtype=np.uint8)
            new_img[:,diff//2:diff//2+int(w*ratio),:] = np.array(img)
        new_img = Image.fromarray(new_img)
    else:
        new_img = img.resize(size[::-1])
    return new_img


def coco_keypoint_mapping(key_pts):
    mapping_dict = {0:0, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9, 8:12, 9:14, 10:16, 11:11, 12:13, 13:15, 14:2, 15:1, 16:4, 17:3}
    new_key_pts = np.zeros((18,2))
    for i in range(len(key_pts)+1):
        if i == 1:
            new_key_pts[i] = (key_pts[5] + key_pts[6]) / 2.0
        else:
            new_key_pts[i] = key_pts[mapping_dict[i]]
    return new_key_pts


def create_mask(body_img, key_pts, seg_map, category):
    seg_map = seg_map.cpu()
    body_img = np.array(body_img)
    parse_array = np.array(seg_map)

    head = (parse_array == 1).astype(np.float32) + \
           (parse_array == 2).astype(np.float32) + \
           (parse_array == 3).astype(np.float32) + \
           (parse_array == 11).astype(np.float32)
    
    arms = (parse_array == 14).astype(np.float32) + \
           (parse_array == 15).astype(np.float32)

    legs = (parse_array == 12).astype(np.float32) + \
           (parse_array == 13).astype(np.float32)

    upper_cloth = (parse_array == 4).astype(np.float32)

    lower_cloth = (parse_array == 5).astype(np.float32) + \
                  (parse_array == 6).astype(np.float32)
    
    dress = (parse_array == 7).astype(np.float32)

    shoes = (parse_array == 9).astype(np.float32) + \
            (parse_array == 10).astype(np.float32)
    
    background = (parse_array == 0).astype(np.float32)
    
    others = (parse_array == 16).astype(np.float32) + \
             (parse_array == 17).astype(np.float32)

    if category == "upper_body":
        parse_mask = upper_cloth + arms
        fixed = head + legs + lower_cloth + shoes + others
    elif category == "lower_body":
        parse_mask = lower_cloth + legs
        fixed = head + arms + upper_cloth + shoes + others
    elif category == "dresses":
        parse_mask = upper_cloth + lower_cloth + dress + arms + legs
        fixed = head + shoes + others

    shoulder_right = (key_pts[2][0], key_pts[2][1])
    shoulder_left = (key_pts[5][0], key_pts[5][1])
    elbow_right = (key_pts[3][0], key_pts[3][1])
    elbow_left = (key_pts[6][0], key_pts[6][1])
    wrist_right = (key_pts[4][0], key_pts[4][1])
    wrist_left = (key_pts[7][0], key_pts[7][1])

    im_arms = Image.new('L', (384,512))
    arms_draw = ImageDraw.Draw(im_arms)

    ARM_LINE_WIDTH = 30
    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
        if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
            arms_draw.line(
                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        else:
            arms_draw.line(np.concatenate(
                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
    elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
        if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
            arms_draw.line(
                np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
        else:
            arms_draw.line(np.concatenate(
                (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
    else:
        arms_draw.line(np.concatenate(
            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')

    hands = np.logical_and(np.logical_not(im_arms), arms)
    parse_mask *= np.logical_not(hands)
    
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    parse_mask *= np.logical_not(np.logical_or(fixed, hands))
    parse_mask = parse_mask[np.newaxis, :, :]
		
    inpaint_mask = torch.from_numpy(parse_mask)
    im_mask = (torch.from_numpy(body_img).permute(2,0,1) / 127.5) - 1
    im_mask = im_mask * (1-inpaint_mask)
    
    return inpaint_mask.unsqueeze(0), im_mask.unsqueeze(0)


def keypoint_to_heatmap(key_pts, size):
    point_num = len(key_pts)
    map_h, map_w = size

    heatmaps = []
    for idx in range(point_num):
        if np.any(key_pts[idx] > 0):
            x, y = key_pts[idx][0], key_pts[idx][1] 
            xy_grid = np.mgrid[:map_w, :map_h].transpose(2,1,0)
            heatmap = np.exp(-np.sum((xy_grid - (x, y)) ** 2, axis=-1) / 9 ** 2)
            heatmap /= (heatmap.max() + np.finfo('float32').eps)
        else:
            heatmap = np.zeros((map_h, map_w))
        heatmap = torch.Tensor(heatmap)
        heatmaps.append(heatmap)

    pose_map = torch.stack(heatmaps)
    return pose_map.unsqueeze(0)


