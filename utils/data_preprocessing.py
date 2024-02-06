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
        ratio = size[1] / h
        img = img.resize(int(w*ratio), int(h*ratio))
        if int(w*ratio) > size[0]:
            diff = int(w*ratio) - size[0]
            new_img = Image.fromarray(np.array(img)[:,diff//2:diff//2+size[0],:])
        else:
            diff = size[0] - int(w*ratio)
            new_img = np.zeros_like(size, dtype=np.uint8)
            new_img = Image.fromarray(new_img[:,diff//2:diff//2+size[0],:])
    else:
        new_img = img.resize(size)
    return new_img


def coco_keypoint_mapping(key_pts):
    mapping_dict = {0:0, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9, 8:12, 9:14, 10:16, 11:11, 12:13, 13:15, 14:2, 15:1, 16:4, 17:3}
    new_key_pts = np.zeros_like(key_pts)
    for i in range(len(key_pts)):
        if i == 1:
            new_key_pts[i] = (key_pts[5] + key_pts[6]) / 2.0
        else:
            new_key_pts[i] = key_pts[mapping_dict[i]]
    return new_key_pts


def create_mask(body_img, key_pts, seg_map):
    seg_map = seg_map.cpu()
    body_img = np.array(body_img)
    parse_array = np.array(seg_map)

    parse_head = (parse_array == 1).astype(np.float32) + \
                    (parse_array == 2).astype(np.float32) + \
                    (parse_array == 3).astype(np.float32) + \
                    (parse_array == 11).astype(np.float32)
    
    parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 2).astype(np.float32) + \
                        (parse_array == 3).astype(np.float32) + \
                        (parse_array == 5).astype(np.float32) + \
                        (parse_array == 6).astype(np.float32) + \
                        (parse_array == 8).astype(np.float32) + \
                        (parse_array == 9).astype(np.float32) + \
                        (parse_array == 10).astype(np.float32) + \
                        (parse_array == 12).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32) + \
                        (parse_array == 16).astype(np.float32)
    
    arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
    
    parse_cloth = parse_mask = (parse_array == 4).astype(np.float32)
    
    parser_mask_changeable = (parse_array == 0).astype(np.float32)
    parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    parse_head = torch.from_numpy(parse_head)  # [0,1]
    parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
    parse_mask = torch.from_numpy(parse_mask)  # [0,1]
    parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
    parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

    parse_mask = parse_mask.cpu().numpy()

    im_arms = Image.new('L', (384,512))
    arms_draw = ImageDraw.Draw(im_arms)
    
    shoulder_right = (key_pts[2][0], key_pts[2][1])
    shoulder_left = (key_pts[5][0], key_pts[5][1])
    elbow_right = (key_pts[3][0], key_pts[3][1])
    elbow_left = (key_pts[6][0], key_pts[6][1])
    wrist_right = (key_pts[4][0], key_pts[4][1])
    wrist_left = (key_pts[7][0], key_pts[7][1])

    ARM_LINE_WIDTH = 90
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
    parse_mask += im_arms
    parser_mask_fixed += hands
    
    # delete neck
    parse_head_2 = torch.clone(parse_head)
    parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
    parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                           np.logical_not(np.array(parse_head_2, dtype=np.uint16))))
    
    # tune the amount of dilation here
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    parse_mask_total = parse_mask_total[:, :, np.newaxis]
		
	# body_img 차원에 맞춰 계산하기위해 parse_mask_total 모양 변경-> 위 과정
    im_mask = (torch.Tensor(im_mask).permute(2,0,1) / 127.5) - 1
    im_mask = body_img * parse_mask_total
    
    inpaint_mask = 1 - parse_mask_total
    inpaint_mask = torch.Tensor(inpaint_mask).permute(2,0,1)
    
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


