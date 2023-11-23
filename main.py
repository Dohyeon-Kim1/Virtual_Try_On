import argparse
import numpy as np
import cv2
from PIL import Image

from models.load_pretrained import FashionPoseEstimation
from functions.transform import cloth_transform

def parse_argument():
    parser = argparse.ArgumentParser(description="Virtual Try On")
    parser.add_argument("--kind", type=str, help="the kind of cloth")
    parser.add_argument("--img1", type=str, help="the path of image 1")
    parser.add_argument("--img2", type=str, help="the path of image 2")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    model = FashionPoseEstimation(kind=args.kind)
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)

    pred1 = model.predict(img1)[0]
    pred2 = model.predict(img2)[0]

    transformed_img = cloth_transform(img2, pred2, pred1, kind="short-sleeved-shirt")
    cv2.imwrite("images/transformed_image.png", transformed_img)