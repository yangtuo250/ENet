# https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2

IMG_SHAPE = (1024, 1024)
ANN_FILE = "~/Downloads/project-2-at-2021-10-29-08-02-fb95281f/result.json"
OUTPUT_DIR = "./anns"
coco = COCO(ANN_FILE)
# catIDs = coco.getCatIds()
catIDs = [0, 2]  # select 0(warping) & 2(wormhole), ignore 1(wood). 1 for warping and 2 for wormhole in mask png


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


imgIds = coco.getImgIds()  # get all image inrespect of category

filterClasses = ['warping', 'wormhole']
for imgid in imgIds:
    imgName = coco.loadImgs(imgid)[0]['file_name']
    maskName = os.path.join(OUTPUT_DIR, imgName.split('/')[-1])
    annIds = coco.getAnnIds(imgIds=imgid, catIds=[0, 2], iscrowd=None)
    anns = coco.loadAnns(annIds)
    mask = np.zeros((1024, 1024))
    for i in range(len(anns)):
        className = getClassName(anns[i]['category_id'], cats)
        pixel_value = filterClasses.index(className) + 1
        mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)
    cv2.imwrite(maskName, mask.astype(np.uint8))
