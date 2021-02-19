# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:31:23 2020

@author: lifu_
"""

import shutil
import cv2 as cv

sets=['train2019',  'val2019', 'test2019']
for image_set in sets:
    image_ids = open('./%s.txt'%(image_set)).read().strip().split()
    for image_id in image_ids:
        img = cv.imread('images/total2019/%s.jpg' % (image_id))
        json='labelme/total2019/%s.json'% (image_id)
        cv.imwrite('images/%s/%s.jpg' % (image_set,image_id), img)
        cv.imwrite('labelme/%s/%s.jpg' % (image_set,image_id), img)
        shutil.copy(json,'labelme/%s/%s.json' % (image_set,image_id))
print("完成")

