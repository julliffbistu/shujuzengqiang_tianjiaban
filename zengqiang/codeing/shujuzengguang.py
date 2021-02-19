import base64
import json
from labelme import utils
import cv2 as cv
import sys
import numpy as np
import random
import re


class DataAugment(object):
    def __init__(self, image_id=1):
        self.add_saltNoise = True
        self.gaussianBlur = True
        self.changeExposure = True
        self.id = image_id
        img = cv.imread(str(self.id)+'.jpg')
        try:
            img.shape
        except:
            print('No Such image!---'+str(id)+'.png')
            sys.exit(0)
        self.src = img
        dst1 = cv.flip(img, 0, dst=None)
        dst2 = cv.flip(img, 1, dst=None)
        dst3 = cv.flip(img, -1, dst=None)
        self.flip_x = dst1
        self.flip_y = dst2
        self.flip_x_y = dst3
        cv.imwrite(str(self.id)+'flipx'+'.png', self.flip_x)
        cv.imwrite(str(self.id)+'flipy'+'.png', self.flip_y)
        cv.imwrite(str(self.id)+'flipxy'+'.png', self.flip_x_y)

    def gaussian_blur_fun(self):
        if self.gaussianBlur:
            dst1 = cv.GaussianBlur(self.src, (5, 5), 0)
            dst2 = cv.GaussianBlur(self.flip_x, (5, 5), 0)
            dst3 = cv.GaussianBlur(self.flip_y, (5, 5), 0)
            dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
            cv.imwrite(str(self.id)+'Gaussian'+'.png', dst1)
            cv.imwrite(str(self.id)+'flipx'+'Gaussian'+'.png', dst2)
            cv.imwrite(str(self.id)+'flipy'+'Gaussian'+'.png', dst3)
            cv.imwrite(str(self.id)+'flipxy'+'Gaussian'+'.png', dst4)

    def change_exposure_fun(self):
        if self.changeExposure:
            # contrast
            reduce = 0.5
            increase = 1.4
            # brightness
            g = 10
            h, w, ch = self.src.shape
            add = np.zeros([h, w, ch], self.src.dtype)
            dst1 = cv.addWeighted(self.src, reduce, add, 1-reduce, g)
            dst2 = cv.addWeighted(self.src, increase, add, 1-increase, g)
            dst3 = cv.addWeighted(self.flip_x, reduce, add, 1 - reduce, g)
            dst4 = cv.addWeighted(self.flip_x, increase, add, 1 - increase, g)
            dst5 = cv.addWeighted(self.flip_y, reduce, add, 1 - reduce, g)
            dst6 = cv.addWeighted(self.flip_y, increase, add, 1 - increase, g)
            dst7 = cv.addWeighted(self.flip_x_y, reduce, add, 1 - reduce, g)
            dst8 = cv.addWeighted(self.flip_x_y, increase, add, 1 - increase, g)
            cv.imwrite(str(self.id)+'ReduceEp'+'.png', dst1)
            cv.imwrite(str(self.id)+'flipx'+'ReduceEp'+'.png', dst3)
            cv.imwrite(str(self.id)+'flipy'+'ReduceEp'+'.png', dst5)
            cv.imwrite(str(self.id)+'flipxy'+'ReduceEp'+'.png', dst7)
            cv.imwrite(str(self.id)+'IncreaseEp'+'.png', dst2)
            cv.imwrite(str(self.id)+'flipx'+'IncreaseEp'+'.png', dst4)
            cv.imwrite(str(self.id)+'flipy'+'IncreaseEp'+'.png', dst6)
            cv.imwrite(str(self.id)+'flipxy'+'IncreaseEp'+'.png', dst8)

    def add_salt_noise(self):
        if self.add_saltNoise:
            percentage = 0.005
            dst1 = self.src
            dst2 = self.flip_x
            dst3 = self.flip_y
            dst4 = self.flip_x_y
            num = int(percentage * self.src.shape[0] * self.src.shape[1])
            for i in range(num):
                rand_x = random.randint(0, self.src.shape[0] - 1)
                rand_y = random.randint(0, self.src.shape[1] - 1)
                if random.randint(0, 1) == 0:
                    dst1[rand_x, rand_y] = 0
                    dst2[rand_x, rand_y] = 0
                    dst3[rand_x, rand_y] = 0
                    dst4[rand_x, rand_y] = 0
                else:
                    dst1[rand_x, rand_y] = 255
                    dst2[rand_x, rand_y] = 255
                    dst3[rand_x, rand_y] = 255
                    dst4[rand_x, rand_y] = 255
            cv.imwrite(str(self.id)+'Salt'+'.png', dst1)
            cv.imwrite(str(self.id)+'flipx'+'Salt'+'.png', dst2)
            cv.imwrite(str(self.id)+'flipy'+'Salt'+'.png', dst3)
            cv.imwrite(str(self.id)+'flipxy'+'Salt'+'.png', dst4)

    def json_generation(self):
        image_names = [str(self.id)+'flipx', str(self.id)+'flipy', str(self.id)+'flipxy']
        if self.gaussianBlur:
            image_names.append(str(self.id)+'Gaussian')
            image_names.append(str(self.id)+'flipx'+'Gaussian')
            image_names.append(str(self.id)+'flipy' + 'Gaussian')
            image_names.append(str(self.id)+'flipxy'+'Gaussian')
        if self.changeExposure:
            image_names.append(str(self.id)+'ReduceEp')
            image_names.append(str(self.id)+'flipx'+'ReduceEp')
            image_names.append(str(self.id)+'flipy'+'ReduceEp')
            image_names.append(str(self.id)+'flipxy'+'ReduceEp')
            image_names.append(str(self.id)+'IncreaseEp')
            image_names.append(str(self.id)+'flipx'+'IncreaseEp')
            image_names.append(str(self.id)+'flipy'+'IncreaseEp')
            image_names.append(str(self.id)+'flipxy'+'IncreaseEp')
        if self.add_saltNoise:
            image_names.append(str(self.id)+'Salt')
            image_names.append(str(self.id)+'flipx' + 'Salt')
            image_names.append(str(self.id)+'flipy' + 'Salt')
            image_names.append(str(self.id)+'flipxy' + 'Salt')
        for image_name in image_names:
            with open(image_name+".png", "rb")as b64:
                base64_data_original = str(base64.b64encode(b64.read()).decode('utf-8'))
                # In pycharm:
                # match_pattern=re.compile(r'b\'(.*)\'')
                # base64_data=match_pattern.match(base64_data_original).group(1)
                # In terminal:
                base64_data = base64_data_original
            with open(str(self.id)+".json", 'r')as js:
                json_data = json.load(js)
                img = utils.img_b64_to_arr(json_data['imageData'])
                height, width = img.shape[:2]
                shapes = json_data['shapes']
                for shape in shapes:
                    points = shape['points']
                    for point in points:
                        match_pattern2 = re.compile(r'(.*)x(.*)')
                        match_pattern3 = re.compile(r'(.*)y(.*)')
                        match_pattern4 = re.compile(r'(.*)xy(.*)')
                        if match_pattern4.match(image_name):
                            point[0] = width - point[0]
                            point[1] = height - point[1]
                        elif match_pattern3.match(image_name):
                            point[0] = width - point[0]
                            point[1] = point[1]
                        elif match_pattern2.match(image_name):
                            point[0] = point[0]
                            point[1] = height - point[1]
                        else:
                            point[0] = point[0]
                            point[1] = point[1]
                json_data['imagePath'] = image_name+".png"
                json_data['imageData'] = base64_data
                json.dump(json_data, open("./"+image_name+".json", 'w'), indent=2)


if __name__ == "__main__":
    i = 1
    while(True):
       dataAugmentObject = DataAugment(i)
       dataAugmentObject.gaussian_blur_fun()
       dataAugmentObject.change_exposure_fun()
       dataAugmentObject.add_salt_noise()
       dataAugmentObject.json_generation()
       i = i + 1
    
    
    
