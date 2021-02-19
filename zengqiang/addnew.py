import base64
import json
from labelme import utils
import cv2 as cv
import sys
import numpy as np
import random
import re
from PIL import Image
from PIL import ImageEnhance

class DataAugment(object):
    def __init__(self, image_id):
        self.add_saltNoise = True
        self.gaussianBlur = True
        self.changeExposure = True
        self.chgsharpnessfun = True
        self.chgcolorfun = True
        self.chgcontrastfun = True
        self.id = image_id
        img = cv.imread(str(self.id)+'.jpg')
        try:
            img.shape
        except:
            print('No Such image!---'+str(id)+'.jpg')
            sys.exit(0)
        self.src = img
        dst1 = cv.flip(img, 0, dst=None)
        dst2 = cv.flip(img, 1, dst=None)
        dst3 = cv.flip(img, -1, dst=None)
        self.flip_x = dst1
        self.flip_y = dst2
        self.flip_x_y = dst3
        cv.imwrite(str(self.id)+'_flip_x'+'.jpg', self.flip_x)
        cv.imwrite(str(self.id)+'_flip_y'+'.jpg', self.flip_y)
        cv.imwrite(str(self.id)+'_flip_x_y'+'.jpg', self.flip_x_y)

    def gaussian_blur_fun(self):
        if self.gaussianBlur:
            dst1 = cv.GaussianBlur(self.src, (5, 5), 0)
            dst2 = cv.GaussianBlur(self.flip_x, (5, 5), 0)
            dst3 = cv.GaussianBlur(self.flip_y, (5, 5), 0)
            dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
            cv.imwrite(str(self.id)+'_Gaussian'+'.jpg', dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_Gaussian'+'.jpg', dst2)
            cv.imwrite(str(self.id)+'_flip_y'+'_Gaussian'+'.jpg', dst3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Gaussian'+'.jpg', dst4)

    def chg_sharpness_fun(self):
        # 锐度，增强因子为1.0是原始图片
        # 锐度增强 3
        # 锐度减弱 0.8
        if self.chgsharpnessfun:
            image1 = Image.fromarray(cv.cvtColor(self.src,cv.COLOR_BGR2RGB))
            image2 = Image.fromarray(cv.cvtColor(self.flip_x,cv.COLOR_BGR2RGB))
            image3 = Image.fromarray(cv.cvtColor(self.flip_y,cv.COLOR_BGR2RGB))
            image4 = Image.fromarray(cv.cvtColor(self.flip_x_y,cv.COLOR_BGR2RGB))

            enh_sha1 = ImageEnhance.Sharpness(image1)
            enh_sha2 = ImageEnhance.Sharpness(image2)
            enh_sha3 = ImageEnhance.Sharpness(image3)
            enh_sha4 = ImageEnhance.Sharpness(image4)
            sharpness = 3.0 * random.randrange(1000, 1300)*0.001
            dst1 = enh_sha1.enhance(sharpness)
            sharpness = 3.0 * random.randrange(1000, 1300)*0.001
            dst2 = enh_sha2.enhance(sharpness)
            sharpness = 3.0 * random.randrange(1000, 1300)*0.001
            dst3 = enh_sha3.enhance(sharpness)
            sharpness = 3.0 * random.randrange(1000, 1300)*0.001
            dst4 = enh_sha4.enhance(sharpness)
            img1 = cv.cvtColor(np.asarray(dst1),cv.COLOR_RGB2BGR)
            img2 = cv.cvtColor(np.asarray(dst2),cv.COLOR_RGB2BGR)
            img3 = cv.cvtColor(np.asarray(dst3),cv.COLOR_RGB2BGR)
            img4 = cv.cvtColor(np.asarray(dst4),cv.COLOR_RGB2BGR)

            cv.imwrite(str(self.id)+'_sharpness'+'.jpg', img1)
            cv.imwrite(str(self.id)+'_flip_x'+'_sharpness'+'.jpg', img2)
            cv.imwrite(str(self.id)+'_flip_y'+'_sharpness'+'.jpg', img3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_sharpness'+'.jpg', img4)

    def chg_color_fun(self):
        # 色度,增强因子为1.0是原始图像
        # 色度增强 1.5
        # 色度减弱 0.8
        if self.chgcolorfun:
            image1 = Image.fromarray(cv.cvtColor(self.src,cv.COLOR_BGR2RGB))
            image2 = Image.fromarray(cv.cvtColor(self.flip_x,cv.COLOR_BGR2RGB))
            image3 = Image.fromarray(cv.cvtColor(self.flip_y,cv.COLOR_BGR2RGB))
            image4 = Image.fromarray(cv.cvtColor(self.flip_x_y,cv.COLOR_BGR2RGB))

            enh_sha1 = ImageEnhance.Color(image1)
            enh_sha2 = ImageEnhance.Color(image2)
            enh_sha3 = ImageEnhance.Color(image3)
            enh_sha4 = ImageEnhance.Color(image4)
            color = 1.5 * random.randrange(1000, 1300)*0.001
            dst1 = enh_sha1.enhance(color)
            color = 1.5 * random.randrange(1000, 1300)*0.001
            dst2 = enh_sha2.enhance(color)
            color = 1.5 * random.randrange(1000, 1300)*0.001
            dst3 = enh_sha3.enhance(color)
            color = 1.5 * random.randrange(1000, 1300)*0.001
            dst4 = enh_sha4.enhance(color)
            img1 = cv.cvtColor(np.asarray(dst1),cv.COLOR_RGB2BGR)
            img2 = cv.cvtColor(np.asarray(dst2),cv.COLOR_RGB2BGR)
            img3 = cv.cvtColor(np.asarray(dst3),cv.COLOR_RGB2BGR)
            img4 = cv.cvtColor(np.asarray(dst4),cv.COLOR_RGB2BGR)

            cv.imwrite(str(self.id)+'_color'+'.jpg', img1)
            cv.imwrite(str(self.id)+'_flip_x'+'_color'+'.jpg', img2)
            cv.imwrite(str(self.id)+'_flip_y'+'_color'+'.jpg', img3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_color'+'.jpg', img4)

    def chg_contrast_fun(self):
        # 对比度，增强因子为1.0是原始图片
        # 对比度增强 1.5
        # 对比度减弱 0.8
        if self.chgcontrastfun:
            image1 = Image.fromarray(cv.cvtColor(self.src,cv.COLOR_BGR2RGB))
            image2 = Image.fromarray(cv.cvtColor(self.flip_x,cv.COLOR_BGR2RGB))
            image3 = Image.fromarray(cv.cvtColor(self.flip_y,cv.COLOR_BGR2RGB))
            image4 = Image.fromarray(cv.cvtColor(self.flip_x_y,cv.COLOR_BGR2RGB))

            enh_sha1 = ImageEnhance.Contrast(image1)
            enh_sha2 = ImageEnhance.Contrast(image2)
            enh_sha3 = ImageEnhance.Contrast(image3)
            enh_sha4 = ImageEnhance.Contrast(image4)
            contrast = 1.5 * random.randrange(1000, 1300)*0.001
            dst1 = enh_sha1.enhance(contrast)
            contrast = 1.5 * random.randrange(1000, 1300)*0.001
            dst2 = enh_sha2.enhance(contrast)
            contrast = 1.5 * random.randrange(1000, 1300)*0.001
            dst3 = enh_sha3.enhance(contrast)
            contrast = 1.5 * random.randrange(1000, 1300)*0.001
            dst4 = enh_sha4.enhance(contrast)
            img1 = cv.cvtColor(np.asarray(dst1),cv.COLOR_RGB2BGR)
            img2 = cv.cvtColor(np.asarray(dst2),cv.COLOR_RGB2BGR)
            img3 = cv.cvtColor(np.asarray(dst3),cv.COLOR_RGB2BGR)
            img4 = cv.cvtColor(np.asarray(dst4),cv.COLOR_RGB2BGR)

            cv.imwrite(str(self.id)+'_contrast'+'.jpg', img1)
            cv.imwrite(str(self.id)+'_flip_x'+'_contrast'+'.jpg', img2)
            cv.imwrite(str(self.id)+'_flip_y'+'_contrast'+'.jpg', img3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_contrast'+'.jpg', img4)

    def change_exposure_fun(self):
        if self.changeExposure:
            # contrast
            reduce = 0.7 * random.randrange(1000, 1300)*0.001
            increase = 1.3 * random.randrange(770, 1000)*0.001
            # brightness
            g = 7 * random.randrange(700, 1500)*0.001
            print("bei shu", reduce, increase, g)
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
            cv.imwrite(str(self.id)+'_ReduceEp'+'.jpg', dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_ReduceEp'+'.jpg', dst3)
            cv.imwrite(str(self.id)+'_flip_y'+'_ReduceEp'+'.jpg', dst5)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_ReduceEp'+'.jpg', dst7)
            cv.imwrite(str(self.id)+'_IncreaseEp'+'.jpg', dst2)
            cv.imwrite(str(self.id)+'_flip_x'+'_IncreaseEp'+'.jpg', dst4)
            cv.imwrite(str(self.id)+'_flip_y'+'_IncreaseEp'+'.jpg', dst6)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_IncreaseEp'+'.jpg', dst8)

    def add_salt_noise(self):
        if self.add_saltNoise:
            percentage = 0.0005
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
            cv.imwrite(str(self.id)+'_Salt'+'.jpg', dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_Salt'+'.jpg', dst2)
            cv.imwrite(str(self.id)+'_flip_y'+'_Salt'+'.jpg', dst3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Salt'+'.jpg', dst4)

    def json_generation(self):
        image_names = [str(self.id)+'_flip_x', str(self.id)+'_flip_y', str(self.id)+'_flip_x_y']
        if self.gaussianBlur:
            image_names.append(str(self.id)+'_Gaussian')
            image_names.append(str(self.id)+'_flip_x'+'_Gaussian')
            image_names.append(str(self.id)+'_flip_y' + '_Gaussian')
            image_names.append(str(self.id)+'_flip_x_y'+'_Gaussian')
        if self.changeExposure:
            image_names.append(str(self.id)+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_y'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_IncreaseEp')
        #if self.add_saltNoise:
            #image_names.append(str(self.id)+'_Salt')
            #image_names.append(str(self.id)+'_flip_x' + '_Salt')
            #image_names.append(str(self.id)+'_flip_y' + '_Salt')
            #image_names.append(str(self.id)+'_flip_x_y' + '_Salt')
        if self.chgsharpnessfun:
            image_names.append(str(self.id)+'_sharpness')
            image_names.append(str(self.id)+'_flip_x'+'_sharpness')
            image_names.append(str(self.id)+'_flip_y'+'_sharpness')
            image_names.append(str(self.id)+'_flip_x_y'+'_sharpness')

        if self.chgcolorfun:
            image_names.append(str(self.id)+'_color')
            image_names.append(str(self.id)+'_flip_x'+'_color')
            image_names.append(str(self.id)+'_flip_y'+'_color')
            image_names.append(str(self.id)+'_flip_x_y'+'_color')

        if self.chgcontrastfun:
            image_names.append(str(self.id)+'_contrast')
            image_names.append(str(self.id)+'_flip_x'+'_contrast')
            image_names.append(str(self.id)+'_flip_y'+'_contrast')
            image_names.append(str(self.id)+'_flip_x_y'+'_contrast')

        for image_name in image_names:
            with open(image_name+".jpg", "rb")as b64:
                #base64_data_original = str(base64.b64encode(b64.read()))
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
                        match_pattern2 = re.compile(r'(.*)_x(.*)')
                        match_pattern3 = re.compile(r'(.*)_y(.*)')
                        match_pattern4 = re.compile(r'(.*)_x_y(.*)')
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
                json_data['imagePath'] = image_name+".jpg"
                json_data['imageData'] = base64_data
                json.dump(json_data, open("./"+image_name+".json", 'w'), indent=2)


if __name__ == "__main__":
    i = 1
    while(True):
        dataAugmentObject = DataAugment('erpi'+ str(i))
        dataAugmentObject.gaussian_blur_fun()
        dataAugmentObject.change_exposure_fun()
        dataAugmentObject.chg_sharpness_fun()
        dataAugmentObject.chg_color_fun()
        dataAugmentObject.chg_contrast_fun()
        #dataAugmentObject.add_salt_noise()
        dataAugmentObject.json_generation()
        i = i + 1
        
