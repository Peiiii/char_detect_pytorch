
import numpy as np
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import cv2

class PILImageRenderer:

    @classmethod
    def np2pil(cls, img):
        return Image.fromarray(np.uint8(img))
    @classmethod
    def pil2np(cls, img):
        return np.array(img).astype(np.float32)
    @classmethod
    def medianBlur(cls,img,ksize):
        img=cls.pil2np(img)
        img=cv2.medianBlur(img,ksize)
        img=cls.np2pil(img)
        return img
    @classmethod
    def blur(cls,img,ksize):
        img=cls.pil2np(img)
        img=cv2.blur(img,ksize)
        img=cls.np2pil(img)
        return img
    @classmethod
    def hsv_transform(cls,img, hue_delta, sat_mult, val_mult):
        img=cls.pil2np(img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        img=cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cls.np2pil(img)

    @classmethod
    def random_hsv_transform(cls,img, hue_vari, sat_vari, val_vari):
        hue_delta = np.random.randint(-hue_vari, hue_vari)
        sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
        val_mult = 1 + np.random.uniform(-val_vari, val_vari)
        return cls.hsv_transform(img, hue_delta, sat_mult, val_mult)
    @classmethod
    def gamma_transform(cls,img, gamma):
        img=cls.pil2np(img)
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        img=cv2.LUT(img, gamma_table)
        img=cls.np2pil(img)
        return img

    @classmethod
    def random_gamma_transform(cls,img, gamma_vari):
        log_gamma_vari = np.log(gamma_vari)
        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
        gamma = np.exp(alpha)
        return cls.gamma_transform(img, gamma)

class NpImageRenderer:

    @classmethod
    def add_gasuss_noise(cls,image, mean=0, var=0.001):
        '''
            添加高斯噪声
            mean : 均值
            var : 方差
        '''
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        return out

    @classmethod
    def add_haze(cls,image, t=0.6, A=1):
        '''
            添加雾霾
            t : 透视率 0~1
            A : 大气光照
        '''
        out = image * t + A * 255 * (1 - t)
        return out

    @classmethod
    def ajust_image(cls,image, cont=1, bright=0):
        '''
            调整对比度与亮度
            cont : 对比度，调节对比度应该与亮度同时调节
            bright : 亮度
        '''
        out = np.uint8(np.clip((cont * image + bright), 0, 255))
        # tmp = np.hstack((img, res))  # 两张图片横向合并（便于对比显示）
        return out

    @classmethod
    def ajust_image_hsv(cls,image, h=1, s=1, v=0.8):
        '''
            调整HSV通道，调整V通道以调整亮度
            各通道系数
        '''
        HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV)
        H2 = np.uint8(H * h)
        S2 = np.uint8(S * s)
        V2 = np.uint8(V * v)
        hsv_image = cv2.merge([H2, S2, V2])
        out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return out

    @classmethod
    def ajust_jpg_quality(cls,image, q=100, save_path=None):
        '''
            调整图像JPG压缩失真程度
            q : 压缩质量 0~100
        '''
        if save_path is None:
            cv2.imwrite("jpg_tmp.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            out = cv2.imread('jpg_tmp.jpg')
            return out
        else:
            cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), q])

    @classmethod
    def add_gasuss_blur(cls,image, kernel_size=(3, 3), sigma=0.1):
        '''
            添加高斯模糊
            kernel_size : 模糊核大小
            sigma : 标准差
        '''
        out = cv2.GaussianBlur(image, kernel_size, sigma)
        return out

    def test_methods(cls):
        img = cv2.imread('test.jpg')
        out = cls.add_haze(img)
        cv2.imwrite("add_haze.jpg", out)
        out = cls.add_gasuss_noise(img)
        cv2.imwrite("add_gasuss_noise.jpg", out)
        out = cls.add_gasuss_blur(img)
        cv2.imwrite("add_gasuss_blur.jpg", out)
        out = cls.ajust_image(img)
        cv2.imwrite("ajust_image.jpg", out)
        out = cls.ajust_image_hsv(img)
        cv2.imwrite("ajust_image_hsv.jpg", out)
        cls.ajust_jpg_quality(img, save_path='ajust_jpg_quality.jpg')

