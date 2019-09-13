import random, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os, time, shutil
# import math_utils
from ImageRenderer import PILImageRenderer as PIR
from ImageRenderer import NpImageRenderer as NIR
from imgaug.gaussian_distort import GaussianDistortion
from imgaug.augmentor import NoiseAugmentor
from imgaug import augmentor

GD = GaussianDistortion(rng=np.random, distort_range=[-1, 1], dsigma_range=[-1, -1])
NA = NoiseAugmentor(rng=np.random)

import utils


def r(val):
    return int(np.random.random() * val)


class Renderer:
    def __init__(self, font_dir='data/fonts', bg_dir='data/bgs', bg_mode='RGBA', scale=1.5):
        self.font_dir = font_dir
        self.bg_dir = bg_dir
        flist = os.listdir(font_dir)
        self.font_files = [font_dir + '/' + i for i in flist]
        bg_list = os.listdir(bg_dir)
        self.bg_files = [bg_dir + '/' + f for f in bg_list]
        self.bg_mode = bg_mode
        self.canvas_mode = 'RGBA'
        self.font_size = 60
        self.canvas_size = (int(self.font_size * 1.5), int(self.font_size * 1.5))
        self.out_size = (int(self.font_size * scale), int(self.font_size * scale))
        self.bg_size = (int(self.canvas_size[0] * 2), int(self.canvas_size[1] * 2))

        self.blur_rate = 0.2

    def gen_one_img(self, char):

        scale = utils.get_random([
            [1.0, 1.1], [1.1, 1.2], [1.2, 1.3], [1.3, 1.4], [1.4, 1.5], [1.5, 1.6], [1.6, 1.7]
        ],
            [1, 1.5, 2, 2.5, 3, 2.5, 1.5]
        )

        scale = 1.3
        self.out_size = (int(self.font_size * scale), int(self.font_size * scale))

        # self.out_size=(int(self.font_size*scale),int(self.font_size*scale))

        bg = self.get_random_bg()
        bg = bg.resize(self.bg_size)

        # bg=self.apply_guassion_blur(bg,0.1)

        canvas_color = self.get_random_canvas_color()
        # canvas_color=(255,0,0)
        canvas = self.get_canvas(canvas_color)
        canvas = self.draw_char_on_canvas(canvas, char)

        # self.show(canvas)
        angel = utils.get_randn_clipped([-30, 30])
        # angel=utils.random_from_range([-20,20])
        # angel=-20
        canvas = self.apply_affine(canvas, angle=angel, max_angle=80)

        # canvas.show()
        angle = utils.get_randn_clipped([-90, 90])
        # print(angel)
        # angle=-45
        canvas = self.apply_rotate(canvas, angle=angle)

        # canvas.show()

        # crop here
        canvas = self.paste_canvas_on_bg(canvas, bg)

        canvas = self.aug_img_using_cv2(canvas, 0.8)

        # self.show(canvas)
        # crop here
        canvas = self.crop_from_bg(canvas)

        # canvas.show()
        canvas = self.aug_img_using_cv2(canvas, 0.5)
        # self.show(canvas)
        canvas = self.add_noise(canvas)
        # self.show(canvas)
        canvas = self.random_aug(canvas, 0)
        self.show(canvas)
        canvas = self.np(canvas)
        # canvas=GD.distort_img(canvas)
        # self.show(canvas)

        # canvas=augmentor.aug_after_crop(np.random,NA,canvas)
        # self.show(canvas)
        canvas = self.np2pil(canvas)
        # canvas.show()
        return canvas

    def apply_rotate(self, img, angle=45):
        img = img.rotate(angle, resample=Image.BICUBIC)
        return img

    def apply_affine(self, img, angle=20, max_angle=80):
        img = self.rot(img, angle=angle, shape=img.size, max_angle=max_angle)
        return img

    def get_random_font(self, font_size=60):
        font_size = self.font_size
        font_file = random.choice(self.font_files)
        font = ImageFont.truetype(font_file, font_size)

        return font

    def gen_bg(self):
        rint = random.randint
        r = random.random
        if r() < 0.2:
            color = rint(0, 20), rint(0, 20), rint(0, 20), rint(200, 255)
        else:
            color = rint(0, 255), rint(0, 255), rint(0, 255), 255

        img = Image.new('RGBA', self.bg_size, color=color)
        return img

    def get_random_bg(self):
        n = random.random()

        if n < 0.15:
            img1 = self.gen_bg()
            return img1
        elif n < 0.8:
            bg_file = random.choice(self.bg_files)
            img2 = Image.open(bg_file).convert(self.bg_mode)
            return img2

        else:
            img1 = self.gen_bg()
            bg_file = random.choice(self.bg_files)
            img2 = Image.open(bg_file).convert(self.bg_mode)
            img2 = img2.resize(self.bg_size)

            img3 = Image.alpha_composite(img1, img2)

            return img3

    def get_random_text_color(self):
        n = random.random()
        alpha = 125
        if n < 0.2:
            color = (255, 255, 255, random.randint(150, 256))
        elif n < 0.5:
            color = random.randint(200, 256), random.randint(200, 256), random.randint(200, 256), random.randint(200,
                                                                                                                 256)
        elif n < 0.6:
            color = random.randint(200, 256), random.randint(200, 256), random.randint(200, 256), random.randint(150,
                                                                                                                 256)
        elif n < 0.7:
            a = 170
            color = random.randint(a, 256), random.randint(a, 256), random.randint(a, 256), random.randint(150, 256)
        elif n < 0.9:
            a = 120
            color = random.randint(a, 256), random.randint(a, 256), random.randint(a, 256), random.randint(150, 256)
        else:
            a = 80
            color = random.randint(a, 256), random.randint(a, 256), random.randint(a, 256), random.randint(200, 256)
        return color

    def get_random_canvas_color(self):
        n = random.random()

        def rc():
            return random.randint(0, 256)

        if n < 0.1:
            color = (0, 0, 0, 0)
        elif n < 0.3:
            color = (0, 0, 0, int(utils.get_random([[0, 80], [80, 170], [170, 255]], [6, 2, 1])))
        elif n < 0.65:
            color = (255, 255, 255, int(utils.get_random([[0, 50], [50, 170], [170, 255]], [7, 2, 1])))
        elif n < 1:
            color = (rc(), rc(), rc(), int(utils.get_random([[0, 50], [50, 170], [170, 255]], [7, 2, 1])))
        # elif n<0.8:
        #     low,high=50,100
        #     color = (random.randint(low,high), random.randint(low,high), random.randint(low,high), random.randint(0, 200))
        # elif n<0.95:
        #     color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(50, 200))
        else:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(150, 200))
        return color

    def get_canvas(self, color):
        img = Image.new(self.canvas_mode, self.canvas_size, color)
        return img

    def draw_char_on_canvas(self, canvas, char):
        (canvas_width, canvas_height) = canvas.size
        draw = ImageDraw.Draw(canvas)
        font = self.get_random_font()
        (char_width, char_height) = font.getsize(char)
        color = self.get_random_text_color()
        xy = (
            (canvas_width - char_width) // 2,
            (canvas_height - char_height) // 2
        )
        draw.text(xy, text=char, fill=color, font=font)
        return canvas

    def rotate_img(self, img, max_x=20, max_y=50, max_z=50):
        img = self.pil2np(img)
        word_img = img
        box_pnts = ((0, 0), (word_img.shape[1] - 1, 0), (0, word_img.shape[0] - 1),
                    (word_img.shape[1] - 1, word_img.shape[0] - 1))
        word_img, _1, _2 = self.apply_perspective_transform(word_img, box_pnts, max_x, max_y, max_z)

        word_img = self.np2pil(word_img)
        return word_img, _1, _2

    def paste_canvas_on_bg(self, canvas, bg):
        c_width, c_height = canvas.size
        bg_width, bg_height = bg.size
        bg.alpha_composite(
            canvas,
            ((bg_width - c_width) // 2, (bg_height - c_height) // 2)
        )
        return bg

    def apply_random_blur(self, img):
        n = random.random()
        if n < self.blur_rate:
            img = img.filter(ImageFilter.GaussianBlur)
        return img

    def apply_guassion_blur(self, img, ratio=0.7):
        n = random.random()
        if n < ratio:
            img = img.filter(ImageFilter.GaussianBlur)
        return img

    def crop_from_bg(self, bg):

        img = self.crop_center(bg, self.out_size, random_offset=True, alpha=0.12)
        return img

    def aug_img_using_cv2(self, img, ratio=0.8):

        r = random.random
        if r() > ratio:
            return img
        img = self.pil2np(img)
        if r() < 0.1:
            img = NIR.add_gasuss_noise(img)
            pass
        if r() < 0.2:
            img = NIR.add_haze(img)
        if r() < 0.2:
            img = NIR.ajust_image(img)
        # if r()<0.2:
        #     img=NIR.ajust_image_hsv(img)
        if r() < 0.2:
            img = NIR.ajust_jpg_quality(img)

        img = self.np2pil(img)
        return img

    def np(self, img):
        if isinstance(img, Image.Image):
            img = self.pil2np(img)
        return img

    def pil(self, img):
        if not isinstance(img, Image.Image):
            img = self.np2pil(img)
        return img

    def show(self, img):
        if isinstance(img, Image.Image):
            img.show()
        else:
            Image.fromarray(np.uint8(img)).show()

    def random_aug(self, img, p=0.2):
        img = self.apply_darker(img)
        return img

    def apply_emboss(self, word_img):
        self.create_kernals()
        word_img = np.array(word_img).astype(np.float32)
        np_img = cv2.filter2D(word_img, -1, self.emboss_kernal)
        pil_img = Image.fromarray(np.uint8(np_img))
        return pil_img

    def create_kernals(self):
        self.emboss_kernal = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])

        self.sharp_kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])

    def apply_darker(self, img):
        # scale=utils.random_from_range([0.1,1])
        scale = utils.get_random(
            [[0.1, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1]],
            [0.5, 1, 2, 3, 5]
        )
        img = self.pil2np(img)
        img = img * scale
        img = self.np2pil(img)
        return img

    def add_noise(self, img):
        r = random.random
        if r() < 0.3:
            img = self.apply_poisson_noise(img)
        if r() < 0.3:
            img = self.apply_sp_noise(img)
        return img

    def np2pil(self, img):
        return Image.fromarray(np.uint8(img))

    def pil2np(self, img):
        return np.array(img).astype(np.float32)

    def apply_sp_noise(self, img):
        """
        Salt and pepper noise. Replaces random pixels with 0 or 255.
        """
        img = np.array(img).astype(np.float32)
        row, col, chn = img.shape
        s_vs_p = 0.5
        amount = np.random.uniform(0.004, 0.01)
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 255.

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        out = Image.fromarray(np.uint8(out))
        return out

    def apply_poisson_noise(self, img):
        """
        Poisson-distributed noise generated from the data.
        """
        img = self.pil2np(img)
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))

        if vals < 0:
            return img

        noisy = np.random.poisson(img * vals) / float(vals)
        img = self.np2pil(noisy)
        return img

    def crop_center(self, bg, box_size, random_offset=False, alpha=0.3):
        box_width, box_height = box_size
        bg_width, bg_height = bg.size
        pad_width = (bg_width - box_width) // 2
        pad_height = (bg_height - box_height) // 2

        if random_offset:
            offset_x = int((random.random() * 2 - 1) * pad_width * alpha)
            offset_y = int((random.random() * 2 - 1) * pad_height * alpha)
            pad_width += offset_x
            pad_height += offset_y

        box_pnts = (
            pad_width,
            pad_height,
            pad_width + box_width,
            pad_height + box_height
        )
        img = bg.crop(box_pnts)
        return img

    def rot(self, img, angle, shape, max_angle):
        """
            添加放射畸变
            img 输入图像
            factor 畸变的参数
            size 为图片的目标尺寸
        """
        if angle == 0:
            return img
        img = self.pil2np(img)
        size_o = [shape[1], shape[0]]
        size = (shape[1] + int(shape[0] * np.cos((float(max_angle) / 180) * 3.14)), shape[0])
        interval = abs(int(np.sin((float(angle) / 180) * 3.14) * shape[0]));
        pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
        if (angle > 0):
            pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
        else:
            pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])
        M = cv2.getPerspectiveTransform(pts1, pts2);
        dst = cv2.warpPerspective(img, M, size);
        dst = self.np2pil(dst)
        return dst

    def rotRandrom(self, img, factor, size):
        """
        添加透视畸变
        """

        img = self.pil2np(img)
        shape = size;
        pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
        pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor), r(factor)],
                           [shape[1] - r(factor), shape[0] - r(factor)]])
        M = cv2.getPerspectiveTransform(pts1, pts2);
        dst = cv2.warpPerspective(img, M, size);
        dst = self.np2pil(dst)
        return dst

    def apply_perspective_transform(self, img, text_box_pnts, max_x, max_y, max_z, gpu=False):
        """
        Apply perspective transform on image
        :param img: origin numpy image
        :param text_box_pnts: four corner points of text
        :param x: max rotate angle around X-axis
        :param y: max rotate angle around Y-axis
        :param z: max rotate angle around Z-axis
        :return:
            dst_img:
            dst_img_pnts: points of whole word image after apply perspective transform
            dst_text_pnts: points of text after apply perspective transform
        """

        x = math_utils.cliped_rand_norm(0, max_x)
        y = math_utils.cliped_rand_norm(0, max_y)
        z = math_utils.cliped_rand_norm(0, max_z)

        # print("x: %f, y: %f, z: %f" % (x, y, z))

        transformer = math_utils.PerspectiveTransform(x, y, z, scale=1.0, fovy=50)

        dst_img, M33, dst_img_pnts = transformer.transform_image(img, gpu)
        dst_text_pnts = transformer.transform_pnts(text_box_pnts, M33)

        return dst_img, dst_img_pnts, dst_text_pnts

    # def resize(self,img,size):
    #     img=cv2.resize(img,size)
    #     return img


def gen_bg():
    pass


def get_font(font_dir='data/fonts'):
    flist = os.listdir(font_dir)
    fn = random.choice(flist)
    fpath = font_dir + '/' + fn
    return fpath


def test1():
    from PIL import Image, ImageDraw, ImageFont
    # get an image
    # base = Image.open('data/bgs/1.png').convert('RGBA')
    # base = Image.open('data/bgs/1.jpg')
    # base.show()

    base = Image.open('data/bgs/1.jpg').convert('RGBA')
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', (200, 200), (0, 0, 0, 100))

    # get a font
    fnt = ImageFont.truetype('data/fonts/stzhongs.ttf', 40)
    fnt2 = ImageFont.truetype('data/fonts/stkaiti.ttf', 40)
    fnt3 = ImageFont.truetype('data/fonts/Deng.ttf', 40)

    # get a drawing context
    d = ImageDraw.Draw(txt)

    # draw text, half opacity
    d.text((10, 10), "爱", font=fnt, fill='white')
    d.text((10, 60), "觉", font=fnt2, fill='white')
    d.text((10, 120), "翻", font=fnt3, fill='white')
    # draw text, full opacity
    # d.text((10, 60), "南柯一梦", font=fnt, fill=(255, 255, 255, 255))

    out = base.alpha_composite(txt, (
        (base.width - txt.width) // 2, (base.height - txt.height) // 2
    ))
    # out.show()
    # base.paste(txt)
    base.show()

    pass


def test2():
    t_start = time.time()
    renderer = Renderer()
    charset = ['英']
    for char in charset:
        for i in range(10):
            img = renderer.gen_one_img(char).convert('RGB')
            # img.show()
    t_end = time.time()

    print('time consumed: %s' % (t_end - t_start))
    pass


# 穆


if __name__ == "__main__":
    test2()
