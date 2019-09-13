import math

import cv2
import numpy as np
# from neupeak.utils.imgproc import _clip_normalize


def rng_between(rng, l, h):
    if isinstance(l, list) or isinstance(l, tuple):
        l = np.array(l)
        h = np.array(h)
        return l + (h - l) * rng.rand(*l.shape)
    else:
        return l + (h - l) * rng.rand()


#sorry flare fast is just an alias of flare
def flare(rng, img, x_min=0., x_max=1., y_min=0., y_max=1., r_min=0.2, r_max=0.8, light_min=(50, ), light_max=(110, )):
    ih, iw = img.shape[:2]
    x = rng_between(rng, x_min, x_max) * ih  #XXX: copied from neupack, wierd
    y = rng_between(rng, r_min, r_max) * iw
    r = rng_between(rng, r_min, r_max) * ih
    light = np.array([rng_between(rng, l, h + 1) for l, h in zip(light_min, light_max)])
    r2 = r**2
    irange = img.shape[0]
    jrange = img.shape[1]
    ivec = np.arange(0, irange)
    jvec = np.arange(0, jrange)
    _r2 = (ivec * ivec).reshape(-1,1) + (jvec * jvec) + x*x + y*y - \
          2*ivec.reshape(-1,1)*x - 2*jvec*y
    w = np.zeros((img.shape[0], img.shape[1]))
    w = np.maximum(w, 1 - _r2 / r2)
    low = np.array([c * w for c in light])
    if low.shape[0] == 1:
        low = np.tile(low, (3, 1, 1))
    low = np.rollaxis(np.rollaxis(low, 2, 0), 2, 0)
    img1 = (img * (255. - low) / 255. + low).astype('uint8')
    return img1


def shadow_fast(rng, img, mask_min, mask_max, cut=1):
    def rand_edge_pt(shape):
        v = rng.rand() * ((shape[0] + shape[1]) * 2)
        if v < shape[0]:
            return (v, 0)
        elif v < shape[0] + shape[1]:
            return (shape[0] - 1, v - shape[0])
        elif v < shape[0] * 2 + shape[1]:
            return (v - shape[0] - shape[1], shape[1] - 1)
        else:
            return (0, v - shape[0] * 2 - shape[1])

    def cut_mask(mask, p, q, disb):
        x_range = mask.shape[0]
        y_range = mask.shape[1]
        xvec = np.arange(0, x_range)
        yvec = np.arange(0, y_range)
        det = (q[0] - p[0]) * (yvec - p[1]) - \
              (xvec.reshape(-1, 1) - p[0]) * (q[1] - p[1])
        d = np.where(det < 0, det, 0)
        d = np.where(det >= 0, d, disb)
        d = np.rollaxis(np.rollaxis(np.array([d, d, d]), 2, 0), 2, 0)
        mask = mask * (disb - d) / disb + d
        return mask

    disb = rng_between(rng, mask_min, mask_max)
    mask = np.ones_like(img).astype('float32')
    for _cut in range(rng.randint(cut) + 1):
        p, q = rand_edge_pt(img.shape), rand_edge_pt(img.shape)
        mask = cut_mask(mask, p, q, disb)
    return (img.astype('float32') * mask).astype('uint8')


def contrast(rng, img, contrast_range):
    c = rng_between(rng, contrast_range[0], contrast_range[1])
    f = 259 * (c + 255.0) / (255 * (259.0 - c))
    v = cv2.addWeighted(img, f, np.zeros_like(img), 0, 128 * (1 - f))
    v = np.clip(v, 0, 255)
    return v


def blinds(rng, img, color_rng=20):
    blind_ver = rng.randint(-color_rng, color_rng, (img.shape[1], 3))
    blind_hor = rng.randint(-color_rng, color_rng, (img.shape[0], 1, 3))
    s = img.astype(np.int16)
    s += blind_ver
    s += blind_hor
    img = np.clip(s, 0, 255).astype(np.uint8)
    return img


def backgroud_noise(rng, img, color_rng=45):
    noise = rng.randint(-color_rng, color_rng, img.shape)
    return np.clip((img.astype(np.int16) + noise), 0, 255).astype(np.uint8)


def repeated_resize_artifact(rng, img, stroke, box_heatmap, coords, target_shape):
    orig_shape = img.shape[:2]

    nr_repeat = rng.randint(0, 3)

    for i in range(nr_repeat):
        x_ratio = rng.uniform(0.25, 1.2)
        y_ratio = rng.uniform(0.25, 1.2)
        shape = tuple([max(1, int(round(l * r))) for l, r in zip(orig_shape, [x_ratio, y_ratio])])
        img = cv2.resize(img, shape[::-1])

    img = cv2.resize(img, target_shape[::-1])
    stroke = [cv2.resize(s, target_shape[::-1]) for s in stroke]
    box_heatmap = [cv2.resize(s, target_shape[::-1]) for s in box_heatmap]
    x_ratio = target_shape[1] / orig_shape[1]
    y_ratio = target_shape[0] / orig_shape[0]
    coords = [[(x * x_ratio, y * y_ratio) for x, y in coord] for coord in coords]

    return img, stroke, box_heatmap, coords


def light_environment(rng, img, color_min, color_max, p=0.3):
    color = rng_between(rng, np.array(color_min), np.array(color_max))
    dst = np.zeros(img.shape, dtype="uint8")
    for i, c in enumerate(color):
        dst[:, :, i] = np.clip(img[:, :, i].astype('float32') * ((1 - p + c * p) * np.ones(img.shape[:2])), 0,
                               255).astype('uint8')
    return dst


def brightness(rng, img, brightness_range):
    brightness = rng_between(rng, brightness_range[0], brightness_range[1])
    return cv2.scaleAdd(img, brightness, np.zeros_like(img))


def gamma(rng, img, gamma_range):
    gamma = rng_between(rng, gamma_range[0], gamma_range[1])
    k = 1.0 / gamma
    img = cv2.exp(k * cv2.log(img.astype('float32') + 1e-15))
    f = math.pow(255.0, 1 - k)
    img = img * f
    img = cv2.add(img, np.zeros_like(img), dtype=0)  # clip
    return img


def motion_blur(rng, img, length_range, angle_base=0, angle_delta_max=360):
    angle = rng_between(rng, angle_base, angle_base + angle_delta_max)
    angle %= 360
    length = rng_between(rng, length_range[0], length_range[1])
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    a = int(max(list(map(abs, (dx, dy)))) * length)
    if a <= 0:
        return img
    kern = np.zeros((a, a))
    cx, cy = a // 2, a // 2
    dx, dy = list(map(int, (dx * length / 2 + cx, dy * length / 2 + cy)))
    ex, ey = list(map(int, (-dx * length / 2 + cx, -dy * length / 2 + cy)))
    cv2.line(kern, (ex, ey), (dx, dy), 1.0)
    s = kern.sum()
    if s == 0:
        kern[cx, cy] = 1.0
    else:
        kern /= s
    if kern is not None:
        return cv2.filter2D(img, -1, kern)
    else:
        return img


def _do_permute_pixels4_gray(img, permute_conf):
    for x, y, nx, ny in permute_conf:
        x = int(x)
        y = int(y)
        nx = int(nx)
        ny = int(ny)
        img[y, x], img[ny, nx] = img[ny, nx], img[y, x]
    return img


def _do_permute_pixels_fast(rng, img, pixel_idx_to_permute):
    img = img.copy()
    h, w = img.shape[:2]
    neighbor_dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    permute_conf = []
    for idx in pixel_idx_to_permute:
        y, x = idx / w, idx % w
        dir_idx = rng.randint(4)
        dx, dy = neighbor_dirs[dir_idx]
        nx, ny = x + dx, y + dy
        if nx >= 0 and nx < w and ny >= 0 and ny < h:
            permute_conf.append((x, y, nx, ny))

    if img.ndim == 3:
        return cv2.merge([_do_permute_pixels4_gray(chn, permute_conf) for chn in cv2.split(img)])
    return _do_permute_pixels4_gray(img, permute_conf)


def permute_pixels(rng, img, ratio_range):
    ratio = rng_between(rng, ratio_range[0], ratio_range[1])
    nr_pixels = np.prod(img.shape[:2])
    nr_pixels_to_permute = max(1, int(nr_pixels * ratio))

    indexes = np.arange(nr_pixels)
    pixel_idx_to_permute = rng.choice(indexes, nr_pixels_to_permute)
    return _do_permute_pixels_fast(rng, img, pixel_idx_to_permute)


def erode_dialate(rng, img):
    from scipy.ndimage.filters import maximum_filter, minimum_filter
    ero = rng.randint(0, 3)
    if ero > 1:
        img = maximum_filter(img, footprint=np.ones((ero, ero, 1)))
    dil = rng.randint(0, 3)
    if dil > 1:
        img = minimum_filter(img, footprint=np.ones((dil, dil, 1)))
    return img


class NoiseAugmentor:
    def __init__(self, rng, change_every_iter=20):
        self._rng = rng
        self._gaussian_cnt = 0
        self._gaussian_noise = None
        self._salt_and_pepper_cnt = 0
        self._drop = None
        self._change_every_iter = change_every_iter

    def gaussian_noise(self, img, sigma):
        if self._gaussian_cnt == 0 or img.shape != self._gaussian_noise.shape:
            self._gaussian_noise = self._rng.randn(*img.shape)
            self._gaussian_cnt = 0
        self._gaussian_cnt += 1
        self._gaussian_cnt %= self._change_every_iter
        # return _clip_normalize(img + self._gaussian_noise * sigma)
        return img+self._gaussian_noise*sigma

    def salt_and_pepper_noise(self, img, black_prob, white_prob):
        if self._salt_and_pepper_cnt == 0 or img.shape != self._drop.shape:
            self._drop = self._rng.uniform(size=(img.shape[0], img.shape[1], 1))
            self._salt_and_pepper_cnt = 0
        self._salt_and_pepper_cnt += 1
        self._salt_and_pepper_cnt %= self._change_every_iter
        white = np.ones(img.shape) * 255
        white_mask = self._drop < white_prob
        black_mask = np.abs(1 - self._drop) < black_prob
        t = img * (1 - white_mask) + white * white_mask
        t = t * (1 - black_mask)
        return t
def aug_raw_carplate(rng, noise_gen, img):
    img = flare(rng, img, r_min=0.1, r_max=2.0, light_min=[50, 50, 50], light_max=[130, 130, 130])
    img = flare(rng, img, r_min=0.1, r_max=2.0, light_min=[50, 50, 50], light_max=[130, 130, 130])
    img = shadow_fast(rng, img, mask_min=0.5, mask_max=1.0, cut=1)
    img = shadow_fast(rng, img, mask_min=0.5, mask_max=1.0, cut=1)
    r = rng.rand()
    if r < 0.5:
        img = rank_one_brightness(rng, img, sigma=0.02, clip=(0.4, 2.0))
    elif r < 0.75:
        img = rank_one_brightness(rng, img, sigma=0.03, clip=(0.4, 2.0))
    img = contrast(rng, img, contrast_range=(-128, 64))
    img = noise_gen.salt_and_pepper_noise(img, black_prob=0.02, white_prob=0.02)
    img = noise_gen.gaussian_noise(img, sigma=(rng.rand() * 60))
    img = blinds(rng, img)
    img = backgroud_noise(rng, img)
    return img


def center_crop_img_to_patch(img, bgpatch):
    if img.ndim == 2 and bgpatch.ndim == 3:
        bgpatch = cv2.cvtColor(bgpatch, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and bgpatch.ndim == 2:
        bgpatch = cv2.cvtColor(bgpatch, cv2.COLOR_GRAY2BGR)
    bgpatch = np.copy(bgpatch)
    bgpatch[(bgpatch.shape[0] - img.shape[0]) // 2:(bgpatch.shape[0] - img.shape[0]) // 2 +
            img.shape[0], (bgpatch.shape[1] - img.shape[1]) // 2:(bgpatch.shape[1] - img.shape[1]) // 2 +
            img.shape[1]] = img
    return bgpatch


def center_crop(img, stroke, box_heatmap, coords, target_size):
    img_shape = img.shape
    crop_shape = target_size
    dy = (img_shape[0] - crop_shape[0]) // 2
    dx = (img_shape[1] - crop_shape[1]) // 2
    img = img[dy:dy + crop_shape[0], dx:dx + crop_shape[1], :]
    stroke = [s[dy:dy + crop_shape[0], dx:dx + crop_shape[1]] for s in stroke]
    box_heatmap = [s[dy:dy + crop_shape[0], dx:dx + crop_shape[1]] for s in box_heatmap]
    coords = [[(x - dx, y - dy) for x, y in coord] for coord in coords]
    return img, stroke, box_heatmap, coords


def spatial_augment(rng, img, stroke, box_heatmap, coords):
    h, w = img.shape[:2]
    angle = rng.uniform(-5, 5)
    dx, dy = rng.uniform(-10, 10), rng.uniform(-5, 5)
    affine = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    affine[0][2] += dx
    affine[1][2] += dy

    points = [(0, 0), (w, 0), (w, h), (0, h)]
    h_scale = rng.uniform(0.8, 1.02)
    v_scale = rng.uniform(0.6, 1.03)
    warp_perspective = cv2.getPerspectiveTransform(
        np.array(points, dtype='float32'),
        np.array([((x - w / 2 + rng.uniform(-10, 10)) * h_scale + w / 2,
                   (y - h / 2 + rng.uniform(-5, 5)) * v_scale + h / 2) for (x, y) in points],
                 dtype='float32'))
    merged = np.zeros((3, 3))
    merged[:2, :] = affine
    merged[2, 2] = 1
    merged = np.dot(merged, warp_perspective)

    img = cv2.warpPerspective(img, merged, (w, h))
    stroke = [cv2.warpPerspective(s, merged, (w, h)) for s in stroke]
    box_heatmap = [cv2.warpPerspective(s, merged, (w, h)) for s in box_heatmap]
    coords = [[np.dot(merged, np.array([x, y, 1]).reshape(3, 1)) for x, y in coord] for coord in coords]
    coords = [[(float(x / z), float(y / z)) for x, y, z in coord] for coord in coords]
    return img, stroke, box_heatmap, coords


def aug_before_crop(rng, img, gaussian_distortion):
    img = light_environment(rng, img, color_min=(.3, .3, .3), color_max=(1.7, 1.7, 1.7))
    img = contrast(rng, img, contrast_range=(-128, 128))
    img = brightness(rng, img, brightness_range=(0.7, 1.2))
    img = gaussian_distortion.distort_img(img)
    return img


def aug_after_crop(rng, noise_gen, img):
    img = gamma(rng, img, gamma_range=(0.3, 1.5))
    img = motion_blur(rng, img, length_range=(0, 24))
    sigma = rng.rand() * 6
    # img = gaussian_blur_given_sigma(img, sigma)
    img = noise_gen.gaussian_noise(img, sigma=(rng.rand() * 30))
    return img
