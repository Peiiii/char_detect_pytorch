import os, glob, shutil, random, PIL, cv2
import numpy as np


class DataIterator:
    def __init__(self):
        self.step = -1
        self.max_step = 9

    def __iter__(self):
        return self

    def __next__(self):
        self.step += 1
        if self.step > self.max_step:
            self.step = -1
            raise StopIteration
        return self.step

    def __len__(self):
        return self.max_step


class DataLoader:
    def __init__(self, data_dir, batch_size=32,charset=None,flags=[]):
        self.data_dir = data_dir
        self.files = glob.glob(data_dir + '/*/*.jpg')
        random.shuffle(self.files)
        self.num_xs = len(self.files)
        self.batch_size = batch_size
        self.num_batches = self.num_xs // self.batch_size
        self.batch_index = -1
        self.current_batch = None
        self.charset=charset
        self.flags=flags
    def __iter__(self):
        return self

    def __next__(self):
        self.batch_index += 1
        if self.batch_index == self.num_batches:
            self.batch_index = -1
            random.shuffle(self.files)
            raise StopIteration

        files = self.files[self.batch_index * self.batch_size:(self.batch_index + 1) * self.batch_size]
        # 加载并预处理图片
        xs = np.array([self.preprocess_x(cv2.imread(f)) for f in files])
        labels = np.array([self.load_label(f) for f in files])
        self.current_batch = {
            'xs': xs,
            'labels': labels
        }
        return (xs, labels)

    def __len__(self):
        return self.num_batches

    def preprocess_x(self, x):
        x = cv2.resize(x, (64, 64))
        x = cv2.normalize(x, None, -1, 1, norm_type=cv2.NORM_MINMAX)
        if 'torch' in self.flags:
            x=np.transpose(x,[2,0,1])
        return x

    def load_label(self, fp):
        '''
        :param fp: image path
        :return:label corresponding to the image
        get the label according to the image path
        It depends on the way you store your label
        '''
        char = os.path.basename(os.path.dirname(fp))
        # 将字符转换为向量
        charset = self.charset
        index = charset.index(char)
        if 'categorical' in self.flags:
            vec = np.zeros((len(charset),))
            vec[index] = 1
            return vec
        else:
            return index
    def get_categorical(self, char, charset):
        index = charset.index(char)
        vec = np.zeros((len(charset),))
        vec[index] = 1
        return vec

# D = DataIterator()
# for i in D:
# 	print(i)
# for i in D:
# 	print(i)
