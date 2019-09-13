import  os,cv2,shutil,time,random,math
import  numpy as np
from fontTools.ttLib import TTCollection, TTFont
import matplotlib.pyplot as plt
from PIL import  Image


# ch_3500=open('data/ch_3500.txt','rb').read().decode().strip()
# ch_2000=open('data/ch_2000.txt','rb').read().decode().strip()
# charset=open('data/charset.txt','rb').read().decode().strip()
def load_bgs(bg_dir):
    dst = []

    for root, sub_folder, file_list in os.walk(bg_dir):
        for file_name in file_list:
            image_path = os.path.join(root, file_name)

            # For load non-ascii image_path on Windows
            bg = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),cv2.IMREAD_COLOR) #, cv2.IMREAD_GRAYSCALE

            dst.append(bg)

    print("Background num: %d" % len(dst))
    return dst




def load_font(font_path):
    """
    Read ttc, ttf, otf font file, return a TTFont object
    """

    # ttc is collection of ttf
    if font_path.endswith('ttc'):
        ttc = TTCollection(font_path)
        # assume all ttfs in ttc file have same supported chars
        return ttc.fonts[0]

    if font_path.endswith('ttf') or font_path.endswith('TTF') or font_path.endswith('otf'):
        ttf = TTFont(font_path, 0, allowVID=0,
                     ignoreDecompileErrors=True,
                     fontNumber=-1)

        return ttf


def display_imgs_from_dir(img_dir,num=16,title=True,show=True):
    output_dir=img_dir
    plt.figure()
    cols=int(math.sqrt(num)//1)+1

    rows=num//cols+1
    for i, fn in enumerate(os.listdir(output_dir)):
        if i >= num:
            break
        fpath = output_dir + '/' + fn
        img = Image.open(fpath)
        plt.subplot(rows, cols, i + 1)
        if title:
            plt.title(fn)
        plt.axis('off')
        plt.imshow(img)
    if show:
        plt.show()

def merge_datasets(data_dir1,data_dir2,target_dir):
    classes=os.listdir(data_dir1)
    ext='.jpg'
    for j,cls in enumerate(classes):
        cls_dir1=data_dir1+'/'+cls
        cls_dir2=data_dir2+'/'+cls
        cls_dir=target_dir+'/'+cls
        os.makedirs(cls_dir) if not os.path.exists(cls_dir) else None

        files1=os.listdir(cls_dir1)
        files1=[cls_dir1+'/'+f for f in files1]
        files2=os.listdir(cls_dir2)
        files2=[cls_dir2+'/'+f for f in files2]

        files=files1+files2

        for i,f in enumerate(files):
            target_file=cls_dir+'/'+cls+'_'+str(i)+ext
            shutil.copyfile(f,target_file)

        print('%s :merged files of class: %s to dir: %s'%(j,cls,cls_dir))
def merge_all_datasets(data_dirs,target_dir):
    classes = os.listdir(data_dirs[0])
    ext = '.jpg'
    for j, cls in enumerate(classes):
        cls_dirs=[i+'/'+cls for i in data_dirs]

        cls_dir = target_dir + '/' + cls
        os.makedirs(cls_dir) if not os.path.exists(cls_dir) else None

        cls_files=[c_dir+'/'+i  for c_dir in cls_dirs for i in os.listdir(c_dir)]

        for i, f in enumerate(cls_files):
            target_file = cls_dir + '/' + cls + '_' + str(i) + ext
            shutil.copyfile(f, target_file)

        print('%s :merged files of class: %s to dir: %s' % (j, cls, cls_dir))

def load_dict_from_text_file(f,divider=','):
    string=open(f,'rb').read().decode().strip()
    records=string.split('\n')
    dic={}
    for rec in records:
        [key,value]=rec.split(divider)
        dic[key]=value
    return dic
def holdon():
    input()

def random_from_range(interval):
    return random.random()*(interval[1]-interval[0])+interval[0]
def get_random(intervals,weights):
    total_weights=sum(weights)
    cumu_weights=weights
    for i in range(1,len(cumu_weights)):
        cumu_weights[i]=cumu_weights[i]+cumu_weights[i-1]
    cumu_weights=np.array(cumu_weights)
    cumu_weights=cumu_weights/total_weights
    n=random.random()
    for i in range(len(intervals)):
        if n<=cumu_weights[i]:
            return random_from_range(intervals[i])
def get_randn_clipped(interval):
    sigma=(interval[1]-interval[0])/4
    mu=(interval[1]+interval[0])/2
    # print(sigma,mu)
    while True:
        num=np.random.randn()*sigma+mu
        # print(num)
        if num>interval[0] and num<interval[1]:
            return num




import time,os,glob,shutil,random
import functools
class Config(dict):
    def __setattr__(self, key, value):
        self[key]=value
    def __getattr__(self, item):
        try:
            v=self[item]
            return v
        except:
            raise Exception('Config object has no key %s '%item)
    def add(self,key,value,description=''):
        self[key]=value
        self.description=description

class Timer:
    def __init__(self,verbose=False):
        self.history=[]
        self.dt_history = []
        self.steps=0
        self.start_time=time.time()
        self.history.append(self.start_time)
        self.verbose=verbose
        if self.verbose:
            print('Timer started at %s'%(self.start_time))

    def step(self):
        t=time.time()
        dt=t-self.history[-1]
        self.dt_history.append(dt)
        self.history.append(t)
        self.steps+=1

        if self.verbose:
            print('time since last step: %s'%(dt))
        return dt

    def end(self):
        t=time.time()
        self.end_time=t
        dt = t - self.history[-1]
        self.dt_history.append(dt)
        self.history.append(t)
        self.steps+=1
        if self.verbose:
            print('time since last step: %s'%(dt))
        return dt


def run_timer(func):
    name = func.__name__
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print('running %s ...'%(name))
        t=Timer(verbose=False)
        ret=func(*args,**kwargs)
        dt=t.end()
        print('finished running %s ,time consumed: %s'%(name,dt))
        return ret
    return wrapper



