import random,os,time
from .renderer import Renderer
from . import utils
from . import config as cfg

def gen_data(data_dir,num_img_per_char):
    t_start=time.time()
    renderer=Renderer()
    # charset=utils.ch_3500
    charset=cfg.charset

    # 字符集
    # charset=charset[:10]

    for j,char in enumerate(charset):
        # 每个汉字一千张图
        for i in range(num_img_per_char):
            img=renderer.gen_one_img(char).convert('RGB')
            output_dir=data_dir+'/'+char
            os.makedirs(output_dir) if not os.path.exists(output_dir) else None
            fn=output_dir+'/'+'%s_%s'%(char,i)+ '.jpg'
            print('%s, %s : %s'%(j,char, i))
            img.save(fn)
    t_end=time.time()
    print('time consumed: %s'%(t_end-t_start))
    pass


def gen_train_data(data_dir='dataset/train',num_per_char=100):
    gen_data(data_dir,num_per_char)


if __name__=="__main__":
    gen_data('../dataset/train_6_rotate_more_300',500)
