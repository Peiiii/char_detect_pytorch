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

ROOT='recongnize'

restore=True

data_dir=ROOT+'/data'

weights_dir=data_dir+'/weights'

weights_init=weights_dir+'/init.model'

log_path=data_dir+'/train.log'

train_data_dir='/home/ocr/wp/datasets/my/aihero/recongnize/train_10'
# val_data_dir='/home/ocr/wp/datasets/my/aihero/recongnize/val/char_5'
val_data_dir='/home/ocr/wp/datasets/aihero/AI+Hero_数据集2/企业验证码单字图'

weights_save_path=weights_dir+'/training.model'
val_step=500
batch_size=256

charset=list(open(ROOT+'/data/ch_3500.txt','r').read().strip())
charset.sort()
charset=charset[:-1]

config=Config()
config.add('restore',True,'True or False')
config.add('weights_dir',weights_dir)
config.add('weights_init',weights_init)
config.add('weights_save_path',weights_save_path)
config.add('log_path',log_path)
config.add('train_data_dir',train_data_dir)
config.add('val_data_dir',val_data_dir)
config.add('val_step',val_step)
config.add('batch_size',batch_size)
config.add('charset',charset)
