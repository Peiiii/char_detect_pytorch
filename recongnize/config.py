

ROOT='recongnize'

restore=True

data_dir=ROOT+'/data'

weights_dir=data_dir+'/weights'

weights_init=weights_dir+'/tmp_best.model'

log_path=data_dir+'/train.log'

train_data_dir='/home/ocr/wp/datasets/my/aihero/recongnize/train_10'
val_data_dir='/home/ocr/wp/datasets/my/aihero/recongnize/val/char_5'


val_step=500
batch_size=256


charset=open(ROOT+'/data/ch_3500.txt','r').read().strip()
charset=list(charset)
charset.sort()