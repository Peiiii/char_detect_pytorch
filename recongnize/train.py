import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from .utils.data_iterator import DataLoader
from . import config as cfg
import numpy as np


class Trainer:
    def __init__(self):
        self.restore = cfg.restore
        self.weights_dir = cfg.weights_dir
        self.weights_init = cfg.weights_init

        self.num_classes = len(cfg.charset)
        self.batch_size = cfg.batch_size
        self.val_step = cfg.val_step
        self.device = torch.device('cuda:0')
        self.model = self.get_model()
        self.log_path = cfg.log_path
        self.patience = 3

        self.train_data_dir = cfg.train_data_dir
        self.val_data_dir = cfg.val_data_dir

    def get_model(self):
        if self.restore:
            try:
                model = torch.load(self.weights_init)
                print('Restore from model %s' % (self.weights_init))
                return model
            except:
                print('Initial weights not found. Model will be initialized randomly.')
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, self.num_classes)
        model.to(self.device)
        return model

    def init_train(self):
        self.best_accuracy = -1
        self.best_loss = np.Inf
        self.bad_epoch = 0
        self.train_acc = 0
        self.val_acc = 0

    def val_and_save(self, watch_bad_epoch=False):
        acc, _ = self.val()
        if acc > self.best_accuracy:
            self.bad_epoch = 0
            self.best_accuracy = acc
            save_path = self.weights_dir + '/tmp_best.model'
            torch.save(self.model, save_path)
            self.log('new best accuracy %s , model save to %s' % (acc, save_path))
        elif watch_bad_epoch and self.train_acc > 0.95:
            self.bad_epoch += 1
            if self.bad_epoch > self.patience:
                self.log('Accuracy is no longer improved, quit training, top accuracy %s .' % (self.best_accuracy))
        return acc

    def train2(self):
        self.init_train()
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=self.train_data_dir,
                                                         transform=train_transform)
        train_feeder = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)

        self.model.train()
        for epoch in range(1000):
            train_loss = []
            train_acc = []
            self.model.train()
            for i, data in enumerate(train_feeder):
                imgs, labels = data
                imgs = imgs.cuda(self.device)
                labels = labels.cuda(self.device)

                optimizer.zero_grad()
                preds = self.model(imgs)
                loss = criterion(preds, labels)
                train_loss.append(float(loss))
                loss.backward()
                optimizer.step()

                _, preds = torch.max(preds, 1)
                acc = float((preds == labels).sum() / len(preds))
                train_acc.append(acc)
                if i % 100 == 0:
                    print('\nepoch/step: %s/%s, batch_train_acc: %s , batch_train_loss: %s ' % (
                    epoch, i, acc, float(loss)))
                else:
                    print('*', end='')

                if i % self.val_step == self.val_step - 1:
                    # continue
                    self.val_and_save()

            train_acc = np.average(train_acc)
            train_loss = np.average(train_loss)
            print('\nepoch: %s, train_acc: %s , train_loss: %s ' % (epoch, train_acc, train_loss))

            self.train_acc = train_acc
            self.val_and_save(watch_bad_epoch=True)

            lr_scheduler.step()

    def val(self):
        val_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_dataset = torchvision.datasets.ImageFolder(root=self.val_data_dir,
                                                       transform=val_transform)
        val_feeder = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        acc_all = []
        loss_all = []
        for i, (imgs, labels) in enumerate(val_feeder, 0):
            imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
            # print(imgs.shape)
            preds = self.model(imgs)
            loss = criterion(preds, labels)
            loss_all.append(float(loss))

            _, preds = torch.max(preds, 1)
            acc_all.append(float((preds == labels).sum() / len(preds)))
        acc = np.average(acc_all)
        loss = np.average(loss_all)

        self.shout('\nvalidation accuracy : %s , loss : %s' % (acc, loss))
        return acc, loss

    def log(self, text, show=True):
        print('\n')
        with open(self.log_path, 'a') as f:
            f.write(text)
            if show:
                self.shout(text)

    def shout(self, text):
        print('*' * 0 + '' + text)

    def accuracy(self, preds, labels):
        return

    def train(self):
        self.init_train()
        train_feeder = DataLoader(data_dir='/home/ocr/wp/datasets/my/aihero/recongnize/train_10',
                                  batch_size=self.batch_size,
                                  charset=cfg.charset,
                                  flags=['torch'])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)

        self.model.train()
        for epoch in range(1000):
            train_loss = []
            train_acc = []
            self.model.train()
            for i, data in enumerate(train_feeder):
                imgs, labels = data
                imgs,labels=torch.Tensor(imgs),torch.Tensor(labels).long()
                imgs,labels = imgs.cuda(self.device),labels.cuda(self.device)

                optimizer.zero_grad()
                preds = self.model(imgs)
                loss = criterion(preds, labels)
                train_loss.append(float(loss))
                loss.backward()
                optimizer.step()

                _, preds = torch.max(preds, 1)
                acc = float((preds == labels).sum() / len(preds))
                train_acc.append(acc)

                print('\nepoch/step: %s/%s, batch_train_acc: %s , batch_train_loss: %s ' % (epoch, i, acc, float(loss))) if i % 100 == 0 else print('*', end='')
                self.val_and_save() if i % self.val_step == self.val_step - 1 else None


            train_acc = np.average(train_acc)
            train_loss = np.average(train_loss)
            print('\nepoch: %s, train_acc: %s , train_loss: %s ' % (epoch, train_acc, train_loss))

            self.train_acc = train_acc
            self.val_and_save(watch_bad_epoch=True)

            lr_scheduler.step()


