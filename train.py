from model import mixer_b

import os
import cv2
import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from PIL import Image
from paddle.callbacks import EarlyStopping, VisualDL, ModelCheckpoint

# 配置模型
train_transforms = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = mixer_b(class_dim=10, pretrained=True)
opt = paddle.optimizer.Adam(learning_rate=1e-5, parameters=model.parameters())
model = paddle.Model(model)
model.prepare(optimizer=opt, loss=nn.CrossEntropyLoss(), metrics=paddle.metric.Accuracy(topk=(1, 5)))

# 配置数据集
train_dataset = Cifar10(transform=train_transforms, backend='pil', mode='train')
val_dataset = Cifar10(transform=val_transforms, backend='pil', mode='test')

checkpoint = ModelCheckpoint(save_dir='save')

earlystopping = EarlyStopping(monitor='acc_top1',
                                mode='max',
                                patience=10,
                                verbose=1,
                                min_delta=0,
                                baseline=None,
                                save_best_model=True)

vdl = VisualDL('log')

# 模型验证
acc = model.fit(train_dataset, val_dataset, batch_size=32, num_workers=0, epochs=5, save_dir='save', callbacks=[checkpoint, earlystopping, vdl])
print(acc)
