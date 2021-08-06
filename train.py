from ppim import mixer_b

import os
import cv2
import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from PIL import Image


# 配置模型
train_transforms = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
val_transforms = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载模型
model = mixer_b(pretrained=True, class_dim=10)
opt = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
model = paddle.Model(model)
model.prepare(optimizer=opt, loss=nn.CrossEntropyLoss(), metrics=paddle.metric.Accuracy(topk=(1, 5)))

# 配置数据集
train_dataset = Cifar10(transform=train_transforms, backend='pil', mode='train')
val_dataset = Cifar10(transform=val_transforms, backend='pil', mode='test')

# 模型验证
acc = model.fit(train_dataset, val_dataset, batch_size=32, num_workers=0, epochs=50)
print(acc)
