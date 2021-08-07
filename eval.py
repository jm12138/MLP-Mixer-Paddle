from model import mixer_b
import os
import cv2
import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision.datasets import Cifar10
from PIL import Image

val_transforms = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
model = mixer_b(class_dim=10)
params = paddle.load(os.path.join('save', "best_model.pdparams"))
model.set_dict(params)
model = paddle.Model(model)
model.prepare(metrics=paddle.metric.Accuracy(topk=(1, 5)))

# 配置数据集
val_dataset = Cifar10(transform=val_transforms, backend='pil', mode='test')


# 模型验证
acc = model.evaluate(val_dataset, batch_size=16, num_workers=0)
print(acc)
