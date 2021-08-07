# MLP-Mixer-Paddle
使用 Paddle 复现 MLP-Mixer 模型

# 精度表现（Cifar10）
```
{'acc_top1': 0.9622, 'acc_top5': 0.9995}
```

# AIStudio 项目
[MLP is all you need ?](https://aistudio.baidu.com/aistudio/projectdetail/1924298)

# 模型微调训练（Cifar10 base mixer-b on ImageNet-1k）
```shell
$ python train.py
```

# 模型精度验证
```shell
$ python eval.py
```
