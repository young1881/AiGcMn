简体中文 | [English](README.md)

# Mnist条件生成器

![](https://img.shields.io/badge/License-MIT-brightgreen.svg) ![](https://img.shields.io/badge/build-passing-brightgreen.svg) ![](https://img.shields.io/badge/Release-Ver2.0-blueviolet.svg) ![](https://img.shields.io/badge/python->=3.8-blue.svg)
## 背景
这是基于 [MINST 数据集]( http://yann.lecun.com/exdb/mnist/) 的 [CGAN](https://arxiv.org/abs/1411.1784) 的实现。

MNIST 数据集来自美国国家标准技术研究院（National Institute of Standards and Technology，简称 NIST）。 训练集（training set）由250个不同的人手写的数字组成，其中50%是高中生，50%来自人口普查局（the Census Bureau）的工作人员。 测试集（test set）也是同样比例的手写数字数据。

## API
接口类文件 `aigcmn.py` ，实现了接口类 `AiGcMn` 。接口类AiGcMn提供一个接口函数generate，该函数的参数是一个整数型 $n$ 维tensor（ $n$ 是batch的大小，每个整数在0~9范围内，代表需要生成的数字），输出是 $n*1*28*28$ 的tensor（n是batch的大小，每个 $1*28*28$ 的tensor表示随机生成的数字图像）。

## 使用方式

这是一个 pytorch 项目，因此可以按如下方式使用：

```
pip install -r requirements.txt
python3 aigcmn.py 
```

之后 `AiGnMn` 类的 `input()` 方法会将输入的整型数转为 $n$ 维tensor。得到的tensor作为 `generate()` 方法的输入后返回 $n*1*28*28$ 的tensor。输出tensor的文本和图片都写入到 `/result` 目录下。

## 贡献者
感谢以下参与项目的人：
[Fannyzzzz](https://github.com/Fannyzzzz), [Skyuan07](https://github.com/Skyuan07), [chatterboxthur](https://github.com/chatterboxthur), [noiho](https://github.com/noiho), [young1881](https://github.com/young1881)
## 协议
[MIT](LICENSE) &copy; Wortox Young