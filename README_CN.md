简体中文 | [English](README.md)

# Mnist条件生成器

## 背景
这是基于 [MINST 数据集]( http://yann.lecun.com/exdb/mnist/) 的 [CGAN](https://arxiv.org/abs/1411.1784) 的实现。

MNIST 数据集来自美国国家标准技术研究院（National Institute of Standards and Technology，简称 NIST）。 训练集（training set）由250个不同的人手写的数字组成，其中50%是高中生，50%来自人口普查局（the Census Bureau）的工作人员。 测试集（test set）也是同样比例的手写数字数据。

## API
接口类文件`aigcmn.py`，实现了接口类`AiGcMn`。接口类AiGcMn提供一个接口函数generate，该函数的参数是一个整数型n维tensor（n是batch的大小，每个整数在0~9范围内，代表需要生成的数字），输出是n\*1\*28\*28的tensor（n是batch的大小，每个1\*28\*28的tensor表示随机生成的数字图像）。

## 使用方式

这是一个 pytorch 项目，因此可以按如下方式使用：

```
pip install -r requirement.txt
python3 aigcmn.py
```

## 协议
[MIT](LICENSE) &copy; Wortox Young