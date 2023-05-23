English | [简体中文](README_CN.md)

# Mnist CGAN

## Background
This is an implementation of [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) based on the [MINST dataset]( http://yann.lecun.com/exdb/mnist/).

The MNIST data set comes from the National Institute of Standards and Technology, National Institute of Standards and Technology (NIST). The training set (training set) consists of digits handwritten by 250 different people, of which 50% are high school students and 50% are from the population Census Bureau (the Census Bureau) staff. The test set (test set) is also the same proportion of handwritten digit data.

## API
The interface class file `aigcmn.py` implements the interface class `AiGcMn`. The interface class AiGcMn provides an interface function generate. The parameter of this function is an integer $n$-dimensional tensor ( $n$ is the size of the batch, each integer is in the range of 0~9, representing the number to be generated), and the output is $n*1*28*28$ tensor ( $n$ is the size of the batch, each $1*28*28$ tensor represents a randomly generated digital image).

## Usage

This is a pytorch project so it could be used as follow:

```
pip install -r requirement.txt
python3 aigcmn.py
```

The output png figure could be found in `/result`.
## License
[MIT](LICENSE) &copy; Wortox Young