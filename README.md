# AllenCV

[![Build Status](https://travis-ci.com/sethah/allencv.svg?branch=master)](https://travis-ci.com/sethah/allencv)
[![codecov](https://codecov.io/gh/sethah/allencv/branch/master/graph/badge.svg)](https://codecov.io/gh/sethah/allencv)

A computer vision library built on top of [PyTorch](https://github.com/pytorch/pytorch),
[Torchvision](https://github.com/pytorch/vision), 
and [AllenNLP](https://github.com/allenai/allennlp).

## Quick Links

* [Tutorials](tutorials)
* [Continuous Build](https://travis-ci.com/sethah/allencv)

## Installation

```
pip install allencv
```

### Installing from source

Clone the git repository:

  ```bash
  git clone https://github.com/sethah/allencv.git
  ```

In a Python 3.6 virtual environment, run:

  ```bash
  pip install --editable .
  ```

This will make `allencv` available on your system but it will use the sources from the local clone
you made of the source repository.

## Running AllenCV

AllenCV follows all the same conventions and supports the same interfaces as AllenNLP. Read more
about the AllenNLP command line interface [here](https://github.com/allenai/allennlp#running-allennlp).
