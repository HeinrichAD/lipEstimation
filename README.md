# Lipschitz Estimation

Repository for the paper [Lipschitz Regularity of deep neural networks:
analysis and efficient estimation](https://arxiv.org/abs/1805.10965)

Basic Python dependencies needed: PyTorch >= 0.3


## Code organization

* `lipschitz_approximations.py`: many estimators
* `lipschitz_utils.py`: toolbox for the different estimators
* `seqlip.py`: SeqLip and GreedySeqLip
* `training.py`: general scheme for train/test
* `utils.py`: utility functions


## Requirements
- python 3.7 or later  
  `which python3.7`
- virtualenv or venv  
  `pip install -U virtualenv`


## Installation

```bash
pip install -U "git+https://github.com/HeinrichAD/lipEstimation.git@master"
```

setup.py
```bash
...
install_requires=[
    ...
    "lipestimation @ git+https://github.com/HeinrichAD/lipEstimation.git@master",
    ...
],
...
```

### Development

```bash
# create virtual environment
virtualenv -p $(which python3.7) .venv
# or
python -m venv .venv

# activate our virtual environment:
source .venv/bin/activate

# update pip (optional)
python -m pip install -U pip

# install
pip install -e .[dev]
```
