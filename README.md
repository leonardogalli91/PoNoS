## PoNoS - POlyak NOnmonotone Stochastic line search

The first nonmonotone stochastic line search for training over-parameterized models.

### 1. Installation

`pip install git+https:github.com/leonardogalli91/PoNoS.git`

### 2. Requirements

```
pip install -r requirements.txt
```

### 3. Experiments

#### 3.1 Dataset

Set to ```True``` the options ```download``` in the file ```src/datasets.py```

#### 3.2 Launching experiments

`python trainval.py -e mnist_mlp -sb results/mnist_mlp -d data -r 1`

where `-e` is the experiment group, `-sb` is the result directory, and `-d` is the dataset directory.
The experiment group is referring to the key of the dict ```EXP_GROUPS```, that can be found in the file ```exp_configs.py```,
from that file it is possible to customize thoroughly the experiment. 

### 4. Plot Results

In the file ```plot.py``` set the variable ```savedir_base``` to point at the root directory where you saved the results, then run

`python plot.py -p mnist_mlp`



