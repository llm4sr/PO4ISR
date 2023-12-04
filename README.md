# PO4ISR
This is the code for our paper "Prompt Optimization for Intent-Aware Session Recommendation".

## Installtion
Install all dependencies via:
```
pip install -r requirements.txt
```

## Datasets
We adopt three real-world datasets from various domains: [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), [Games](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) and [Bundle](https://github.com/BundleRec/bundle_recommendation) dataset. You can find the datasets used in the experiment under the `Dataset` directory. Each dataset includes both its ID and text forms. In addition to providing a small subset of training data versions, we also provide full versions of the training data.
## PO4ISR
### Tune
The `tune.py` corresponds to the process of prompt optimization. Before running the code, you need to fill in your OpenAI API token in the `./PO4ISR/assets/openai.yaml` file and wandb token in the `./PO4ISR/assets/overall.yaml` file. 
```
python tune.py --dataset='dataset name' --sample_num='number of training data'
```
### Test
The `test.py` file corresponds to the evaluation of the prompt.
```
python test.py --dataset='dataset name' --seed='value of the seed'
```
The top 1/2 optimized prompts for the three domains can be found in the `prompts.py`. If you want to test the results with these prompts, you can replace them in the `test.py`.

## NIR
```
python test.py --dataset='dataset name' --seed='value of the seed' --api_key='your OpenAI API token'
```
## Conventional methods
### Parameter Tuning and Settings for Conventional methods
We use [Optuna](https://optuna.org/) to automatically find out the optimal hyperparameters of all methods with 30 trails. The search space for item embedding size,learning rate, and batch size are {32, 64, 128}, {10−4, 10−3, 10−2} and {64, 128, 256}, respectively. For SKNN, 𝐾 is searched from
{50, 100, 150}. For NARM, the hidden size and layers are searched in [50, 200] stepped by 50 and in {1, 2, 3}, respectively. For GCE-GNN, the number of hops, and the dropout rate for global and local aggregators are respectively searched in {1, 2}, [0, 0.8] stepped by 0.2,
and {0, 0.5}. For MCPRN, 𝜏 and the number of purpose channels are separately searched in {0.01, 0.1, 1, 10} and {1, 2, 3, 4}. For HIDE, the number of factors is searched in {1, 3, 5, 7, 9}; the regularization and balance weights are searched in {10−5, 10−4, 10−3, 10−2}; the window size is searched in [1, 10] stepped by 1; and the sparsity
coefficient is set as 0.4. For Atten-Mixer, the intent level 𝐿 and the number of attention heads are respectively searched in [1, 10] stepped by 1 and in {1, 2, 4, 8}. The optimal parameter settings are shown in Table 1.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Table 1: Parameter settings for Conventional methods.

|  | Bundle | ML-1M | Games |
| :------: | :------: | :------: | :------: |
| SKNN | ![equation](https://latex.codecogs.com/svg.image?K=50)| ![equation](https://latex.codecogs.com/svg.image?K=50) | ![equation](https://latex.codecogs.com/svg.image?K=50) |
| FPMC | ![equation](https://latex.codecogs.com/svg.image?embedding\\_size=32) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=128)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=64) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=128) |![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)|
|NARM|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=32)<br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001)<br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)<br> ![equation](https://latex.codecogs.com/svg.image?hidden\\_size=100)<br> ![equation](https://latex.codecogs.com/svg.image?layers=2)| ![equation](https://latex.codecogs.com/svg.image?embedding\\_size=64)<br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.0001)<br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)<br> ![equation](https://latex.codecogs.com/svg.image?hidden\\_size=50)<br> ![equation](https://latex.codecogs.com/svg.image?layers=2)| ![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128)<br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01)<br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=128)<br> ![equation](https://latex.codecogs.com/svg.image?hidden\\_size=100)<br> ![equation](https://latex.codecogs.com/svg.image?layers=3)|
|STAMP|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=32) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.0001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)|
|GCE-GNN|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)<br> ![equation](https://latex.codecogs.com/svg.image?num\\_hop=2)<br>![equation](https://latex.codecogs.com/svg.image?dropout\\_gcn=0)<br>![equation](https://latex.codecogs.com/svg.image?dropout\\_local=0.5)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=32) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)<br> ![equation](https://latex.codecogs.com/svg.image?num\\_hop=1)<br>![equation](https://latex.codecogs.com/svg.image?dropout\\_gcn=0)<br>![equation](https://latex.codecogs.com/svg.image?dropout\\_local=0.5)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)<br> ![equation](https://latex.codecogs.com/svg.image?num\\_hop=2)<br>![equation](https://latex.codecogs.com/svg.image?dropout\\_gcn=0)<br>![equation](https://latex.codecogs.com/svg.image?dropout\\_local=0)|
|MCPRN|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=32) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=128)<br>![equation](https://latex.codecogs.com/svg.image?\tau=0.01)<br>![equation](https://latex.codecogs.com/svg.image?purposes=2)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)<br>![equation](https://latex.codecogs.com/svg.image?\tau=0.1)<br>![equation](https://latex.codecogs.com/svg.image?purposes=4)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.01) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)<br>![equation](https://latex.codecogs.com/svg.image?\tau=1)<br>![equation](https://latex.codecogs.com/svg.image?purposes=1)|
|HIDE|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=64) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.0001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)<br>![equation](https://latex.codecogs.com/svg.image?n\\_factor=1)<br>![equation](https://latex.codecogs.com/svg.image?regularization=1e-3)<br>![equation](https://latex.codecogs.com/svg.image?balance\\_weights=0.01)<br>![equation](https://latex.codecogs.com/svg.image?window\\_size=6)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)<br>![equation](https://latex.codecogs.com/svg.image?n\\_factor=1)<br>![equation](https://latex.codecogs.com/svg.image?regularization=1e-2)<br>![equation](https://latex.codecogs.com/svg.image?balance\\_weights=0.001)<br>![equation](https://latex.codecogs.com/svg.image?window\\_size=5)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)<br>![equation](https://latex.codecogs.com/svg.image?n\\_factor=1)<br>![equation](https://latex.codecogs.com/svg.image?regularization=1e-5)<br>![equation](https://latex.codecogs.com/svg.image?balance\\_weights=1e-5)<br>![equation](https://latex.codecogs.com/svg.image?window\\_size=3)|
|Atten-Mixer|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=32) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.0001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)<br> ![equation](https://latex.codecogs.com/svg.image?level\\_L=7) <br> ![equation](https://latex.codecogs.com/svg.image?number\\_of\\_attention\\_heads=1)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=32) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=64)<br> ![equation](https://latex.codecogs.com/svg.image?level\\_L=10) <br> ![equation](https://latex.codecogs.com/svg.image?number\\_of\\_attention\\_heads=2)|![equation](https://latex.codecogs.com/svg.image?embedding\\_size=128) <br> ![equation](https://latex.codecogs.com/svg.image?learning\\_rate=0.001) <br> ![equation](https://latex.codecogs.com/svg.image?batch\\_size=256)<br> ![equation](https://latex.codecogs.com/svg.image?level\\_L=3) <br> ![equation](https://latex.codecogs.com/svg.image?number\\_of\\_attention\\_heads=4)|
### Tune
You can run the following command to tune the model and find the optimal parameter combination.
```
python tune.py --dataset='dataset name' --sample_num='number of training data' --model='model name'
```
### Test
After completing the tuning process, we can find the optimal parameter combination in the `tune_log` directory. Additionally, we have placed the optimal parameters obtained during the experiment in the `config.py`. You can also use the following command to directly test the model.
```
python test.py --dataset='dataset name' --model='model name' --seed='value of the seed'
```
## Acknowledgment
We refer to the following repositories to improve our code:
* Conventional methods part with [Understanding-Diversity-in-SBRSs](https://github.com/qyin863/Understanding-Diversity-in-SBRSs)
* NIR parts with [LLM-Next-Item-Rec](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec)
