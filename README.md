# On the Paradox of Certified Training <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>


The code accompanying our TMLR 10/2022 paper: [**On the Paradox of Certified Training**](https://openreview.net/forum?id=atJHLVyBi8).

For a brief overview of the work, check out our **[blogpost](https://www.sri.inf.ethz.ch/blog/paradox)**.

## Requirements

We require the following packages:
```
cudatoolkit=10.1
matplotlib=3.3.2
numpy=1.19.2
pytorch==1.6.0
scikit-learn=0.23.2
tabulate=0.8.7
torchvision==0.7.0 
tqdm=4.50.2
```

To set up a sufficient conda environment, start by installing pytorch 1.6.0 in a python 3.6.9 environment:
```
$ conda create -n myenv python=3.6.9
$ conda activate myenv
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

Then, install the remaining required packages using the provided file `requirements.txt`:
```
$ pip install -r requirements.txt
```

## Running the code

To run certified training with convex relaxations invoke `cert.py`. In `scripts/` we provide the example commands used to train all models shown in Tables 2, 3, and 4. 

The training will save model snapshots to `saved_models/` and the results to `results/` (the folders will be automatically created). 

To run with different settings, modify the examples, referring to `args_factory.py` which documents the command-line flags.

Additionally, we provide snapshots of the models from Table 2 in `pretrained/`. In `scripts/table2.sh` we give an example of a command used to evaluate one such snapshot.

## Citation

```
@article{
    jovanovic2022on,
    title={On the Paradox of Certified Training},
    author={Nikola Jovanovi{\'c} and Mislav Balunovi\'{c} and Maximilian Baader and Martin Vechev},
    journal={Transactions on Machine Learning Research},
    year={2022},
    url={https://openreview.net/forum?id=atJHLVyBi8},
    note={}
}
```
