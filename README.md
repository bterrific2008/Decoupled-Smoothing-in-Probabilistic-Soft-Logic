# Decoupled Smoothing in Probabilistic Soft Logic

Experiments for "Decoupled Smoothing in Probabilistic Soft Logic".

## Replication code: "Decoupled smoothings on graphs"

The original paper and code for "Decoupled Smoothings on Graphs" can be found here:

* [Decoupled smoothing on graphs](https://dl.acm.org/citation.cfm?doid=3308558.3313748) (WWW 2019) - [Alex Chin](https://ajchin.github.io/), [Yatong Chen](https://github.com/YatongChen/), [Kristen M. Altenburger](http://kaltenburger.github.io/), [Johan Ugander](https://web.stanford.edu/~jugander/).
* [Github Repository](https://github.com/YatongChen/decoupled_smoothing_on_graphs)

## Machine Learning Framework: "Probabilistic Soft Logic"

Probabilistic Soft Logic is a machine learning framework for developing probabilistic models. You can find more information about PSL available at the [PSL homepage](https://psl.linqs.org/) and [examples of PSL](https://github.com/linqs/psl-examples). 

## Documentation

This repository contains code to run PSL rules for one hop (homophily) and two hop (monophily) methods to predict genders in a social network. 
We provide links to the datasets (Facebook100) in the data sub-folder.

### Obtaining the data

This repository set-up assumes that the FB100 (raw `.mat` files) have been acquired and are saved the data folder. Follow these steps:
1. The Facebook100 (FB100) dataset is publicly available from the Internet Archive at https://archive.org/details/oxford-2005-facebook-matrix and other public repositories. Download the datasets.
2. Save raw datasets in placeholder folder data. They should be in the following form: `Amherst41.mat`.

### Reproducing results

To reproduce the results, you can run `run_method.sh` or `run_all.sh`. Make sure that permissions are set so you can run these scripts. For example, 

`run_method.sh`: This runs a selected method for a specified seed for a specified percentage for either learning or evaluation.

This takes the following positional parameters: 
* data: what datafile you would like to use
* random seed: what seed to use
* percent labeled: what percentage of labeled data
* {learn|eval}: specify if you're learning or evaluating
* method dir: this is the path to the directory you'd like the run

`run_all.sh`: This runs a selected method for all random seeds at all percentages

This takes the following positional parameters: 
* data: what datafile you would like to use
* method dir: this is the path to the directory you'd like the run

Output will be written in the following directory

results/{method run}/{eval|learn}/{data used}/{random seed}/

`results/decoupled-smoothing/{eval|learn}/{method run}/{data used}/{random seed}/`

The directory will contain a set of folders for the inferences found at each percent labeled, named `inferred-predicates{pct labeled}`.
The folder will also contain the a copy of the `base.data`, `gender.psl`, files and output logs from the runs.
