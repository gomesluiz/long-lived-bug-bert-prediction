# Bug Severity Predictor for Mozilla

This repository contains code and associated files for deploying a bug severity predictor for Mozilla projects using Heroku.

## Project Overview

In this project, I'll build a severity predictor for the [Mozilla project](https://www.mozilla.org/en-US/) that uses the description of a bug report stored a in [Bugzilla Tracking System](https://bugzilla.mozilla.org/home) to predict its severity. 

The severity in the Mozilla project indicates how severe the problem is – from blocker ("application unusable") to trivial ("minor cosmetic issue"). Also, this field can be used to indicate whether a bug is an enhancement request. In my project, I have considered five severity levels: **trivial**, **minor**, **major**, **critical**, and **blocker**. I have ignored the default severity level (often **"normal"**) because this level is considered as a choice made by users when they are not sure about the correct severity level. 

This project will be broken down into four main notebooks:

**Notebook 1: Data Preparation**
* Download the necessary data from [Mendeley Data] (https://data.mendeley.com/datasets/v446tfssgj/2) and extract the files into the folder **data/raw**.
* Explore the existing data features and the data distribution.
* Clean and pre-process the bug reports data for next steps in machine learning workflow.
* Export cleaned data into the folder **data/cleaned**
* Notebook file path [Here](./1-data-preparation/prepare-data.ipynb).

**Notebook 2: Data Analysis**
* Analyze the dataset to summarize their main characteristics using visual methods.
* Explore the class distributions and word distributions by bug severity levels. 
* Notebook file path [Here](./2-data-analysis/exploratory-data-analysis.ipynb).

**Notebook 3: Feature Engineering**
* Extract features in tensor data format from cleaned data via [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5) deep learning network. 
* Export train/test `.pt` files that hold the relevant features and class labels for train/test data points into the folder **data/processed**.
* Notebook file path [Here](./3-feature-engineering/extract-features.ipynb).

**Notebook 4: Predictive Modeling**
* Define a multilabel classification model and a training script.
* Train the model using [XGBoost](https://xgboost.readthedocs.io/en/latest/) and [Hyperopt](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a).
* Evaluate and save the trained model into **data/model**.
* Notebook file path [Here](./4-predictive-modeling/built-and-save-model.ipynb).

## Packages 
The table below shows the main required third-party Python packages to run 
codes and notebooks of this project.

* flask
* flask-wtf
* flask-boostrap
* gnunicorn
* matplotlib
* numpy
* pandas
* scikit-learn
* seaborn
* torch 
* transformers 
* xbgboost

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements packages
describe in requirements.txt in the project root.

```bash
pip install -Uqr requirements.txt
```

## Usage

```bash
export FLASK_APP=app.py 
flask run
```
To predict a bug severity level, access the [local address](http://127.0.0.1:5000) in a web browser and type a **bug id** from [Mozilla Bug Tracking System](https://bugzilla.mozilla.org/home).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
