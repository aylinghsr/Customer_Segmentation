# Customer Segmentation using Data Mining algorithms

## Introduction
In this project, the aim is to import and use several Data Mining methods on [Customer Segmentation dataset](https://www.kaggle.com/datasets/vetrirah/customer) (on [Kaggle](https://www.kaggle.com)) in order to achieve an accurate customer segmentation. In marketing, customer segmentation is the process of splitting a broad market, which is usually consisting of already existing and potential customers, into sub-groups, also known as segments, based on shared characteristics. This is done for offering different programs, such as prices, promotions or distribution, to different segments.

In this project, the following ML models are used: Decision Tree, Random Forest, Naive Bayes, KNN, Perceptron, and Logistic Regression.
There is also a model ensemble of the mentioned models, excluding Decision Tree and Perceptron. The ensemble adopts Majority Voting technique.

Since we are merely interested in the correctly classified data points, accuracy is used as the evaluation metric of this project.

## Installation
First, you need to clone this repository to your local machine via the following command:
```shell
$ git clone https://github.com/aylinghsr/Customer_Segmentation.git
```
In case you don't have `git` installed on your computer, you can download the zip file of this repository and then, extract it.

## Requirements
This project is written in Python3 and requires Scikit-learn, Pandas, and Numpy libraries.

All the required libraries can be installed by running the following command:
```shell
$ pip install -r requirements.txt
```
If the command above results in an error, you can also try:
```shell
$ python -m pip install -r requirements.txt
```
Also, the dataset (.csv files) should be downloaded on your computer.

Dataset: [Customer Segmentation](https://www.kaggle.com/datasets/vetrirah/customer)

## Usage
Run:
```shell
$ cd Customer_Segmentation
$ python main.py
```