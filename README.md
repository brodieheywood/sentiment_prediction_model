# Sentiment Prediction Model

## Description

A sentiment-predicting Python model.

A report for BCIT's Predictive Analytics class, completed October 2020.

Includes iPython notebooks for training a sentiment prediction model on 
labelled IMDB reviews (including strategies for cleaning the training data). 
Also includes a trained model and a script to run predictions on sample data.

## Contents

- `report_writeup.pdf`: Details of the project, including the reasoning behind 
some of the decisions I made in the training code.
- `training_code/` (directory): Code (.ipynb files) used to clean the training 
data and train the model.
- `model_prediction/` (directory): Contains an image of the model 
(`lstm_model.h5`), unlabelled sample input (`sample_reviews_input.csv`), and a 
Python script that uses the model to make predictions (`predict_rating.py`).

## Making a prediction with the model

The scipt in the `model_prediction/` directory, `predict_rating.py`, can be run 
to make a prediction on the included sample data.

The script takes two arguments: the name of the input file and a name for the 
output file. You can use it on the included sample data, like this:

`python3 predict_rating.py sample_reviews_input.csv output.csv`

Where `output.csv` holds sentiment predictions that the model makes for each 
review.
