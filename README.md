# Introduction

This is a team project I did as part of my MSc. My teammates and their LinkedIn profiles are as follows:

- Peter Devine (https://uk.linkedin.com/in/peter-devine-030b2012a)
- Agata Goszczyńska (https://uk.linkedin.com/in/agatagoszczynska/en)
- Madalina Lupu (https://uk.linkedin.com/in/madalinalupu)

# Problem exposition, motivation and understanding

The business problem we aim to solve in this notebook is related to predicting the electricity usage of 370 clients, on behalf 
of an electricity provider. This is from the "UCI Electricity Load" dataset, in which the data stretches from January 1st 2011 
to January 1st 2015.

We hypothesise that our data supplied is linked to time in some way. In other words, we think that there is some link between 
the current electricity usage and the usage of the recent past. We hope to show this through our models - that we can predict 
future usage more accurately than a naive estimate using previous data.

# Approach and Insights

We start by applying a linear model to this data before moving onto reforming the task as a supervised learning one and using
an ensemble and neural network-based approach. With this approach, we aim to find a model that is proficient at predicting the 
next value in this time series. 

In the linear model stage, we use a naive forecast or persistence model to establish a baseline performance. Then we conduct 
different analyses including differencing, using tests for stationarity, autocorrelation plots and examining residuals to
decide upon the parameters we use in our ARIMA model.

Having done this, we re-formulate the problem to be a supervised learning task. In this step, we use a multitude of different
methods/approaches including:

- ensemble
- bagging and boosting 
- Long-short Term Memory networks (LSTM) model and specialised non-linear algorithms
- Facebook Prophet

Out of all these models, we choose the best data and evaluate its performance further.

Since these data are very numerous, we have decided to make our dataset smaller by only focusing on one client for our 
validation stage level of model selection. We randomly chose this client to be client MT_237. For our final model evaluation, 
we will try our model on several different clients.

# Results

We succeed in our goal of considerably improving upon the performance of our naive forecast. Our Gradient Boosting algorithm 
has the consistently best results, closely followed by the Random Forest algorithm. Gradient Boosting, Random Forest, K-nearest
neighbours and AdaBoost have performances higher than that of the naive forecast, whilst Ridge Regression's performance is almost 
the same as that of the naive forecast.

On the other hand, our LSTM has good performance, almost as good as that of Gradient Boosting, and just a bit better than 
Random Forests. 

On training data, the naive forecast RMSE is 541.8, while the mean RMSE across 3 folds of Gradient Boosting is 486.4. 

On test data, the naive forecast RMSE is 599.7, while the mean RMSE across 3 folds of Gradient Boosting is __. 

Due to time constraints, the hyperparameters for all models bar Gradient Boosting chosen by trial and error to see which ones 
afforded the best results. Since Gradient Boosting was our best performing model on training data, we conducted a grid search 
before.

If we were being more rigorous, we would have done grid search over each model for each split to find the optimal
configuration of hyperparameters. Moreover, further experimentation with architectures and implementations of RNNS (E.g. using a GRU instead of an LSTM) could lead to both
speed and accuracy boosts for this network.

# How to run this project

- Download the Jupyter notebook
- Download the data from https://drive.google.com/open?id=1VwRHhWiTu2_On6SWD9vqUZQ9uddlJuwf. The file size was too large to be
present in this repository
- Change path to load the data from your local machine
- Pip install Keras and FBProphet (as shown in the notebook)
- Import all required libraries (as shown in the notebook)
- Run the whole Jupyter Notebook

Some of the analysis, especially regarding FBProphet, is computational resource-heavy and may require some time to complete. 
The code regarding the FBProphet analysis can be commented out as it is not one of the better-performing methods.
