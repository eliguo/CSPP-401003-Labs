# CSPP-401003-Labs

## Lab 1: MLP for Regression

- The data is divided into 2 files called `california_housing_train.csv` and `california_housing_test.csv` - one for training and the other for testing. Use this to form your train and test predictor and response variables.
- You need to model a feed-forward neural network which will predict the *median_house_value* given the predictors.
- You will have to perform pre-processsing steps such as remove missing data, standardization, removing correlated predictors.
- Once you have completed this come up with a model architecture with your choice of number of hidden layers, number of neurons in each layer and activation functions. Remember that this is a regression problem and your choices have to be in line with it.
- Compile and fit the model on the train data with an appropriate validation split. Make plots of the train and validation loss for visual inspection.
- Predict the results on the test data and compute the test MSE.

## Lab 2: Neural Networks with Functional API
In this lab, we will use the `real_estate` dataset that is given to you as `real_estate_1.csv` and `real_estate_2.csv`. This dataset provides a number of features that eventually lead to predicting the prices of the houses. To demonstrate that the functional API in Keras can take multiple inputs we have divided the file dataset into two CSV files. Perform regression using MLP on the dataset with the help of [functional API](https://www.tensorflow.org/guide/keras/functional).

- Read both the datasets into two separate dataframes `df_1` and `df_2`
- Assign the predictor variables to two separate variables `X_1` and `X_2` from `df_1` and `df_2` respectively. Assign the response variable 'Y house price of unit area' to a single variable `y`
- Split the data into train and validation sets with 70% of the data for training and 30 for testing. Set `random_state` as 40. Remember to maintain separate variables - `X_1_train, X_1_test,X_2_train, X_2_test, y_train, y_test`
- Use the functional API to define the inputs to the model using Keras.Input
- Concatenate the input layers and pass them through dense layers and then an output layer using `keras.layers.concatenate()`
- Define your model using `Keras.Model` with multiple inputs and a single output
- Use mean squared error as the evaluation metric to compile the model
- Fit the model with a validation split of 20%
- Plot the training and validation loss as a function of the number of epochs used to train
- Print the train and validation MSE

Your aim is to get the lowest MSE. For this:
- Try changing the number of hidden layers
- Try changing the number of nodes in the hidden layers
- Number of epochs
- Even the predictors - create new ones, remove the existing ones
