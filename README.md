# credit_risk_classification

This challenge involved making logistic regression models to identify the creditworthiness of borrowers.

## Overview of the Analysis

The purpose of this analysis was to build a model to predict whether loans in a given dataset should be labeled as healthy loans or high-risk ones.

The dataset was from a peer-to-peer lending services company in the form of a csv file that contained 77,536 rows of borrower information across eight columns of data: loan size(amount), interest rate, borrower income, debt-to-income ratio, number of accounts (with the lending company), derogatory marks, total debt, and loan status.  

To predict which of those loans should be labeled healthy and which should be labeled high-risk, several stages of the machine learning process were followed, and two models built for making the predictions.

Using a Jupyter notebook, the data file was read and placed into a dataframe, then split into training and testing sets, employing the train_test_split module from sklearn.

With the data split into training and testing sets, an initial logistic regression model was then instantiated and fitted onto the training data. Predictions were subsequently made on the testing data using the .predict() operator. Evaluation of the model's results was then performed by generating a balanced accuracy score, a confusion matrix, and a classification report.

In like manner, a second regression model was made, using resampled training data. To resample the data, the RandomOverSampler (ROS) was imported from imbleearn. An ROS model was instantiated using the ROS, and the original training data fitted to the ROS model. Distinct values of the resampled data were checked, and an equal number of healthy and high-risk loan labels confirmed. 

From there, the LogisticRegression classifier was used to instantiate a new regression model that was fitted onto the resampled data. Predictions were made with that model, and again evaluation of the model's results performed by generating a balanced accuracy score, a confusion matrix, and a classification report.

## Results

* Machine Learning Model 1:
  * Accuracy score - 99%
  * Precision scores - 100% (healthy loans), 84% (high-risk loans)
  * Recall scores - 99% (healthy loans), 94% (high-risk loans)
  
* Machine Learning Model 2:
  * Accuracy score - 99%
  * Precision scores - 100% (healthy loans), 84% (high-risk loans)
  * Recall scores - 99% (healthy loans), 99% (high-risk loans)

## Summary

The classification report for the first model revealed a substantial imbalance in the number of data points supporting the test data for healthy loans versus high-risk loans, setting an initial expectation that the scores for healthy loan labels would be higher than those for high-risk labels based on the divergence between the two in terms of observation points.

That was further evidenced by the lower precision and f1 scores, in particular, for high-risk loans. Whereas those scores for healthy loans were 100%, they were 84% and 89%, respectively, for high-risk loans. Still, the recall score that can be looked to for stating that something will be as labeled, was 94% for high-risk loans - and accordingly, the f1 score, balancing the precision and recall scores,  was brought up to 89%.

So, although the scores for high-risk loans were lower than their counterparts for healthy loans, the scores for high-risk loans were still well above 80%, indicating that the data collected was strong. And that was also further confirmed by the weighted average scores (presenting the precision, recall, and f1 scores as if there were no imbalance) of 99%.

Accordingly, then, the first logistic regression model predicted fairly well both healthy and high-risk loan labels from the original dataset.

The classification report for the second model, produced after resampling the training data to equate the number of high-risk loan labels with healthy ones, showed higher recall and f1 scores for high-risk loan labels. Whereas use of the original data resulted in recall and f1 scores of 94% and 89%, respectively, for high-risk loan labels, use of resampled data resulted in corresponding scores of 99% and 91%. And complementing those higher scores were like ones for the macro averages as well: 99% for recall, and 95% for f1.

So, the second logistic regression model, fit with oversampled data, predicted with even higher accuracy high-risk loan labels.

In summary, each of the models had an accuracy score of 99%. And both models resulted in the same precision score for high-risk loan labels: 84%. But the second model produced a higher recall score for high-risk loans: 99% versus 94%. Accordingly, although the first model was a good predictor, the second model was even better, especially considering the importance of being able to predict high-risk loan labels well.
