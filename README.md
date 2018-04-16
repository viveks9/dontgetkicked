# Dontgetkicked
Don't Get Kicked from Kaggle Competition

Model Code: wide_deep directory -- based on Tenseflow tutorial on building linear classifier:
https://www.tensorflow.org/tutorials/wide_and_deep

Submission: submission directory -- submission as described in kaggle

Data: data directory -- all the clean data used for submission


Overall notes:

1. Initial data has lot of missing fields, replaced (imputed) with mean of the column where possible
2. Used a logistic regression based on both linear and DNN classifier as outlined in the tutorial
3. Data was heavily imbalanced (only 8% kicked cars)
4. Augmented the data using small changes to kicked cars info and adding those rows back to original dataset
5. Tried combination of fields like month of purchase, combination of retail / auction prices -- need to look into it more
6. Got an accuracy of 75% (against baseline of 63%)
7. Final submission is in to_submit.csv.
8. You can run new predictions by using the code in wide_deep_cars_predict.py --model=wide_deep (you will need the model which I am uploading as a tarball as well)




