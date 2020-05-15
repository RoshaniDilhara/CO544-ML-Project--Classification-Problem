Group 2 


1. CO544-Project-Final-Result.py - This is the source code that we used to get our final predictions.
Handling missing values : By using an imputation method called Regression Imputation
Applied classification models :Logistic Regression, Decision Tree, Decision Tree - maximum depth 3, Decision Tree - maximum depth 4, k-neighbours, Linear Discriminant analysis, gaussian naive bayes, support vector machine(svm), Random forest
Selected model : SVM
Final accuracy : 95.652% (SVM)
Other best accuracies : K-NN model - 94.202%


2. CO544-Project-MeanMode.py -
Handling missing values : By imputing the attribute mean and mode for all missing values
Applied classification models :Logistic Regression, Decision Tree, Decision Tree - maximum depth 3, Decision Tree - maximum depth 4, k-neighbours, Linear Discriminant analysis, gaussian naive bayes, support vector machine(svm), Random forest
Highest accuracy :  K-NN model - 95.652% 
Other best accuracies : Random Forest Classifier model - 92.753%




3. CO544-Project-xgboost.py - 
Applied classification model : Xgboost (or XGBClassifier)
Handling missing values : Replace the missing values with numpy.nan and let the classifier handle the missing values. It calculates imputations and find the best imputation for each missing value
Final accuracy : 88.405%


4.  CO544-Project.py - 
Handling missing values : Replace numeric missing values with “0” and others as an unique value for that nominal attribute
Applied classification models : Logistic Regression, Decision Tree Classifier, K-Neighbors Classifier, Linear Discriminant Analysis, GaussianNB and Support Vector machine
Highest test accuracy model : Support Vector machine
Final accuracy : 95.652%


5. drop_missing_lda.py -
Missing values handled by : Dropping rows with missing values
Categorical data handled by : label encoding for attribute ‘A1’ and one hot encoding for other attributes with categorical data
Classification models tried : Logistic Regression, Decision Tree, Decision Tree - maximum depth 3, Decision Tree - maximum depth 4, k-neighbours, Linear Discriminant analysis, gaussian naive bayes, support vector machine(svm), Random forest, AdaBoost, Quadratic discriminant analysis, Gaussian process classifier
Selected models (models with high accuracy for test dataset): Linear Discriminant Analysis  
Final accuracy : 94.202%


6. ML_project_gpc.py - 
Missing values handled by : Mean for numerical data and mode for categorical data
Categorical data handled by : label encoding for attribute ‘A1’ and one hot encoding for other attributes with categorical data
Classification models tried : Logistic Regression, Decision Tree, Decision Tree - maximum depth 3, Decision Tree - maximum depth 4, k-neighbours, Linear Discriminant analysis, gaussian naive bayes, support vector machine(svm), Random forest, AdaBoost, Quadratic discriminant analysis, Gaussian process classifier
Selected models (models with high accuracy for test dataset): Gaussian Process classifier
Final accuracy : 86.956%


7. ML_project_gpc_and_svm.py - 
Missing values handled by : Replace missing values with numpy.nan
Categorical data handled by : label encoding for attribute ‘A1’ and one hot encoding for other attributes with categorical data
Classification models tried : Logistic Regression, Decision Tree, Decision Tree - maximum depth 3, Decision Tree - maximum depth 4, k-neighbours, Linear Discriminant analysis, gaussian naive bayes, support vector machine(svm), Random forest, AdaBoost, Quadratic discriminant analysis, Gaussian process classifier
Selected models (models with high accuracy for test dataset): Gaussian Process classifier and Support vector machine
Final accuracy : 89.855%