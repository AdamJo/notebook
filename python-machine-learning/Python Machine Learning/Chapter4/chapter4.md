# Chapter 4
## This chapter will cover the following topics

* Removing and imputing missing values from the dataset
* Getting categorical data into shape for machine learning algorithm
* Selecting relevant features for the model construction

##### Dealing with missing data
* multiple ways to go about This
* use isnull() function to check for nulls
* DataFrame.isnull().sum() return the cells that are null
  - drop the row containing the empty class
    - DataFrame.dropna()
  - drop the column containing the empty class
  - * DataFrame.dropna(axis=1)
* See chapter code for more examples

##### imputing missing values
* mean imputation : replace the missing value by the mean values of the entire feature column.
* fit method : is used to learn the parameters from the training data
* transform method : uses those parameters to transform the data.
  - data arrays must be the same for the transform as the fit
* predict method in supervised learning tasks provide additional class labels for fitting the model.
* categorical data
 - Ordinal : features can be understood as categorical values that can be sorted or ordered
  - T-shirt size would be an ordinal feature ; XL > L > M
 - nominal features don't imply any order and, to continue with the previous example
  - T-shirt color since red cannot be larger then blue
* provide class labels as integer arrays to avoid technical glitches.
* one-hot-encoding : used to fix the problem of using numbers as replacements for strings
  - create a dummy feature for each unique value in the nominal feature column.
  - shirt colors
    - convert each color into new features
    - blue, green, red = blue=1, green=0, red=0

| price |	size | price | color_blue |	color_green |	color_red |
| ----- | ----- | ----- | ----- | ----- |
| 0 | 10.1 | 1 | 0 | 1 | 0 |
| 1	| 13.5 | 2 | 0 |0 | 1 |
| 2	| 15.3 | 3 | 1 | 0 | 0 |

##### Seature Scaling
* normalization : rescaling of the features to a range of [0, 1]
  * special case of min-max Scaling
  * apply min-max scaling to each feature column
    - x<sup>(i)</sup><sub>norm</sub> = x<sup>(i)</sup> - x<sub>min</sub> / x<sub>max</sub> - x<sub>min</sub>
    - x<sup>(i)</sup> : is a particular sample
    - x<sub>min</sub> : smallest value in a feature column
    - x<sub>max</sub> : largest value
  * used when we need values in a bounded interval
  * scales data to a limited range of values
* standardization
  - center the feature columns at mean 0 with standard deviation 1 so that the feature columns take the form of normal distribution
  - pro : maintains useful information about outliers and makes algorithms less sensitive to them in contrast to min-max scaling
  - x<sup>(i)</sup><sub>std</sub> = x<sup>(i)</sup> - &mu;<sub>x</sub> / &sigma;<sub>x</sub>
    - &mu;<sub>x</sub> is the sample mean of a particular feature column
    - &sigma;<sub>x</sub> corresponding standard deviation

##### Selecting meaningful features
* common solutions to fix overfitting
  - collect more training data
  - Introduce a penalty for complexity via regulation
  - Choose a simpler model with fewer parameters
  - Reduce the dimensionality of the data

##### Sparse solutions with L1 regulation
* L2 regulation is one approach to reduce the complexity of a model by penalizing large individual weights
* difference between L1 & L2
  - L2 : ||w||<sup>2</sup><sub>2</sub> = &sum;<sup>m</sup><sub>j=1</sub>w<sup>2</sup><sub>j</sub>
    * L2 shape is normally plotted as a circle
  - L1 : ||w||<sub>1</sub> = &sum;<sup>m</sup><sub>j=1</sub>|w<sub>j</sub>|
    * replace the square of the weights by the sum of the absolute values of the weights.
    * L1 shape is normally plotted as a diamond
* L1 regularization yields sparse feature vectors; most feature weights will be zero
  * sparsity is useful with high-dimensional datasets with many features that are irrelevant.
    - especially when having more irrelevant dimensions than samples
  * **a technique for feature selection**
* goal is to find the combination of weight coefficients that minimize the cost function for the training data.
* regularization : adding a penalty term to the cost function to encourage smaller weights  ( penalize large weights )
* since L1 is more likely to land on an axis it will result in a more sparce result
* zero out irrelevant features via logistic regression

##### Sequential feature selection algorithms
* reduce complexity of the model and avoid overfitting is [dimensionality reduction] via feature selection
  - good for unregularized models
* 2 main categories of dimensionality reduction
  - feature selection : select subset of original features
  - feature extraction : derive information from the feature set to construct a new feature subspace
* Sequential feature selection algorithms are a family of greedy search algorithms
  - used to reduce an initial dimensionality feature space to a k-dimensional feature subspace where k < d.
* goal : to automatically select a subset of features that are most relevant to the problem to improve computational efficiency or reduce the generalization error of the model by removing irrelevant features or Noise
  * useful for algorithms that don't support regularization
* Sequential Backwards Selection (SBS) (algorithm)
  - aims to reduce dimensionality of the initial feature subspace
    - using minimum decay in performance  of the classifier to improve upon computational efficiency.
    - help improve the predictive power of a model if a model suffers from overfitting

  - removes features from the full feature subset until the new feature subset contains the desired number of features.
  - algorithm in 4 steps :
    1. Initialize the algorithm with k=d, where d is the dimensionality of the full feature space X<sub>d</sub>.
    2. Determine the feature x<sup>-</sup> that maximizes the criterion x<sup>-</sup> = argmaxJ(X<sub>k</sub>-x) where x &isin; X<sub>K</sub>
    3. Remove the feature x<sup>-</sup>  from the feature set:
      - X<sub>k</sub> - 1 = X<sub>k</sub> - 1 = X<sub>k</sub> - x<sup>-</sup>, k=k-1
    4. Terminate if k equals the number of desired features, if not, go to step 2.
  * see book for implementation since it is not in scikit
  * this can help find features that we can predict 100% much easier by separating them from others.
##### Assessing feature importance with random forests
* measure feature importance as the average impurity decrease computed from  all decision trees in the forest.
  - without making any assumptions whether our data is linearly separable or not.
* con : two or more features are highly correlated, one feature may be ranked very highly while the information of the other feature(s) may not be fully captured.
  - we don't need to worry about this if we only want the predictive performance of a model rather than the interpretation of feature importance

##### Summary
* before feeding data to machine learning algorithm, we also have to make sure that we encode categorical variables correctly
* L1 regularization to avoid overfitting by reducing the complexity of a model.
* used sequential feature selection to remove irrelevant features ( select meaningful features from a dataset )


[overfitting]: ../GLOSSARY#overfitting
[dimensionality reduction]: ../GLOSSARY#dimensionality-reduction
