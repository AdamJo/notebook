# GLOSSARY

##### class labels
Discrete, unordered values that can be understood as the group memberships of the instances.

##### multi-class classification
Has more than 2 [class labels](#class-labels-classification)

##### binary classification
Where the machine learning algorithm learns a set of rules in order to distinguish between two possible classes : spam and non-spam email.

##### regression analysis
We are given a number of predictor (explanatory) variables and a continuous response variable (outcome), and we try to find a relationship between those variables that allows us to predict an outcome.

##### clustering
Is an exploratory data analysis technique that allows us to organize a pile of information into meaningful subgroups (clusters) without having any prior knowledge of their [group memberships](#group-memberships).
also known as *unsupervised classification*

##### unsupervised learning
Dealing with unlabeled data or data of unknown structure.

##### supervised learning
To learn a model from **labeled training data** that will let us make a prediction about unseen or future data.

##### classification
To predict the categorical [class labels] of new instances based on past observations.

##### dimensionality reduction
Reduce the amount *noise data* which can degrade the predictive performance of certain algorithms, and compress the data onto a smaller dimensional subspace while retaining most of the relevant information.

##### group memberships
The labeled data within a data set.

##### zero mean
Noise data does not present a net disturbance to the system.

##### unit variance
measures how far a set of numbers are spread out

##### highly correlated
When a independent variable changes, so does the dependent variable in some positive or negative relationship.

##### generalization performance (aka generalization error || out-of-sample error)
A measure of how accurately an algorithm is able to predict outcome values for previously unseen data.

##### optimization techniques
Finding the minimizer of a function subject to constraints
  * EXAMPLE : Stock market.
    - Minimize variance of return subject to getting at least $50.
    - source : https://www.cs.berkeley.edu/~jordan/courses/294-fall09/lectures/optimization/slides.pdf

##### hyperparameter
parameters that are not learned from the data but represent the knobs of a model that we can turn to improve its performance.

##### perceptron
* a Perceiving and Recognizing Automaton
* is an algorithm for supervised learning of binary classifiers: functions that can decide whether an input (represented by a vector of numbers) belongs to one class or another.

##### activation function
* &empty;(z)
  - this takes a linear combination of certain input values x and a corresponding weight vector w, where z is the co-called net input (z = w<sub>1</sub>x<sub>1</sub> + ... + w<sub>m</sub>x<sub>m</sub>)

##### epochs
maximum number of passes over the training dataset

##### unit step function
* usually denoted by H (but sometimes u or θ), is a discontinuous function whose value is zero for negative argument and one for positive argument.
* EXAMPLE :
 * &empty;(z) = (1 if z &ge; &theta;, -1 otherwise)

##### heaviside step function
see unit step function

##### One-vs.-All (OVA)
aka One-vs.-Rest (OvR) : Train one classier per class. Then treat it as +1 and the rest of the data as -1.  Using a new data sample we would use our *n* classifiers, where *n* is the number of class labels, and assign the class label the the highest confidence to particular sample.

##### Adaptive Linear Neuron
ADAptive LInear NEuron (Adaline)

##### quantizer
block passes its input signal through a stair-step function so that many neighboring points on the input axis are mapped to one point on the output axis. The effect is to quantize a smooth signal into a stair-step output.

##### objective function
* key ingredient of supervised machine learning algorithms is to define an objective function to be optimized during the learning process.
 - in this case it is a cost function that normally gets minimized.

##### cost function
A cost function lets us figure out how to fit the best straight line to our data

##### gradient descent
aka ( batch gradient descent )
is a first-order optimization algorithm. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point.

##### mini-batch learning
compromise between batch and stochastic gradient descent is mini-batch learning-rate
apply batch gradient descent to smaller subsets of the training data

##### stochastic gradient descent
aka SGD
is a gradient descent optimization method for minimizing an objective function that is written as a sum of differentiable functions.

##### Feature Scaling
method used to standardize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.

##### sum of squared errors
In statistics, the residual sum of squares (RSS), also known as the sum of squared residuals (SSR) or the sum of squared errors of prediction (SSE), is the sum of the squares of residuals (deviations of predicted from actual empirical values of data).

##### cross-validation
is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice

##### variance
measures how far a set of numbers are spread out

##### bias
if it is calculated in such a way that it is only systematically different from the population parameter of interest

##### hyperplane
The decision boundary.  Area of space that separating the training data.

##### margin
distance between the separating hyperplane ( decision boundary )

##### support vectors
samples closest to the hyperplane

##### Gini impurity
measure of how often a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset

##### entropy
Is a measure of unpredictability of information content.

##### mean imputation
replace the missing value by the mean values of the entire feature column.

##### overfitting ( high variance )
means that model fits the parameters too closely to the particular observations in the training dataset but does not generalize well to real data.

##### regularization
adding a penalty term to the cost function to encourage smaller weights  ( penalize large weights )

##### covariance matrix
covariance matrix (also known as dispersion matrix or variance–covariance matrix) is a matrix whose element in the i, j position is the covariance between the i th and j th elements of a random vector. A random vector is a random variable with multiple dimensions.
https://en.wikipedia.org/wiki/Covariance_matrix

##### eigenvectors
square matrix is a vector that does not change its direction under the associated linear transformation. In other words—if v is a vector that is not zero, then it is an eigenvector of a square matrix A if Av is a scalar multiple of v.
https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors

##### eigenvalues
a scalar known as the eigenvalue or characteristic value associated with the eigenvector v
https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors

##### hyperparameter
optimal values of tuning parameters
Example : regularization in logistic regression / depth parameter of a decision tree

##### confusion matrix
is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one (in unsupervised learning it is usually called a matching matrix)
https://en.wikipedia.org/wiki/Confusion_matrix

##### polarity
* the presence or manifestation of two opposite or contrasting principles or tendencies.
* Linguistics.
 - (of words, phrases, or sentences) positive or negative character.

##### clustering (clustering-analysis)
is a technique that allows us to find groups of similar objects, objects that are more related to each other than to objects in other groups.
  * grouping movies, music, and documents by topic
  * finding customers that share similar interests based on common purchase behaviors as a basis for recommendations engines.
