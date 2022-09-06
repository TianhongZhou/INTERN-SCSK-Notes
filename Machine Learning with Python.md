# Machine Learning with Python

## Machine Learning

### What is machine learning?

- Machine learning is the subfield of computer science that gives "computers the ability to learn without being explicitly programmed"

- AI process loop

  - observe -> plan -> optimize -> action -> learn and adapt

    ​     ^------------------------------------------------------v

  - observe -- identify patterns using the data

  - plan -- find all possible solutions

  - optimize -- find optimal solution from the list of possible solutions

  - action -- execute the optimal solution

  - learn and adapt -- is the result giving expected result, if no adapt

### Types of Machine Learning

- Regression -- Predicting continuous values
- Classification -- Predicting the item class / category of a case
- Clustering -- Finding the structure of data; summarization
- Associations -- Associating frequent co-occurring items/events
- Anomaly detection -- Discovering abnormal and unusual cases
- Sequence mining -- Predicting next events; client-stream (Markov Model, HMM)
- Dimension Reduction -- Reducing the size of data (PCA)
- Recommendation systems -- Recommending items

### Supervised Learning

- Classification
  - Classifies labeled data
- Regression
  - Predicts trends using previous labeled data
- Building supervised learning machine learning models has three stages:
  - Training: The algorithm will be provided with historical input data with the mapped output. The algorithm will learn the patterns within the input data for each output and represent that as a statistical equation, which is also commonly known as a model
  - Testing or validation: In this phase the performance of the trained model is evaluated, usually by applying it on a dataset (that was not used as part of the training) to predict the class or event
  - Prediction: Here we apply the trained model to a data set that was not part of either the training or testing. The prediction will be used to drive business decisions

### Unsupervised Learning

- Dimension reduction
- Density estimation
- Market basket analysis
- Clustering
  - Finds patterns and groupings from unlabeled data

### Python libraries for machine learning

- NumPy -- Math library to work with n-dimensional arrays
- SciPy -- Collection of numerical algorithms and domain-specific toolboxes (signal processing, optimization, statistics)
- Matplotlib -- 2D / 3D plotting package
- Pandas -- High-level library for data importing, manipulation and analysis (numerical tables, time series)
- Scikit-learn -- Collection of algorithms and tools for ML



## Regression

### Variable

- Independent variables x
- Dependent variables y

### Types of regression models

- Simple Regression
  - Simple Linear Regression
  - Simple Non-linear Regression
- Multiple Regression
  - Multiple Linear Regression
  - Multiple Non-linear Regression
- Ordinal regression
- Poisson regression
- Fast forest quantile regression
- Linear, Polynomial, Lasso, Stepwise, Ridge regression
- Bayesian linear regression
- Neural network regression
- Decision forest regression
- Boosted decision tree regression
- KNN (K-nearest neighbors)



## Classification

### What is classification

- A supervised learning approach
- Categorizing some unknown items into a discrete set of categories or "classes"
- The target attribute is a categorical variable

### Classification algorithms

- Decision Trees
- Naive Bayes
- Linear Discriminant Analysis
- K-Nearest Neighbor
- Logistic Regression
- Neural Networks
- Support Vector Machines (SVM)

### K-Nearest Neighbor (KNN)

- What is KNN?
  - A method for classifying cases based on their similarity to other cases
  - Cases that are near each other are said to be "neighbors"
  - Based on cimilar cases with same class labels are near each other
- Algorithm
  - Pick a value for K
  - Calculate the distance of unknown case from all cases
  - Select the K-observations in the training data that are "nearest" to the unknown data point
  - Predict the response of the unknown data point using the most poplar response value from the K-nearest neighbors
- How to select K?
  - Increase K, test the training set, until accuracy reach the summit

### Evaluation Metrics - Accuracy

- Jaccard index -- Jaccard similarity coefficient
  $$
  J(y,\hat{y})=\frac{|y\cap \hat{y}|}{|y\cup\hat{y}|}=\frac{|y\cap \hat{y}|}{|y|+|\hat{y}|-|y\cap\hat{y}|}
  $$

- F1-score -- Confusion matrix
  $$
  P(Precision) = \frac{TP}{TP+FP} \\
  R(Recall) = \frac{TP}{TP+FN}\\
  F_1score=\frac{1}{\frac{1}{2}(\frac{1}{P}+\frac{1}{R})}=\frac{2PR}{P+R}
  $$

- Log loss (logarithmic loss) -- Measure the performance of classifier where the predicted output is a probability value (0~1)
  $$
  y\cdot \log(\hat{y}) + (1-y)\cdot \log(1-\hat{y})\\
  LogLoss = -\frac{1}{m}\sum^{m}_{i=0}(y_i\cdot \log(\hat{y_i})+(1-y_i)\cdot \log(1-\hat{y_i}))
  $$

### Decision tree

- What is decision tree?

  - The basic intuition behind a decision tree is to map out all possible decision paths in the form of a tree

- Algorithm

  - Choose an attribute from your dataset
  - Calculate the significance of attribute in splitting of data
  - Split data based on the value of the best attribute
  - Go to step 1

- p(A) means the percentage of A/all

  p(B) means the percentage of B/all
  $$
  \text{Entropy} = -p(A) \cdot \log(p(A)) - p(B) \cdot \log(p(B))
  $$
  When there is only A or B in the set, Entropy = 0; When there are half A and B, Entropy = 1

- p1, p2, p3 are the percentage of each part

  E(p1), E(p2), E(p3) are the entropy of each part
  $$
  \text{Information Gain}=(\text{Entropy before split}) - (\text{weighed entropy after split}) \\
  \qquad \qquad \qquad \qquad \qquad \quad=(\text{Entropy before split}) - [p_1 \cdot E(p_1) + p_2 \cdot E(p_2) + p_3 \cdot E(p_3)]
  $$

### Logistic Regression

- What is Logistic Regression?

  - A classification algorithm for categorical variables

- Use cases

  - Predicting the probability of a person having a heart attack
  - Predicting the mortality in injured patients
  - Predicting a customer's propensity to purchase a product or halt a subscription
  - Predicting the probability of failure of a given process or product
  - Predicting the likelihood of a homeowner defaulting on a mortgage

- When is suitable?

  - If data is binary
  - If need probabilistic results
  - When need a linear decision boundary
  - If need to understand the impact of a feature

- Logistic (Sigmoid) function
  $$
  \sigma(\theta^TX)=\frac{1}{1+e^{-\theta^TX}}
  $$

- What is the output of this model?

  - P(y=1|X)
  - P(y=0|X) = 1-P(y=1|X)

- Training process

  - Initialize theta

  - Calculate 
    $$
    \hat{y}=\sigma(\theta^TX)
    $$
    for a customer

  - Compare the output of y hat with actual output of customer, y, and record it as error

  - Calculate the error for all customers

  - Change the theta to reduce the cose

  - Go back to step 2

### Cost function

- Change the weight -> Reduce the cost

- Cost function
  $$
  Cost(\hat{y},y)=\hat{y}-y=\sigma(\theta^TX)-y \\
  Cost(\hat{y},y)=\frac{1}{2}(\sigma(\theta^TX)-y)^2 \\
  J(\theta) = \frac{1}{m}\sum^{m}_{i=1}Cost(\hat{y}, y)=\frac{1}{2m}\sum^{m}_{i=1}(\sigma(\theta^TX)-y)^2
  $$

- Alternative Cost function
  $$
  J(\theta)=-\frac{1}{m}\sum^m_{i=1}y\cdot\log(\hat{y})+(1-y)\cdot\log(1-\hat{y})\quad\begin{equation*}
       \begin{cases}
           -\log(\hat{y}) \qquad \qquad \text{if} \ \ y=1 \\
           -\log(1-\hat{y}) \qquad \ \text{if} \ \ y=0
       \end{cases}
   \end{equation*}
  $$

- How to find the best parameters for out model?

  - Minimize the cost function

- How to minimize the cost funtion?

  - Using Gradient Descent

### Gradient descent

- Cost function
  $$
  J(\theta)=-\frac{1}{m}\sum^m_{i=1}y\cdot\log(\hat{y})+(1-y)\cdot\log(1-\hat{y}) \\
  \hat{y}=\sigma(\theta_1x_1+\theta_2x_2)
  $$

- Calculate partial derivative
  $$
  \frac{\partial J}{\partial \theta_1}=-\frac{1}{m}\sum^m_{i=1}(y-\hat{y})\hat{y}(1-\hat{y})x_1 \\
  \nabla J = \begin{bmatrix} \frac{\partial J}{\partial\theta_1} \\ \vdots \\ \frac{\partial J}{\partial \theta_n}\end{bmatrix} \\
  \theta_{new}=\theta_{old}-\eta\cdot\nabla J
  $$

### Training algorithm recap

- Initialize the parameters randomly
  $$
  \theta^T=[\theta_0,\theta_1,\theta_2,\cdots]
  $$

- Feed the cost function with training set, and calculate the error
  $$
  J(\theta)=-\frac{1}{m}\sum^m_{i=1}y\cdot\log(\hat{y})+(1-y)\cdot\log(1-\hat{y})
  $$

- Calculate the gradient of cost function
  $$
  \nabla J=\begin{bmatrix} \frac{\partial y}{\partial \theta_1},\frac{\partial y}{\partial \theta_2},\cdots,\frac{\partial y}{\partial \theta_k}\end{bmatrix}
  $$

- Go to step 2 until cost is small enough
  $$
  \theta_{new}=\theta_{old}-\eta\cdot\nabla J
  $$

- Predict the new customer X
  $$
  P(y=1|X)=\sigma(\theta^TX)
  $$

### Support Vector Machines (SVM)

- What is SVM?

  - SVM is a supervised algorithm that classifies cases by finding a separator
  - 1. Mapping data to high-dimensional feature space
    2. Finding a separator

- Map the sample data to a high-dimensional space, find the segmentation hyperplane, and map it back (may become a curve)

- Mapping: Data transformation

  - Kernel function, low dimension to high dimension mapping function, could be
    - Linear
    - Polynomial
    - Radial basis function (RBF)
    - Sigmoid

- Segmentation:

  - Find w and b such that
    $$
    \Phi(\mathrm{w}=\frac{1}{2}\mathrm{w}^T\mathrm{w})
    $$
    is minimized; and for all
    $$
    \{(x_i,y_i)\}:y_i(\mathrm{w}^Tx_i+b)\geq 1
    $$

- SVM use cases

  - Pros and cons
    - Advantages
      - Accurate in high-dimensional spaces
      - Memory efficient
    - Disadvantages
      - Prone to over-fitting
      - No probability estimation
      - Not very efficient if your dataset is large
  - Use Cases
    - Image recognition
      - Image classification
      - Handwritten digits recognition
    - Text mining tasks
      - Text category assignment
    - Detecting spam
    - Sentiment analysis
    - Gene Expression Classification
    - Regression, outlier detection and clustering



## Clustering

### What is clustering?

- Clustering means finding clusters in a dataset, unsupervised
- Cluster -- A group of objects that are similar to other objects in the cluster, and dissimilar to data points in other clusters

### Use cases

- Retail / Marketing
  - Identifying buying patterns of customers
  - Recommending new books or movies to new customers
- Banking
  - Fraud detection in credit card use
  - Identifying clusters of customers
- Insurance
  - Fraud detection in claims analysis
  - Insurance risk of customers
- Publication
  - Auto-categorizing news based on their content
  - Recommending similar news articles
- Medicine
  - Characterizing patient behavior
- Biology
  - Clustering genetic markers to identify family ties

### Why Clustering?

- Exploratory Data Analysis
- Summary generation
- Outliers detection
- Finding duplicates
- Pre-processing step

### Algorithms

- Partitioned-based Clustering
  - Relatively efficient (large & medium datasets)
  - eg: k-Means, k-Median, Fuzzy c-Means
- Hierarchical Clustering (small dataset size)
  - Produces trees of clusters
  - eg: Agglomerative, Divisive
- Density-based Clustering
  - Produces arbitrary  shaped cluters
  - eg: DBSCAN

### K-Means Clustering