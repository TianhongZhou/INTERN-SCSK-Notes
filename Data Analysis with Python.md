# Data Analysis with Python

## Introduction

### Why Data Analysis?

- Plays important roles in
  - Discovering useful information
  - Answering question
  - Predicting future or the unknown
- Python packages for data analysis
  - Scientifics Computing libraries
    - Pandas		- Data structures & tools
    - Numpy         - Arrays & Matrices
    - SciPy           - Integrals, solving differential equations, optimization
  - Visualization libraries
    - Matplotlib     - Plots & graph
    - Seaborn       - Plots: heat maps, time series, violin plots
  - Algorithmic libraries
    - Scikit-learn   - Machine learning: regression, classification
    - Statsmodels - Explore data, estimate statistical models, and perform statistical tests

### Import Data

- Two properties

  - Format
    - csv
    - json
    - xlsx
    - hdf
  - Path
    - Computer
    - Internet

- Print dataframe in Python

  - ```python
    # print whole dataframe
    df
    ```

  - ```python
    # print first n lines
    df.head(n)
    ```

  - ```python
    # print last n lines
    df.tail(n)
    ```

- Adding headers

  - ```python
    headers = ["A", "B", "C"]
    df.columns = headers
    ```

### Exporting Data

- Export to different format

- | Data Format              | Read            | Save          |
  | ------------------------ | --------------- | ------------- |
  | csv                      | pd.read_csv()   | df.to_csv()   |
  | json                     | pd.read_json()  | df.to_json()  |
  | excel                    | pd.read_excel() | df.to_excel() |
  | sql                      | pd.read_sql()   | df.to_sql()   |
  | HDFStore:PyTables (HDF5) | pd.read_hdf()   | df.to_hdf()   |

### Getting Started Analyzing data in Python

- Data types

  - ```python
    df.dtypes()
    ```

- Statistical summary

  - ```python
    # only summary values
    df.describe()
    ```

  - ```python
    # summary all
    df.describe(include="all")
    ```

- Concise summary

  - ```python
    df.info()
    ```



## Data Wrangling

### Pre-processing Data in Python

- Pre-processing data = Data cleaning, Data wrangling

### Simple Dataframe Operation

- Column operation

  - ```python
    df["column_name"]
    ```

### Dealing with Missing Values

- What is missing values?

  - No data value in stored for a variable (feature) in an observation
  - Could be represented as "?", "N/A", 0 or just a blank cell

- How to deal with missing values?

  - Check with the data collection source
  - Drop the missing values
    - Drop the variable (column)
    - Drop the data entry (row)
  - Replace the missing values
    - Replace it with an average of similar data points
    - Replace it by frequency
    - Replace it based on other functions
  - Leave it as missing data

- Drop missing

  - ```python
    # delete rows with NaN, NaT or other missing values
    dropna()
    ```

  - ```python
    # axis=0 means delete rows
    # axis=1 means delete columns
    dropna(subset=["name"], axis=1)
    ```

  - ```python
    # inplace=True means change the original df
    # inplace=False means return a new df, keep the original one the same
    dropna(subset=["name"], axis=0, inplace=True)
    ```

- Replace missing

  - ```python
    # replace NaN value to average
    mean = df["name"].mean()
    df["name"].replace(np.nan, mean)
    ```

  - ```python
    # some data set uses ? to represent missing value, need to convert it into NaN first, then 
    # delete
    df["name"].replace('?', np.nan, inplace=True)
    df.dropna(subset=["name"], axis=0, inplace=True)
    ```

### Data formatting

- Standardize

  - ```python
    df["city"].replace("NY", "New York")
    df["city"].replace("N.Y.", "New York")
    ```

- Units convert

  - ```python
    df["city-mpg"]=235/df["city-mpg"]
    df.rename(columns={"city_mpg": "city-L/100km"}, inplace=True)
    ```

- Check data types

  - ```python
    df.dtypes()
    ```

- Change data types

  - ```python
    df["name"] = df["name"].astype("int")
    ```

### Data Normalization

- Similar value range

- Similar intrinsic influence on analytical model

- Simple feature scaling

  - $$
    x_{new} = \frac{x_{old}}{x_{max}}
    $$

    ```python
    df["name"] = df["name"] / df["name"].max()
    ```

- Min-Max

  - $$
    x_{new} = \frac{x_{old} - x_{min}}{x_{max} - x_{min}}
    $$

    ```python
    df["name"] = (df["name"]-df["name"].min()) / (df["name"].max()-df["name"].min())
    ```

- Z-score

  - $$
    x_{new} = \frac{x_{old} - \mu}{\sigma}
    $$

    $$
    \mu = \frac{1}{m} \sum^{m}_{i=1}x_i
    $$

    $$
    \sigma = \sqrt{\frac{1}{m}\sum^{m}_{i=1}(x_i-\mu)^2}
    $$

    ```python
    df["name"] = (df["name"]-df["name"].mean()) / df["name"].std()
    ```

### Binning - Categorize

- Concept

  - Binning: Grouping of values into "bins"
  - Converts numerical into categorical variables
  - Group a set of numerical values into a set of "bins"

- Example

  - ```python
    binwidth = int((max(df["price"]) - min(df["price"])) / 4
    bins = range(min(df["price"], max(df["price"], binwidth)
    group_names = ['Low', 'Medium', 'High']
    df["price-binned"] = df.cut(["price"], bins, labels=group_names)
    ```

### Turning Categorical into Quantitative Variables

- Add dummy variables for each unique category

- Assign 0 or 1 in each category

- ```python
  pd.get_dummies[df["name"]]
  ```



## Exploratory Data Analysis

### Descriptive statistics

- Describe basic features of data

- Giving short summaries about the sample and measures of data

- Summarize statistics data using describe()

  ```python
  df.describe()
  ```

- Summarize categorical data using value_counts()

  ```python
  df["name"].value_counts()
  ```

- Seaborn - Box Plots

  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.boxplot(x="drive-wheels", y="price", data=df)
  plt.show()
  ```

- Matplotlib - Scatter Plots

  ```python
  plt.scatter(df["engine-size"], df["price"])
  plt.title ("Scatterplot of Engine size vs. Price")
  plt.xlable("Engine size")
  plt.ylabel("Price")
  plt.show()
  ```

### GroupBy in Python

- Pandas - groupby()

  - Can be applied on categorical variables

  - Group data into categories

  - Single or multiple variables

    ```python
    df_test = df[['drive-wheels', 'body-style', 'price']]
    df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    print(df_grp.sort_values(by='price'))
    ```

- Pandas - Pivot()

  - One variable displayed along the columns and the other variable displayed along the rows

    ```python
    df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
    print(df_pivot)
    ```

- Heatmap

  - Plot target variable over multiple variables

  - ```python
    plt.pcolor(df_pivot, cmap='RdBu')
    plt.colorbar()
    plt.show()
    ```

### Analysis of  Variance (ANOVA)

- Statistic comparison of groups

- Why do we perform ANOVA?

  - Finding correlation between different groups of a categorical variable

- What we obtain from ANOVA?

  - F-test score  --  Variation between sample group means / variation within sample group
  - P-value  --  Confidence degree

- F-test

  - X and Y follow normal distribution and are independent from each other, where
    $$
    X \sim N(\mu_x,\sigma_x^2), \ Y\sim N(\mu_y, \sigma_y^2)
    $$

    $$
    \mu_x = \frac{1}{m} \sum^{m}_{i=1}x_i \qquad
    \mu_y = \frac{1}{m} \sum^{m}_{i=1}y_i \\
    \sigma_x = \sqrt{\frac{1}{m}\sum^{m}_{i=1}(x_i-\mu_x)^2} \qquad
    \sigma_y = \sqrt{\frac{1}{m}\sum^{m}_{i=1}(y_i-\mu_y)^2} \\
    F = \frac{\sigma_x^2}{\sigma_y^2}
    $$

  - The smaller F value is, the smaller influence variable category have toward target variable

  - The larger F value is, the larger influence variable category have toward target variable

  - The smaller F value is, the larger P value is

  - The larger F value is, the smaller P value is

  - Example

    ```python
    from scipy import stats
    df_anova = df[["make", "price"]]
    group_anova = df_anova.groupby(["make"])
    val = stats.f_oneway(group_anova.get_group("honda")["price"], group_anova.get_group("subaru")["price"])
    print(val)
    ----------------------------------------------------------------------------------------------
    F_onewayResult(statistic=0.19744030127462606, pvale=0.6609478240622193)
    
    df_anova = df[["make", "price"]]
    group_anova = df_anova.groupby(["make"])
    val = stats.f_oneway(group_anova.get_group("honda")["price"], group_anova.get_group("jaguar")["price"])
    print(val)
    ----------------------------------------------------------------------------------------------
    F_onewayResult(statistic=400.925870564337, pvalue=1.0586193512077862e-11)
    ```

### Correlation

- What is correlation?

  - Measure to what extent different variables are interdependent
  - Correlation doesn't imply causation

- Linear Relationship

  - Correlation between two features

    ```python
    sns.regplot(x="engine-size", y="price", data=df)
    plt.ylim(0,)
    plt.show()
    ```

- Pearson Correlation

  - Measure the strength of the correlation between two features

    - Correlation coefficient

      - Close to +1			Larger Positive relationship
      - Close to -1             Larger Negative relationship
      - Close to 0              No relationship

    - P-value

      - < 0.001                  Strong certainty in the result
      - < 0.05                    Moderate certainty in the result
      - < 0.1                      Weak certainty in the result
      - 0.1 <                      No certainty in the result

    - Strong Correlation

      - Correlation coefficient close to +1 or -1
      - P-value < 0.001

    - Example

      ```python
      pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
      ------------------------------------------------------------------------------------------
      Correlation Coefficient = 0.8095745670036559
      P-value = 6.36905742825998e-48
      => Strong correlation
      ```

- Heatmap

  - ```python
    # Matplotlib
    df_corr = df.corr()
    
    fig, ax = plt.subplots()
    im = ax.pcolor(df_corr, cmap='RdBu')
    
    row_labels = df_corr.columns
    col_labels = df_corr.columns
    
    ax.set_xticks(np.arange(df_corr.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(df_corr.shape[0]) + 0.5, minor=False)
    
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)
    
    plt.xticks(rotation=90)
    fig.colorbar(im)
    plt.show()
    ```

  - ```python
    # Seaborn
    df_corr = df.corr()
    mask = np.zeros_like(df_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(df_corr, mask=mask, cmap='RdBu')
    plt.show()
    ```



## Model Development

- A model can be thought of as a mathematical equation used to predict a value given one or more other values
- Relating one or more independent variables to dependent variables
- Usually the more relevant data you have, the more accurate your model is

### Simple Linear Regression (SLR)

- The predictor (independent) variable x

- The target (dependent) variable y

- $$
  y=\theta_0+\theta_1x
  $$

- Noise

  - Data will add some noise to fit normal distribution

- Process

  - Calculate parameters according to x and y in training set

  - Calculate predict result
    $$
    \hat{y} = \theta_0+\theta_1x
    $$
    using parameters and x in testing set

  - Calculate error between expectation result and testing result (MAE, MSE, RMSE, RAE, RSE)

- ```python
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression()
  X = df[["highway-mpg"]]
  y = df["price"]
  lm.fit(X, y)
  y_hat = lm.predict(X)
  print(lm.intercept_)
  print(lm.coef_)
  --------------------------------------------------------------------------------------------------
  # theta_0
  38423.305858157386
  # theta_1
  [-821.73337832]
  ```

### Multiple Linear Regression (MLR)

- Two or more predictor (independent) variable x

- One continuous target (dependent) variable y

- $$
  y=\theta_0+\theta_1x_1+\cdots+\theta_nx_n
  $$

- ```python
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression()
  X = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
  y = df["price"]
  lm.fit(X, y)
  y_hat = lm.predict(X)
  print(lm.intercept_)
  print(lm.coef_)
  --------------------------------------------------------------------------------------------------
  # theta_0
  -15806.62462632922
  # theta_1,2,3,4
  [53.49574423, 4.70770099, 81.53026382, 36.05748882]
  ```

### Model Evaluation using Visualization - Regression Plot

- Why use regression plot?

  - It gives us a good estimated of 
    - The relationship between two variables
    - The strength of the correlation
    - The direction of the relationship (positive or negative)

- Regression Plot

  - Shows a combination of
    - The scatterplot: each point represents a data
    - The fitted linear regression line

- ```python
  import seaborn as sns
  sns.regplot(x="highway-mpg", y="price", data=df)
  plt.ylim(0,)
  plt.show()
  ```

### Model Evaluation using Visualization - Residual Plot

- Expected characteristic of residual (if not fit, should not use linear regression)

  - Average 0
  - Same as upper and lower part
  - Random distributed with equal density
  - No trend of curve

- ```python
  import seaborn as sns
  sns.residplot(x="highway-mpg", y="price", data=df)
  plt.show()
  ```

### Model Evaluation using Visualization - Distribution Plot

- Find residual between real data and predict data

- Normalize these two data and compare the fitting of them

- The more fitting they are, the more accurate the model is

- ```python
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression()
  lm.fit(df[["highway-mpg"]], df["price"])
  lm.fit(df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], df["price"])
  y_hat = lm.predict(X)
  
  axl = sns.distplot(y, hist=False, color="r", label="Actual Value")
  sns.distplot(y_hat, hist=False, color="b", label="Fitted Value", ax=axl)
  ```

### Polynomial Regression

- Concept

  - A special case of the general linear regression model
  - Useful for describing curvilinear relationships

- Curvilinear relationships:

  - By squaring or setting higher-order terms of the predictor variables

- Example

  - Quadratic
    $$
    y=\theta_0+\theta_1x_1+\theta_2x_1^2
    $$

  - Cubic
    $$
    y=\theta_0+\theta_1x_1+\theta_2x_1^2+\theta_3x_1^3
    $$

  - Higher order
    $$
    y = \theta_0+\theta_1x_1+\theta_2x_1^2+\cdots+\theta_nx_1^n
    $$

- Process

  - Calculate polynomial

    ```python
    x = df["highway-mpg"]
    y = df["price"]
    f = np.polyfit(x, y, 3)
    p = np.polyld(f)
    print(p)
    ----------------------------------------------------------------------------------------------
    -1.557x^3+204.8x^2-8965x+1.379e+05
    ```

  - Plot

    ```python
    x_new = np.linspace(15, 55, 100)
    y_new = p(x_new)
    plt.plot(x, y, '.', x_new, y_new, '-')
    plt.title("Polynomial Fit with Matplotlib for Price")
    plt.xlabel("highway-mpg")
    plt.ylabel("Price of Cars")
    plt.show()
    ```

- For multi-variable, need to change variables first

  From
  $$
  y=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_1x_2+\theta_4x_1^2+\theta_5x_2^2
  $$
  to
  $$
  y=\theta_0w_0+\theta_1w_1+\theta_2w_2+\theta_3w_3+\theta_4w_4+\theta_5w_5
  $$

- Example

  ```python
  W = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
  
  from sklearn.preprocessing import PolynomialFeatures
  pr = PolynomialFeatures(degree=2)
  W_pr = pr.fit_transform(W)
  ```

- Pre-process -- Normalize each feature simultaneously

  ```python
  X = df[["horsepower", "highway-mpg"]]
  
  from sklearn.preprocessing import StandardScaler
  SCALE = StandardScaler()
  SCALE.fit(X)
  x_scale = SCALE.transform(X)
  ```

### Pipelines

- There are many steps to getting a prediction -- 

  X -> Normalizations -> Polynomial Transform -> Linear Regression -> y hat

- We can put these steps into pipeline

- ```python
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.linear_model import LinearRegression
  from sklearn.preprocessing import StandardScaler
  from sklearn.pipeline import Pipeline
  
  Input = [("scale", StandardScaler()), ("polynomial", PolynomialFeatures(include_bias=False)), ("model", LinearRegression())]
  pipe = Pipeline(Input)
  
  X = df[["horsepower", "curb-weight", "engine_size", "highway-mpg"]]
  pipe.fit(X, y)
  y_hat = pipe.predict(X)
  ```

### Measure for In-sample Evaluation

- A way to numerically determine how good the model fits on dataset

- Two important measures to determine the fit of a model

  - Mean Squared Error (MSE)
  - R-squared (R^2)

- $$
  MAE = \frac{1}{m}\sum^m_{i=1}|y_i-\hat{y}_i|
  $$

- $$
  MSE = \frac{1}{m}\sum^{m}_{i=1}(y_i-\hat{y_i})^2
  $$

- $$
  RMSE = \sqrt{MSE}
  $$

- $$
  RAE = \frac{\frac{1}{m}\sum^{m}_{i=1}|y_i-\hat{y_i}|}{\frac{1}{m}\sum^{m}_{i=1}|y_i-\bar{y}|}
  $$

- $$
  RSE = \frac{\frac{1}{m}\sum^{m}_{i=1}|y_i-\hat{y_i}|}{\frac{1}{m}\sum^{m}_{i=1}|y_i-\bar{y}|}
  $$

- $$
  R^2=1-RSE
  $$

### Mean Squared Error (MSE)

- ```python
  from sklearn.metrics import mean_squared_error
  mean_squared_error(df["price"], y_hat)
  ```

### R-squared (R^2)

- The Coefficient of Determination, or R-squared (R^2)

- Is a measure to determine how close the data is to the fitted regression line

- The percentage of variation of the target variable (y) that is explained by the linear model (y hat)

- Think about as comparing a regression model to a simple model (i.e. the mean of the data points)

- R^2 can only be used for linear regression and polynomial regression

- If R^2 -> 1, then prediction is accurate

- If R^2 -> 0, then prediction is not accurate

- Simple / Multiple Linear Regression

  ```python
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import mean_squared_error
  
  lm = LinearRegression()
  # SLR
  x = df[["highway-mpg"]]
  # MLR
  x = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
  y = df["price"]
  lm.fit(x, y)
  yhat = lm.predict(x)
  
  # R^2
  lm.score(x, y)
  # MSE
  mean_squared_error(y, yhat)
  ```

- Polynomial Regression

  ```python
  import numpy as np
  from sklearn.metrics import r2_score
  x = df["highway-mpg"]
  y = df["price"]
  f = np.polyfit(x, y, 3)
  p = np.polyld(f)
  yhat = p(x)
  
  # R^2
  r_squared = r2_score(y, yhat)
  # MSE
  mean_squared_error(y, yhat)
  ```

### Prediction and Decision making

- To determine final best fit, we look at a combination of
  - Do the predicted values make sense
  - Visualization
  - Numerical measures for evaluation
  - Comparing Models

### Comparing MLR and SLR

- Is a lower MSE always implying a better fit?
  - Not necessarily
- MSE for an MLR model will be smaller than the MSE for an SLR model, since the errors of the data will decrease when more variables are included in the model
- Polynomial regression will also have a smaller MSE then regular regression
- A similar inverse relationship holds for R^2



## Model Evaluation

- In-sample evaluation
  - Tell us how well our model will fit the data used to train it
- Model evaluation
  - Tell us how well the trained model can be used to predict new data
- Solution
  - In-sample data (or Training data)
  - Out-of-sample-data (or Test data)

### Training / Testing Sets

- Split dataset into

  - Training set (70%)
  - Testing set (30%)

- Build and train the model with a training set

- Use testing set to assess the performance of predictive model

- When we have completed testing our model we should use all the data to train the model to get the best performance

- ```python
  from sklearn.model_selecton import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
  
  x_data					features or independent variables
  y_data					dataset target
  x_train, y_train		 parts of available data as training set
  x_test, y_test			 parts of available data as testing set
  test_size                 percentage of testing set among all data
  random_state			 random seed
  ```

### Generalization Performance

- Generalization error is measure of how well our data does at predicting previously unseen data
- The error we obtain using our testing data is an approximation of this error

### Training Set /  Testing Set percentage

- In the same model
  - The more training set data
    - Model more accurate when training
    - Model less general when testing
  - The more testing set data
    - Model less accurate when training
    - Model more general when testing

### Cross Validation

- Most common out-of-sample evaluation metrics

- More effective use of data (each observation is used for both training and testing)

- Fold data into k fold, each of them being testing set for once

- Using cross validation to calculate R^2

  ```python
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.9, random_state=0)
  lre.fit(x_train[["horsepower"]], y_train)
  
  from sklearn.model_selection import cross_val_score
  Rcross = cross_val_score(lre, x_data[["horsepower"]], y_data, cv=4)
  
  print(Rcross)
  print("mean=", Rcross.mean(), ", standard deviation=", Rcross.std())
  --------------------------------------------------------------------------------------------------
  [0.7746232	0.51716687	0.74785353	0.04839605]
  mean=0.522009915042119, standard deviation=0.2911839444756029
  ```

- Using cross validation to get predict y hat

  ```python
  from sklearn.model_selection import cross_val_predict
  yhat = cross_val_predict(lre, x_data[["horsepower"]], y_data, cv=4)
  
  print(y_data.values[:5])
  print(yhat[:5])
  --------------------------------------------------------------------------------------------------
  [13495.	16500.	16500.	13950.	17450.]
  [14141.63807508	14141.63807508	20814.29423473	12745.03562306	14762.35027598]
  ```

### Under-fitting

- Consider data set
  $$
  y=f(x)+\text{noise}
  $$

- If we fit it using linear model, it will be under-fitting

  - R^2 -> 0
  - MSE too large

### Over-fitting

- If we fit data using polynomial with high-order, model will focus on the noise and being over-fitting

### Model Selection

- Bias-variance dilemma

  - When order is too low
    - Model is not accurate enough
    - Test Error lead by Bias
  - When order is too high
    - Model is not stable enough
    - Test Error lead by Variance

- Training Error

  - The more complex the model is, the less training error it has

- Test Error

  - Decrease as the model fits the trend in the data
  - Increase as the model fits the noise in the data

- Even if the model is just-fit, there is still error, mainly from both f(x) and noise

- We can get just-fit order from R^2

  ```python
  Rsqu_test = []
  order = [1, 2, 3, 4]
  for n in order:
  	pr = PolynomialFeatures(degree=n)
  	x_train_pr = pr.fit_transform(x_train[["horsepower"]])
  	x_test_pr = pr.fit_transform(x_test[["horsepower"]])
  	lr.fit(x_train_pr, y_train)
  	Rsqu_test.append(lr.score(x_test_pr, y_test))
  
  plt.plot(order, Rsqu_test)
  plt.xlabel("order")
  plt.ylabel("R^2")
  plt.title("R^2 Using Test Data")
  plt.text(3, 0.75, "Maximum R^2")
  plt.show()
  ```

### Ridge Regression - Alpha

- Using Alpha to control the coefficient of high-order items, in case over-fitting

- The more Alpha is, the less coefficient is

- If Alpha is too small, it will have no adjustment on coefficient, Model will still be over-fitting

- If Alpha is too large, Model will be not sensitive enough, and will be under-fitting

- For some specific Alpha,

  - Can fit model using RigeModel.fit()

  - Can get y hat using RigeModel.predict()

  - Can get R^2 using RigeModel.score()

    ```python
    from sklearn.linear_model import Ridge
    
    RigeModel = Ridge(alpha=0.1)
    RigeModel.fit(x_train_pr, y_train)
    yhat = RigeModel.predict(x_test_pr)
    
    print("predicted:", yhat[0:4])
    print("test set:", y_test[0:4].values)
    
    RigeModel.score(x_test_pr, y_test)
    RigeModel.score(x_train_pr, y_train)
    ----------------------------------------------------------------------------------------------
    predicted: [6567.83081933	9597.97151399	20836.22326843	19347.69543463]
    test set: [6295.	10698.	13860.	13499.]
    ```

### Ridge Regression - CV set

- Cross Validation Set

  - For picking Alpha, we partial the data into Train set, Cross Validation set, Test set
    - Train Set			For calculating theta
    - CV Set               For selecting alpha
    - Test Set             For test accuracy

- Example

  ```python
  # alpha -> R^2
  Rsqu_test = []
  Rsqu_train = []
  dummy1 = []
  ALFA = 5000 * np.array(range(0, 100))
  for alfa in ALFA:
  	RigeModel = Ridge(alpha=alfa)
  	RigeModel.fit(x_train_pr, y_train)
  	Rsqu_test.append(RigeModel.score(x_test_pr, y_test))
  	Rsqu_train.append(RigeModel.score(x_train_pr, y_train))
  	
  # Plot
  width = 12
  height = 10
  plt.figure(figsize=(width, height))
  plt.plot(ALFA, Rsqu_test, label="validation data")
  plt.plot(ALFA, Rsqu_train, "r", label="training Data")
  plt.xlabel("alpha")
  plt.ylabel("R^2")
  plt.legend()
  plt.show()
  ```

### Hyper-parameters

- Used for compensation, punishment, and conciliation
- Hyper-parameters do not attend the training for model, and is not necessary for function
- Optimize MSE, R^2 using Alpha
- If there are many hyper-parameters in an algorithm, we need to try for the best set of hyper-parameters

### Grid Search

- Produce many sets of hyper-parameters

- For every set, train Model using Train Set, and calculate effect using CV Set

- Choose the best set of hyper-parameters

- One hyper-parameter

  ```python
  # find best alpha
  from sklearn.model_selection import GridSearchCV
  
  parameters1 = [{"alpha": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}]
  RR = Ridge()
  Grid1 = GridSearchCV(RR, parameters1, cv=4)
  Grid1.fit(x_data[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_data)
  
  BestRR = Grid1.best_estimator_
  print(BestRR)
  --------------------------------------------------------------------------------------------------
  Ridge(alpha=10000, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
  
  # get R^2 of CV
  scores = Grid1.cv_results_
  for param, mean_test_score, mean_train_score in zip(scores["params"], scores["mean_test_score"], scores["mean_train_score"]):
      print(param, "R^2 on test set:", mean_test_score, "R^2 on trainset:", mean_train_score)
  --------------------------------------------------------------------------------------------------
  {'alpha': 0.001} 	R^2 on testset: 0.6654883665835498 R^2 on trainset: 0.8140026987974528
  {'alpha': 0.1} 		R^2 on testset: 0.665488937795743 R^2 on trainset: 0.8140026987941957
  {'alpha': 1} 		R^2 on testset: 0.6654941271777726 R^2 on trainset: 0.8140026984720654
  {'alpha': 10} 		R^2 on testset: 0.6655456808122747 R^2 on trainset: 0.8140026665999185
  {'alpha': 100} 		R^2 on testset: 0.6660293599956986 R^2 on trainset: 0.8139997918510513
  {'alpha': 1000} 	R^2 on testset: 0.6689682153688159 R^2 on trainset: 0.8138704882636019
  {'alpha': 10000} 	R^2 on testset: 0.673346359342164 R^2 on trainset: 0.8125837432264054
  {'alpha': 100000} 	R^2 on testset: 0.6578188384315317 R^2 on trainset: 0.7895414464863878
  {'alpha': 1000000} 	R^2 on testset: 0.6022997056641799 R^2 on trainset: 0.7213544117939192
  
  # calculate R^2 of test set with best alpha
  print(BestRR.score(x_test[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_test))
  --------------------------------------------------------------------------------------------------
  0.8411649831036149
  ```

- Multiple hyper-parameters

  ```python
  # find best alpha
  from sklearn.model_selection import GridSearchCV
  
  parameters2 = [{"alpha": [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000], "normalize":[True, False]}]
  Grid2 = GredSearchCV(Ridge(), parameters2, cv=4)
  Grid2.fit(x_data[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_data)
  BestRR = Grid2.best_estimator_
  print(BestRR)
  --------------------------------------------------------------------------------------------------
  Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None, normalize=True, andom_state=None, solver='auto', tol=0.001)
  
  # get R^2 of CV
  scores = Grid2.cv_results_
  for param, mean_test_score, mean_train_score in zip(scores["params"], scores["mean_test_score"], scores["mean_train_score"]):
      print(param, "R^2 on testset:", mean_test_score, " R^2 on trainset:", mean_train_score)
  --------------------------------------------------------------------------------------------------
  {'alpha': 0.001, 'normalize':True} R^2 on testset: 0.6660554729300521 R^2 on trainset: 0.8140019687091062
  {'alpha': 0.001, 'normalize':False} R^2 on testset: 0.6654883665835498 R^2 on trainset: 0.8140026987974528
  {'alpha': 0.1, 'normalize':True} R^2 on testset: 0.6941756253562563 R^2 on trainset: 0.8105467683113939
  {'alpha': 0.1, 'normalize':False} R^2 on testset: 0.665488937795743 R^2 on trainset: 0.8140026987941957
  {'alpha': 1, 'normalize':True} R^2 on testset: 0.6904869345835353 R^2 on trainset: 0.7491044403684128
  {'alpha': 1, 'normalize':False} R^2 on testset: 0.6654941271777726 R^2 on trainset: 0.8140026984720654
  {'alpha': 10, 'normalize':True} R^2 on testset: 0.3213768752320559 R^2 on trainset: 0.34185604290210514
  {'alpha': 10, 'normalize':False} R^2 on testset: 0.6655456808122747 R^2 on trainset: 0.8140026665999185
  {'alpha': 100, 'normalize':True} R^2 on testset: 0.017055171026273973 R^2 on trainset: 0.04960447968256004
  {'alpha': 100, 'normalize':False} R^2 on testset: 0.6660293599956986 R^2 on trainset: 0.8139997918510513
  {'alpha': 1000, 'normalize':True} R^2 on testset: -0.030196174506588125 R^2 on trainset: 0.005184451598996331
  {'alpha': 1000, 'normalize':False} R^2 on testset: 0.6689682153688159 R^2 on trainset: 0.8138704882636019
  {'alpha': 10000, 'normalize':True} R^2 on testset: -0.035168740046103375 R^2 on trainset: 0.0005207847579786762
  {'alpha': 10000, 'normalize':False} R^2 on testset: 0.673346359342164 R^2 on trainset: 0.8125837432264054
  {'alpha': 100000, 'normalize':True} R^2 on testset: -0.035668584455798936 R^2 on trainset: 5.2101975528046074e-05
  {'alpha': 100000, 'normalize':False} R^2 on testset: 0.6578188384315317 R^2 on trainset: 0.7895414464863878
  {'alpha': 100000, 'normalize':True} R^2 on testset: -0.035668584455798936 R^2 on trainset: 5.2101975528046074e-05
  {'alpha': 100000, 'normalize':False} R^2 on testset: 0.6578188384315317 R^2 on trainset: 0.7895414464863878
  
  # calculate R^2 of test set with best alpha
  print(BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test))
  --------------------------------------------------------------------------------------------------
  0.840859719294301
  ```

  