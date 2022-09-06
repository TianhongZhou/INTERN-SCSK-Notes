# Data Pre-processing

## 特征缩放 

### 数据标准化

#### 归一化

- 将数据各维特征映射到指定的范围之内：[0, 1]或者[-1, 1]

- 压缩量纲

- 归一化类型

  - 极大极小归一化
    $$
    \begin{cases}
    	\mathrm{X} = \frac{\mathrm{X}_{old}-\min(\mathrm{X}_{old})}{\max(\mathrm{X_{old}})-\min(\mathrm{X}_{old})}, \qquad [0,1] \\
    	\mathrm{X} = \frac{2(\mathrm{X}_{old}-\min(\mathrm{X}_{old}))}{\max(\mathrm{X_{old}})-\min(\mathrm{X}_{old})}-1, \ [-1,1] \\
    \end{cases}
    $$
    
  - 均值归一化
    $$
    \mathrm{X} = \frac{\mathrm{X}_{old}-mean(\mathrm{X}_{old})}{\max(\mathrm{X_old})-\min(\mathrm{X_{old}})}
    $$
    
  - 非线性归一化

    - 缩小各维特征的量纲
    - 例如取对数

- 特点

  - 改变原始数据的数据分布，不能保留原始信息

- 作用

  - 加快训练速度
  - 平衡各维特征权重，避免数值尺度过大、过小的特征对模型的干扰

- 缺陷

  - 改变了原始数据的数据分布，破坏了数据结构

- 归一化 0 到 1 范围

  ```python
  def normalization(data):
      _range = np.max(data) - np.min(data)
      return (data - np.min(data)) / _range
  ```

- 归一化 -1 到 1 范围

  ```python
  def normalization(data):
      _range = np.max(abs(data))
      return data / _range
  ```

- ```python
  X_train = np.array([[ 1., -1., 2.], [ 2., 0., 0.], [ 0., 1., -1.]])
  min_max_scaler = preprocessing.MinMaxScaler()
  X_train_minmax = min_max_scaler.fit_transform(X_train)
  ```


#### z-score

- 将数据缩放到以0为中心，标准差为1的某种数据分布

- 保留原始数据信息，不会改变数据分布类型

- $$
  \mathrm{X}=\frac{\mathrm{X}_{old}-\mu}{\sigma}
  $$

- ```python
  from sklearn import preprocessing
  data = preprocessing.scale(values)
  ```


#### 归一化和z-score的区别与联系

- 联系
  - 都是对原始数据做线性变化
  - 将样本点平移然后缩短距离
  - 使原始数据的不同特征具有可比性
- 区别
  - 归一化对目标函数的影响体现在数值上，而z-score对目标函数的影响体现在数据几何分布上
  - 归一化改变了数据的量级并同时也改变了数据的分布，z-score只改变了数据的量级但未改变数据的分布类型
  - 标准化处理数据，不会改变目标函数的等高线投影，并且会继续保持原有目标函数扁平性，而归一化处理数据会使目标函数的等高线投影呈现圆形
  - 在梯度下降算法中，归一化处理数据有助于加快算法收敛速度
- 何时使用
  - 数据中各维特征尺度差异大(量纲)，目标函数易受尺度大的特征的干扰。比如涉及距离计算的模型：knn、kmeans、dbscan、svm等需要将数据量纲统一标准
  - 使用梯度下降的参数估计模型：使用归一化处理后的数据可以提高算法收敛速度
  - 涉及皮尔逊相关系数的模型：使用标准化处理的数据可以及方便计算相似度
  - PCA降维算法需要去中心化，可以使用z-score处理
  - 对数值范围有具体要求的，需使用归一化处理数据，比如图像处理，其中像素强度必须归一化以适应一定范围(RGB颜色范围为0到255)
  - 概率模型对特征量纲差异性不敏感，可以不做。如决策树
  - 一般：不确定使用哪种数据处理方式时，就用z-score处理，至少z-score处理不会改变数据分布类型，即不会破坏数据结构



### 中心化/零均值化

- 中心化处理后的数据，数据均值为0向量，即将原始数据平移到原点附近

- 中心化处理数据是一个一个平移的过程，不会改变数据分布类型

- $$
  \mathrm{X}=\mathrm{X}_{old}-mean(\mathrm{X}_{old})
  $$

- 作用

  - 方便计算协方差矩阵
  - 去除截距项（偏置项）的影响

- ```python
  X -= numpy.mean(X, axis=0)
  ```



### 正则化

- 正则化处理数据将每个样本的某个范数（L1范数、L2范数）缩放为1，即对每个样本计算其p-范数，然后对该样本中的每个元素除以该范数，使得处理后的数据的每个样本的p-范数等于1

- $$
  \mathrm{X}=\frac{\mathrm{X}_{old}}{||\mathrm{X}_{old}||_p}\implies ||\mathrm{X}||_p=1
  $$

- 主要应用于文本分类和聚类中，对需要计算样本间相似度有很大作用

- ```python
  # Use sklearn.preprocessing.normalizer class to normalize data.
  import numpy as np
  from sklearn.preprocessing import Normalizer
  
  x = np.array([1, 2, 3, 4], dtype='float32').reshape(1,-1) 
  options = ['l1', 'l2', 'max']
  for opt in options:
      norm_x = Normalizer(norm=opt).fit_transform(x)
  ```
  
  

### 离散化

- 数据离散化是指将连续的数据进行分段，使其变为一段段离散化的区间

- 分段的原则有基于等距离、等频率或优化的方法

- 数据离散化的原因
  - 模型需要
    - 比如决策树、朴素贝叶斯等算法，都是基于离散型的数据展开的
    - 如果要使用该类算法，必须将离散型的数据进行
    - 有效的离散化能减小算法的时间和空间开销，提高系统对样本的分类聚类能力和抗噪声能力
  - 离散化的特征相对于连续型特征更易理解
  - 可以有效的克服数据中隐藏的缺陷，使模型结果更加稳定

- 等频法
  - 使得每个箱中的样本数量相等，例如总样本n=100，分成k=5个箱，则分箱原则是保证落入每个箱的样本量=20

- 等宽法
  - 使得属性的箱宽度相等，例如年龄变量（0-100之间），可分成 [0,20]，[20,40]，[40,60]，[60,80]，[80,100]五个等宽的箱

- 聚类法
  - 根据聚类出来的簇，每个簇中的数据为一个箱，簇的数量模型给定

- ```python
  # 等距离散
  df['Age_discretized'] = pd.cut(df.Age, 4, labels=range(4))
  df.groupby('Age_discretized').count()
  ```

- ```python
  # 使用聚类实现离散化
  # 数据准备
  data = df['Income']
  # 改变数据形状
  data_re = data.reshape((data.index.size, 1))
  # 创建k-means模型并指定聚类数量
  km_model = KMeans(n_clusters=4, random_state=2018)
  # 模型导入数据
  result = km_model.fit_predict(data_re)
  # 离散数据并入原数据
  df['Income_discretized'] = result
  df.groupby('Income_discretized').count()
  ```

- ```python
  # 使用4分位离散数据
  df['Spend_discretized'] = pd.qcut(df.Spend, 4, labels=['C', 'B', 'A', 'S'])
  df.groupby('Spend_discretized').count()
  ```

- ```python
  # 等频率离散
  # 设置离散区间数
  k = 4
  # 获取数据
  data = df.Age
  # 设置频率范围
  w = [1.0*i/k for i in range(k+1)]
  # 使用describe获取频率区域的分界点
  w = data.describe(percentiles = w)[4:4+k+1]
  w[0] = w[0]*(1-1e-10)
  # 根据分界点进行数据离散处理
  df['Age2'] = pd.cut(data, w, labels = range(k))
  df.groupby('Age2').count()
  ```

- ```python
  # 数据二值化
  # 建立模型 根据平均值作为阈值
  data = df['Income']
  binarizer_scaler = preprocessing.Binarizer(threshold=data.mean())
  # 二值化处理
  result = binarizer_scaler.fit_transform(data.reshape(-1, 1))
  # 数据合并
  df['Income2'] = result
  df.groupby('Income2').count()
  ```



### 稀疏化

- 针对离散型且标称变量，无法进行有序的LabelEncoder时，通常考虑将变量做0，1哑变量的稀疏化处理，例如动物类型变量中含有猫，狗，猪，羊四个不同值，将该变量转换成is猪，is猫，is狗，is羊四个哑变量
- 若是变量的不同值较多，则根据频数，将出现次数较少的值统一归为一类'rare'
- 稀疏化处理既有利于模型快速收敛，又能提升模型的抗噪能力



## 缺失值处理

### 删除缺失值

- 最简单最暴力
- 一般不推荐

### 填充缺失值

- 缺失值替换为：平均值、中位数、众数

- 使用KNN算法填充：缺失样本点周围最近的k个样本的均值或者最大值填充

- 加权平均值/期望替换缺失值：比较含缺失值的样本与其他样本之间的相似度，计算其加权平均值
  $$
  \Sigma(特征值\cdot相似度)/\text{sum}(相似度)
  $$
  
- 

### 代码案例

- 忽略并删除元组数据（可能造成丢失信息的问题）

  ```python
  import pandas as pd
  import numpy as np
  temp = {'col1':[1,np.nan,2,np.nan,4,5],'col2':[6,5,4,3,2,1]}
  df = pd.DataFrame(temp)
  df1 = df.dropna(axis=1) # delete columns have n/a
  df2 = df.dropna(axis=0) # delete rows have n/a
  ```

- 自动填充空值（极少数的缺失，可以考虑手动补齐。其他情况手动补齐不现实）

  - 根据近邻值补齐，向后填充

    ```python
    import pandas as pd
    import numpy as np
    temp = {'col1':[1,np.nan,2,np.nan,4,5],'col2':[6,5,4,3,2,1]}
    df = pd.DataFrame(temp)
    df1 = df.fillna(method='ffill')
    ```

  - 根据近邻值补齐，向前填充

    ```python
    import pandas as pd
    import numpy as np
    temp = {'col1':[1,np.nan,2,np.nan,4,5],'col2':[6,5,4,3,2,1]}
    df = pd.DataFrame(temp)
    df1 = df.fillna(method='bfill')
    ```

  - 通过属性的均值来补齐

    ```python
    import pandas as pd
    import numpy as np
    temp = {'col1':[1,np.nan,2,np.nan,4,5],'col2':[6,5,4,3,2,1]}
    df = pd.DataFrame(temp)
    
    for column in list(df.columns[df.isnull().sum()>0]):
        mean_value = df[column].mean()
        df.fillna(mean_value,inplace=True)
    ```
    
  - 基于回归
  
    ```python
    import pandas as pd 
    import numpy as np 
    from scipy.stats import linregress
    import matplotlib.pyplot as plt 
    
    df = pd.DataFrame({'col1':[12,16,24,30,np.nan,58,np.nan,87],
    'col2':[14,16,26,36,47,58,69,79]})
    df1 = df.copy()
    df1.dropna(axis=0,inplace=True)
    lr = linregress(df1['col2'],df1['col1'])
    k = lr[0]
    d = lr[1]
    df1['lr'] = df1['col2']*k+d
    
    dfnew = df.copy()
    for i in range(len(dfnew)):
        if np.isnan(df['col1'][i]) == True:
            dfnew['col1'][i] = (dfnew['col2'][i])*k+d
    
    fig, ax = plt.subplots()
    plt.grid(alpha=0.5)
    ax.scatter(x=dfnew['col2'],y=dfnew['col1'],facecolor='red')
    ax.scatter(x=df['col2'],y=df['col1'],facecolor='blue')
    plt.plot(df1['col2'],df1['lr'])
    ```



## 噪声处理

- 数据分箱

  - 减少次要观察误差的影响

  - 重要性与优势

    - 分类模型的建立需要对连续变量离散化，特征离散化后，模型会更稳定，降低模型过拟合的风险
    - 离散特征的增加和减少都很容易，易于模型的快速迭代
    - 稀疏矩阵向量内积乘法运算速度快，计算结果方便存储，容易扩展

    - 离散化后的特征对异常数据有很强的鲁棒性，避免异常数据对模型的干扰
    - 离散化后，相当于为模型引入了非线性，能够提升模型的表达能力，加大拟合

  - 步骤

    - 数据排序（sort），并分到不同的等量箱子（equi-size bin）内
    - 接下来对每个箱子内的数据，考虑用箱子均值（bin means）做平滑，用箱子中位数（bin medians）做平滑，用箱子边界（bin boundaries）来做平滑等等

  - 示例

    - 按数据范围来分箱

      ```python
      import pandas as pd
      import numpy as np
      df = pd.DataFrame({'score':[14,15,16,76,45,67,34,87,98,67,45,32]})
      dfcut = df.copy()
      dfcut['cut_group'] = pd.cut(dfcut['score'],4)
      dfcut['cut_group'].value_counts()
      ```
      
    - 按数据频率，控制每个箱子的数据个数
    
      ```python
      import pandas as pd
      import numpy as np
      df = pd.DataFrame({'score':[14,15,16,76,44,67,34,87,98,67,45,32]})
      dfcut = df.copy()
      dfcut['qcut_group'] = pd.qcut(dfcut['score'],4)
      dfcut['qcut_group'].value_counts()
      ```
      
    - 边界设置

      ```python
      import pandas as pd
      import numpy as np
      df = pd.DataFrame({'score':[14,15,16,76,44,67,34,87,98,67,45,32]})
      dfcut = df.copy()
      dfcut['cut_group'] = pd.cut(dfcut['score'],bins=[0,60,70,85,100])
      dfcut['cut_group'].value_counts()
      ```
      
    - 改变分箱的标签
    
      ```python
      import pandas as pd
      import numpy as np
      df = pd.DataFrame({'score':[14,15,16,76,44,67,34,87,98,67,45,32]})
      dfcut = df.copy()
      dfcut['cut_group'] = pd.cut(dfcut['score'],bins=[0,60,70,85,100],labels=['not-pass','good','better','excellent'])
      dfcut['cut_group'].value_counts()
      ```



## 离群点处理

- 检测离群点的方法

  - 简单统计分析

    - 根据箱线图、各分位点判断是否存在异常

    - 如pandas的describe函数

    - ```python
      import matplotlib.pyplot as plt
      
      plt.figure(figsize = (2,4))
      sns.boxplot(y = df['value'])
      ```

  - 3 sigma原则

    - 若数据存在正态分布，偏离均值的3 sigma之外的点为离群点

    - $$
      P(|x-\mu|>3\sigma)<=0.003
      $$

    - ```python
      import pandas as pd
      import random
      import seaborn as sns
      import numpy as np
      
      # 生成1个正态分布，均值为0，标准差为1，1000个点
      data = np.random.normal(0, 1, size=1000) 
      df = pd.DataFrame(data,columns=["value"])
      sns.distplot(df['value'])
      # 计算均值
      mean = df['value'].mean()
      # 计算标准差
      sigma = df['value'].std()
      # 异常值范围范围 (均值-3*标准差, 均值+3*标准差)
      [mean-3*sigma,mean+3*sigma]
      >>> [-3.082683606205224, 3.052458853421742]
      # 筛选异常值
      [ i for i in data if not mean-3*sigma<i<mean+3*sigma]
      >>> [-3.3411613233772415,
            3.1618429735810696,
            3.8839317879436583,
            3.5785429625413543,
            3.0799355798019774]
      ```
    
  - 基于绝对离差中位数
  
    - 对抗离群数据的距离值方法
    - 计算各观测值与平均值的距离总和的方法
    - 放大了离群值的影响
  
  - 基于距离
  
    - 通过定义对象之间的临近性度量，根据距离判断异常对象是否远离其他对象
    - 缺点是计算复杂度较高，不适用于大数据集和存在不同密度区域的数据集

  - 基于密度
  
    - 离群点的局部密度显著低于大部分近邻点，适用于非均匀的数据集

  - 基于聚类

    - 利用聚类算法，丢弃远离其他簇的小簇
  
  - 其他方法
  
    - 利用df
  
      ```python
      plt.figure(figsize = (8,6))
      outlier = df.boxplot(return_type = 'dict')
      y = outlier['fliers'][0].get_ydata()
      ```
  
- 离群值处理方法

  - 一般保留，异常检测模型情况下，异常数据本身就是目标数据

  - 彻底删除

    - 如果想找出一般的规律，而且异常值也不太多，可以考虑删除
    - 异常值可能会影响结论

  - 对数转换

    - 对数据进行对数转换可以让数据的分布变的更加集中并且不会变化原有数据的相对关系和分布状态

    - 这样对存在离群值的数据集合影响变化不是很大

      ```python
      df["value"] = np.log(df["value"])
      sns.distplot(df['value'])
      ```

  - 保留异常

    - 因为异常值代表的也是真实发生的事件，背后是具体的行为
    - 有些值即使异常，也不会影响模型
    - 在进行拟合的时候并没有使模型造成太大偏差

  - 截尾

    - 只取数据集合中2.5% ~ 97.5%的范围内数据

      ```python
      def cut_data(data,cut1,cut2):
          len_data = len(data)
          p1 = int(len_data * cut1)
          p2 = int(len_data * cut2)
          return data[p1:p2]
      
      cut_data(data,0.25,0.975)
      ```

      

## 多项式特征

- 数据升维
- 例如把x1,x2变为x1^2,x2^2



## 数据转换

### 数据转换方法

- 数据换转为高斯分布方法

  - 对数变换
    $$
    \mathrm{X}=\log(\mathrm{X}_{old})
    $$
    
  - 倒数变换
    $$
    \mathrm{X}=\frac{1}{\mathrm{X}_{old}}
    $$
    
  - 平方根变换
    $$
    \mathrm{X}=\sqrt{\mathrm{X}_{old}}
    $$
    
  - 指数变换
    $$
    \mathrm{X}=\text{asin}(\sqrt{\mathrm{X}_{old}})
    $$



## 数据缩减

### 存在的问题

- 太多的预测特征会不必要地使分析的解释复杂化
- 保留过多的特征可能会导致过度拟合

### 解决方法

- 降维
  - 减少预测特征的数量
  
  - 确保预测特征是独立的
  
  - 主成分分析（PCA）和因子分析（FA）
    - PCA通过空间映射的方式，将当前维度映射到更低的维度，使得每个变量在新空间的方差最大
    
      ```python
      from sklearn.decomposition import PCA
      
      X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
      pca = PCA(n_components=2)
      pca.fit(X)
      PCA(copy=True, n_components=2, whiten=False)
      print(pca.explained_variance_ratio_) 
      [ 0.99244...  0.00755...]
      ```
    
    - FA则是找到当前特征向量的公因子（维度更小），用公因子的线性组合来描述当前的特征向量
    
      - 在进行因子分析之前，需要先进行充分性检测，主要是检验相关特征阵中各个变量间的相关性，是否为单位矩阵，也就是检验各个变量是否各自独立
    
        - Bartlett's球状检验
    
          -  检验总体变量的相关矩阵是否是单位阵（相关系数矩阵对角线的所有元素均为1,所有非对角线上的元素均为零）
    
          - 即检验各个变量是否各自独立
    
          - 如果不是单位矩阵，说明原变量之间存在相关性，可以进行因子分子；反之，原变量之间不存在相关性，数据不适合进行主成分分析
    
          - ```python
            from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
             
            chi_square_value, p_value = calculate_bartlett_sphericity(df)
            ```
    
        - KMO检验
    
          - 检查变量间的相关性和偏相关性，取值在0-1之间
    
          - KOM统计量越接近1，变量间的相关性越强，偏相关性越弱，因子分析的效果越好
    
          - 通常取值从0.6开始进行因子分析
    
          - ```python
            from factor_analyzer.factor_analyzer import calculate_kmo
            
            kmo_all,kmo_model=calculate_kmo(df)
            ```
    
      - 选择因子个数
    
        - 计算相关矩阵的特征值，进行降序排列
    
        - ```python
          faa = FactorAnalyzer(25,rotation=None)
          faa.fit(df)
           
          # 得到特征值ev、特征向量v
          ev,v=faa.get_eigenvalues()
          ```
    
      - 因子旋转
    
        - 建立因子分析模型
    
          ```python
          # 选择方式： varimax 方差最大化
          # 选择固定因子为 2 个
          faa_two = FactorAnalyzer(2,rotation='varimax')
          faa_two.fit(df)
          ```
    
        - 查看因子方差
    
          ```python
          # 公因子方差
          faa_two.get_communalities()
          # 查看每个变量的公因子方差
          pd.DataFrame(faa_two.get_communalities(),index=df.columns)
          ```
    
        - 查看旋转后的特征值
    
          ```python
          faa_two.get_eigenvalues()
          ```
    
        - 查看成分矩阵
    
          ```python
          # 变量个数*因子个数
          faa_two.loadings_
          ```
    
        - 查看因子贡献率
    
          - 总方差贡献：variance (numpy array)
    
            ```python
            faa_two.get_factor_variance()
            ```
    
          - 方差贡献率：proportional_variance (numpy array) 
    
          - 累积方差贡献率：cumulative_variances (numpy array)
    
  - 奇异值分解（SVD）
    - SVD的降维可解释性较低，且计算量比PCA大，一般用在稀疏矩阵上降维，例如图片压缩，推荐系统
    
    - ```python
      import numpy as np
       
      # Creating a matrix A
      A = np.array([[3,4,3],[1,2,3],[4,2,1]])
      # Performing SVD
      U, D, VT = np.linalg.svd(A)
      # Checking if we can remake the original matrix using U,D,VT
      A_remake = (U @ np.diag(D) @ VT)
      ```
    
    - ```python
      import numpy as np
      from sklearn.decomposition import TruncatedSVD
       
      # Creating array 
      A = np.array([[3,4,3],[1,2,3],[4,2,1]])
      # Fitting the SVD class
      trun_svd =  TruncatedSVD(n_components = 2)
      A_transformed = svd.fit_transform(A)
      ```
    
  - 聚类
    - 将某一类具有相似性的特征聚到单个变量，从而大大降低维度
    
  - 线性组合
    - 将多个变量做线性回归，根据每个变量的表决系数，赋予变量权重，可将该类变量根据权重组合成一个变量
    
  - 流行学习
    - 流行学习中一些复杂的非线性方法



## 数据集成

- 合并不同来源的数据

  ```python
  import pandas as pd
  df1=pd.DataFrame({'Name':['Tom','Jerry','Mark'],'A_score':[90,88,90]})
  df2=pd.DataFrame({'Name':['Tom1','Jerry1','Mark1'],'A_score':[90,88,90]})
  frames = [df1,df2]
  df=pd.concat(frames)
  ```

- 合并不同来源的主数据

  - 合并由于实体名字定义的相同数据（比如：学号，有的专业叫id，优的专业叫student_id。但对应的内容是一样的，集成的时候需要合并这两列）

- 识别和解决不同数据值的冲突

  - 比如：两份温度数据，一份按摄氏温度记录（℃），另一份按华氏温度记录（℉）；两份速度数据，一份按（m/s）记录，另一份按（km/h）来记录
  - 类似这类数据，合并的时候需要统计计量方式

- 移除重复数据（duplicates）和冗余数据（redundant）

  - 对于重复数据，最简单的办法是判断数据内容完全相同的行

    如果完全相同，则直接删除

    ```python
    import pandas as pd
    df1=pd.DataFrame({'Name':['Tom','Tom','Tom','Jerry','Mark'],'A_score':[90,90,90,88,90]})
    df2=df1.drop_duplicates()
    ```



## 除python外的数据预处理

主流即R语言和Python