# Walmart Sales Time Series Forecasting Using Machine and Deep Learning
## Datasets
**Walmart Recruiting - Store Sales Forecasting** downloaded from
https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
 - **train.csv** - CSV Data file containing following attributes
	- Store
	- Dept
	- Date
	- Weekly_Sales
	- IsHoliday

115064 Data rows
 - **stores.csv** - CSV Data File containing following attributes 
	 - Store
	 - Type
	 - Size
	 
45 Data rows
 - **features.csv** - CSV Data file containing following attributes
	- Store
	- Date
	- Temperature
	- Fuel_Price
	- MarkDown1
	- MarkDown2
	- MarkDown3
	- MarkDown4
	- MarkDown5
	- CPI
	- Unemployment
	- IsHoliday
	
8190 Data rows
## Machine Learning Models
- Linear Regression Model
- Random Forest Regression Model
- K Neighbors Regression Model
- XGBoost Regression Model
- Custom Deep Learning Neural Network
## Data Preprocessing
- ### **Handling Missing Values**
	- CPI, Unemployment of features dataset had 585 null values.
	- MarkDown1 had 4158 null values
	- MarkDown2 had 5269 null values
	- MarkDown3 had 4577 null values
	- MarkDown4 had 4726 null values
	- MarkDown5 had 4140 null values
	All missing values were filled using median of respective columns.
- ### **Merging Datasets**
	- Main Dataset merged with stores dataset
	- Resulting Dataset merged with features dataset
	- Total data rows 421570 and 15 attributes
	- Date column converted into datetime data type
	- Date attribute set as index of combined dataset
- ### **Splitting Date Column**
	- Split the Date column into Year, Month, Week
- ### **Aggregate Weekly Sales**
	- Create max, min, mean, median, std of weekly sales 
- ### **Outlier Detection and Other abnormalities**
	- Markdowns were summed into Total_MarkDown
	- Outliers were removed using z-score
	- Data rows 375438 and 20 columns
	- Negative weekly sales were removed
	- 374247 Data rows and 20 columns
- ### **One-hot-encoding**
	- Store, Dept, Type columns were one-hot-encoded using get_dummies method
	- After one-hot-encoding, no. of columns becomes 145
- ### **Data Normalization**
	- Numerical columns normalized using MinMaxScaler in the range 0 to 1 
- ### **Recursive Feature Elimination**
	- Random Forest Regressor used to calculate feature ranks and importance with 23 estimators
	-  Features selected to retain
		- mean, median, Week, Temperature, max, CPI, Fuel_Price, min, std, Unemployment, Month, Total_MarkDown, Dept_16, Dept_18, IsHoliday, Dept_3, Size, Dept_9, Year, Dept_11, Dept_1, Dept_5, Dept_56
	- No. of attributes after feature elimination - 24
## Splitting Dataset
- Dataset was splitted into 80% for training and 20% for testing
- Target feature - Weekly_Sales 
## Linear Regression Model
- Linear Regressor Accuracy - 92.28
- Mean Absolute Error - 0.030057
- Mean Squared Error - 0.0034851
- Root Mean Squared Error - 0.059
- R2 - 0.9228
- `LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)`
## Random Forest Regression Model
- Random Forest Regressor Accuracy - 97.889
- Mean Absolute Error - 0.015522
- Mean Squared Error - 0.000953 
- Root Mean Squared Error - 0.03087 
- R2 - 0.9788
- n_estimators - 100
- `RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None, oob_score=False, random_state=None, verbose=0, warm_start=False)`
## K Neighbors Regression Model
- KNeigbhbors Regressor Accuracy - 91.9726
- Mean Absolute Error - 0.0331221
- Mean Squared Error - 0.0036242
- Root Mean Squared Error - 0.060202
- R2 - 0.919921
- Neighbors - 1
- `KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=1, p=2, weights='uniform')`
## XGBoost Regression Model
- XGBoost Regressor Accuracy - 94.21152
- Mean Absolute Error - 0.0267718
- Mean Squared Error - 0.0026134
- Root Mean Squared Error - 0.051121
- R2 - 0.942115235
- Learning Rate - 0.1
- n_estimators - 100
- `XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0, importance_type='gain', learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=None, subsample=1, verbosity=1)`
## Custom Deep Learning Neural Network Model
- Deep Neural Network accuracy - 90.50328
- Mean Absolute Error - 0.033255
- Mean Squared Error - 0.003867
- Root Mean Squared Error - 0.06218 
- R2 - 0.9144106
- Kernel Initializer - normal
- Optimizer - adam
- Input layer with 23 dimensions and 64 output dimensions and activation function as relu
- 1 hidden layer with 32 nodes
- Output layer with 1 node 
- Batch Size - 5000
- Epochs -100
## Comparing Models
- Linear Regressor Accuracy - 92.280797
- Random Forest Regressor Accuracy - 97.889071
- K Neighbors Regressor Accuracy - 91.972603
- XGBoost Accuracy - 94.211523
- DNN Accuracy - 90.503287
## Citations
- **Walmart Recruiting - Store Sales Forecasting**
https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

> Written with [StackEdit](https://stackedit.io/).
