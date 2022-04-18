import pandas as pd
from sklearn.linear_model import LinearRegression

class ParModel:
	"""This class implements the Periodic Autoregressive Model (PAR)
	"""

	def __init__(self, data, n_predictors=12, n_forecasts=1):
		"""Initialize the ParModel

		Args:
			data (pandas.core.frame.DataFrame, optional): The data that will be used to 
															train and test the model
		"""
		self.data = data
		self.n_predictors = n_predictors
		self.n_forecasts = n_forecasts


	def resample_data(self):
		"""This method resamples the dataframe in a way where each representes a
		date of forecast and each column is a predictor that will be used to 
		predict the last columns of the dataframe.

		Args:
			n_predictors (int): Number of predictors that will be used to forecast
								the target column.
		""" 

		pivoted_data = [] # This list will storage the data resampled

		n_columns = self.n_predictors + self.n_forecasts

		for start_date in self.data.index[:-n_columns]:								# First date of the columns it's nth months before the prediction
			end_date = start_date + pd.DateOffset(months=n_columns-1)				# Last date of the columns it's the month to predict
			pivoted_data.append(self.data.loc[start_date:end_date].values.T[0])
		
		# Renaming the columns. Columns with p are the predictors and target is month to predict
		self.x_columns = [f'p{i}' for i in range(self.n_predictors)]
		self.y_columns = [f'target{i}' for i in range(1, self.n_forecasts+1)]
		self.columns_names = [*self.x_columns, *self.y_columns]
		self.par_data = pd.DataFrame(pivoted_data, columns=self.columns_names) 		# Dataframe that model will use

		# Setting the index of the pivoted table.
		self.par_data.index = self.data.index[self.n_predictors-1:-(self.n_forecasts+1)]	# The index of the columns refers to the month when the predictons are made		
		self.par_data.dropna(inplace=True) 											# Removing null values from the DataFrame
		self.par_data.index.names = ['forecast date']

	def train_test_split_data(self, data=None, threshold=.7):
		"""Split the dataset in train an test sets.

		Args:
			threshold (float, optional): Threshold to divide the dataset. Defaults to .7.
		"""
		if data is None:
			data = self.par_data
		
		split_threshold = int(len(data)*threshold)		
		self.train = data.iloc[:split_threshold]		
		self.test = data.iloc[split_threshold:]		


	def fit(self, train=None, x_columns=None, y_columns=None):
		"""Fit the ParModel

		Args:
			train (pandas.core.frame.DataFrame, optional): Dataset to train the model. Defaults to None.
			x_columns (list, optional): List of string with the predictor columns label. Defaults to None.
			y_column (string, optional): Target column label. Defaults to None.
		"""

		# If the optional arg aren't provide the model will use the default values
		if train is None:
			train = self.train
		if x_columns is None:
			x_columns = self.x_columns
		if y_columns is None:
			y_columns = self.y_columns
		
		# Fitting the Linear Regression model with the train data
		self.reg = LinearRegression().fit(train[x_columns], train[y_columns])
		self.coef_ = self.reg.coef_
		self.intercept_ = self.reg.intercept_
	

	def predict(self, test=None, x_columns=None):
		"""Predict the y after the model has fitted.

		Args:
			test (pandas.core.frame.DataFrame, optional): Dataset to test the model. Defaults to None.
			x_columns (list, optional): List with the predictor columns label. Defaults to None.

		Returns:
			pandas.core.frame.DataFrame: The predicted data
		"""

		# If the optional args aren't provided it will used the default values
		if test is None:
			test = self.test
		if x_columns is None:
			x_columns = self.x_columns
		
		# Predicting the test target with the model fitted
		return self.reg.predict(test[x_columns])

