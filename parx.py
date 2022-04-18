from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from src.PAR.par import ParModel
import pandas as pd

class ParxModel(ParModel):

	def __init__(self, data, climate_indexes, n_predictors=12, n_forecasts=1):
		super().__init__(data, n_predictors, n_forecasts)
		self.climate_indexes = climate_indexes
	

	def resample_data(self):
		super().resample_data()
		self.parx_data = self.climate_indexes.join(self.par_data)
		self.parx_data_old = self.parx_data.copy()
		self.parx_data = self.parx_data.dropna()
		self.x_columns = self.parx_data.columns[:-self.n_forecasts]
		self.y_column = self.parx_data.columns[-self.n_forecasts]
	

	def train_test_split_data(self, threshold=0.7):
		super().train_test_split_data(data=self.parx_data, threshold=threshold)


	def fit(self, train=None, x_columns=None, y_columns=None):

		if train is None:
			train = self.train
		if x_columns is None:
			x_columns = self.x_columns
		if y_columns is None:
			y_columns = self.y_columns
		
		self.regx = make_pipeline(StandardScaler(), LinearRegression())
		self.regx.fit(train[x_columns], train[y_columns])
		self.reg = self.regx.named_steps['linearregression']
		self.reg_coef_ = self.reg.coef_
		self.intercept_ = self.reg.intercept_
	

	def predict(self, test=None, x_columns=None):
		if test is None:
			test = self.test
		if x_columns is None:
			x_columns = self.x_columns
		
		return self.regx.predict(test[x_columns])
