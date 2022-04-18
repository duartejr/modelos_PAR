from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.PAR.par import ParModel


class ParPcrModel(ParModel):

    def __init__(self, data, n_predictors=12, n_forecasts=1):
        super().__init__(data, n_predictors, n_forecasts)

    
    def fit(self, train=None, x_columns=None, y_columns=None, n_components=1):

        if train is None:
            train = self.train
        if x_columns is None:
            x_columns = self.x_columns
        if y_columns is None:
            y_columns = self.y_columns
                
        self.pcr = make_pipeline(StandardScaler(), PCA(n_components=n_components),
                                 LinearRegression())
        self.pcr.fit(train[x_columns], train[y_columns])
        self.pca = self.pcr.named_steps['pca']
        self.reg = self.pcr.named_steps['linearregression']
        self.reg_coef_ = self.reg.coef_
        self.intercepct = self.reg.intercept_
    

    def predict(self, test=None, x_columns=None):
        if test is None:
            test = self.test
        if x_columns is None:
            x_columns = self.x_columns
        
        return self.pcr.predict(test[x_columns])
