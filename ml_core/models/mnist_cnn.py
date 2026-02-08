import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
class MNISTClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64),
        learning_rate_init: float = 0.001,
        max_iter: int = 10,
        batch_size: int = 64,
        alpha: float = 0.0001,
        random_state: int = 42,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.alpha = alpha
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self._pipeline = None
        self._classes = None
    def _build_pipeline(self):
        classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            batch_size=self.batch_size if self.batch_size != 'auto' else 'auto',
            alpha=self.alpha,
            random_state=self.random_state,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            solver='adam',
            activation='relu',
            verbose=False,
        )
        self._pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        return self._pipeline
    def fit(self, X, y):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        self._classes = np.unique(y)
        if self._pipeline is None:
            self._build_pipeline()
        self._pipeline.fit(X_flat, y)
        return self
    def predict(self, X):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        return self._pipeline.predict(X_flat)
    def predict_proba(self, X):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        return self._pipeline.predict_proba(X_flat)
    def score(self, X, y):
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
        return self._pipeline.score(X_flat, y)
    @property
    def classes_(self):
        return self._classes
    def get_params(self, deep=True):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'random_state': self.random_state,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
        }
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._pipeline = None
        return self
