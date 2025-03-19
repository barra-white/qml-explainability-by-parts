# import dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class DiabetesData:
    """
    Utility class to load and preprocess the Pima Indians Diabetes dataset.
    Functions:
        preprocess_data: Loads the dataset, performs a 70-30 train-test split while maintaining the class distribution, and scaling the data using the MinMaxScaler.
        
        get_num_features: Returns the number of features in the dataset.
        
        get_num_classes: Returns the number of classes in the dataset.
        
        get_class_names: Returns the class names in the dataset.
        
        get_feature_names: Returns the feature names in the dataset.
    """
    def __init__(self, pathname: str):
        self.pathname = pathname
        self.data = pd.read_csv(self.pathname)
        
    def preprocess_data(self, train_size=0.7, random_state=42, shuffle=True):
        """
        Loads the Pima Indians Diabetes dataset, performs an 70-30 train-test split while maintaining the class distribution,
        and scales the features using the MinMaxScaler (fitting only on the training data to avoid leakage)

        Parameters:
            train_size (float): Proportion of data to use for training (default: 0.7)
            random_state (int): Seed for reproducibility (default: 42)
            shuffle (bool): Whether to shuffle the dataset before splitting (default: True)
        """
        # load dataset
        
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        
        #scaling
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=y
        )

        return X_train, X_test, y_train, y_test
    
    def preprocess_data_ranged(self, train_size=0.7, random_state=42, shuffle=True):
        """
        Loads the Pima Indians Diabetes dataset, performs an 70-30 train-test split while maintaining the class distribution,
        and scales the features using the MinMaxScaler (fitting only on the training data to avoid leakage)

        Parameters:
            train_size (float): Proportion of data to use for training (default: 0.7)
            random_state (int): Seed for reproducibility (default: 42)
            shuffle (bool): Whether to shuffle the dataset before splitting (default: True)
        """
        # load dataset
        
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        
        #scaling
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)
        
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=y
        )

        return X_train, X_test, y_train, y_test
        
    def get_num_features(self) -> int:
        """
        Returns the number of features in the Pima Indians Diabetes dataset.
        """
        return self.data.shape[1] - 1
    
    def get_num_classes(self) -> int:
        """
        Returns the number of classes in the Pima Indians Diabetes dataset.
        """
        return len(self.data.iloc[:, -1].unique())
    
    def get_class_names(self) -> list[str]:
        """
        Returns the class names in the Pima Indians Diabetes dataset.
        """
        return self.data.iloc[:, -1].unique()
    
    def get_feature_names(self) -> list[str]:
        """
        Returns the feature names in the Pima Indians Diabetes dataset.
        """
        return self.data.columns[:-1]
    
    





# def preprocess_data(train_size=0.7, random_state=42, shuffle=True):
#     """
#     Loads the breast cancer dataset, performs an 70-30 train-test split while maintaining the class distribution,
#     and scales the features using the MinMaxScaler (fitting only on the training data to avoid leakage).

#     Parameters:
#         train_size (float): Proportion of data to use for training (default: 0.7).
#         random_state (int): Seed for reproducibility (default: 42).
#         shuffle (bool): Whether to shuffle the dataset before splitting (default: True).
#     """
    
#     # Load the Iris dataset.
#     wine = load_wine()
#     X = wine.data
#     y = wine.target
    
#     # encode labels
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
    
#     # split the data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=y
#     )

#     return X_train, X_test, y_train, y_test

# def get_num_features() -> int:
#     """
#     Returns the number of features in the wine dataset.
#     """
#     return load_wine().data.shape[1]

# def get_num_classes() -> int:
#     """
#     Returns the number of classes in the wine dataset.
#     """
#     return len(load_wine().target_names)

# def get_class_names() -> list[str]:
#     """
#     Returns the class names in the wine dataset.
#     """
#     return load_wine().target_names

# def get_feature_names() -> list[str]:
#     """
#     Returns the feature names in the wine dataset.
#     """
#     return load_wine().feature_names