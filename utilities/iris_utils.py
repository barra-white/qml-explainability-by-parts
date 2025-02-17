# import dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(train_size=0.8, random_state=42, shuffle=True):
    """
    Loads the Iris dataset, performs an 80-20 train-test split while maintaining the class distribution,
    and scales the features using the MinMaxScaler (fitting only on the training data to avoid leakage).

    Parameters:
        train_size (float): Proportion of data to use for training (default: 0.8).
        random_state (int): Seed for reproducibility (default: 42).
        shuffle (bool): Whether to shuffle the dataset before splitting (default: True).
    """
    
    # Load the Iris dataset.
    iris = load_iris()
    X = iris.data
    y = iris.target

    # First split the data to avoid data leakage on scaling.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=y
    )

    # Initialize and fit the scaler on the training data.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    # Transform the test data using the same scaler.
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def get_num_features() -> int:
    """
    Returns the number of features in the Iris dataset.
    """
    return load_iris().data.shape[1]

def get_num_classes() -> int:
    """
    Returns the number of classes in the Iris dataset.
    """
    return len(load_iris().target_names)

def get_class_names() -> list[str]:
    """
    Returns the class names in the Iris dataset.
    """
    return load_iris().target_names

def get_feature_names() -> list[str]:
    """
    Returns the feature names in the Iris dataset.
    """
    return load_iris().feature_names