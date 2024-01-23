import pandas as pd
import numpy as np
import random
import string
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm
import matplotlib.pyplot as plt

def run_complex_analysis(rows, columns):
    np.random.seed(42)
    random.seed(42)

    def generate_random_string(length):
        return ''.join(random.choices(string.ascii_lowercase, k=length))

    def generate_random_data(rows, columns):
        data = {'Feature_' + str(i): np.random.rand(rows) for i in range(columns)}
        data['Target'] = np.random.choice([0, 1], rows)
        return pd.DataFrame(data)

    def generate_classification_data(rows, columns):
        X, y = make_classification(n_samples=rows, n_features=columns, random_state=42)
        feature_columns = [f"Feature_{i}" for i in range(columns)]
        return pd.DataFrame(X, columns=feature_columns), pd.Series(y, name='Target')

    def train_random_forest(X_train, y_train):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        return clf

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy:.2f}')

    def plot_distribution():
        data = np.random.normal(size=1000)
        plt.hist(data, bins=30, density=True, alpha=0.7, color='blue')
        plt.title('Normal Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    def complex_function_1():
        result = []
        for x in np.linspace(0, 2*np.pi, 100):
            result.append(np.cos(x))
        return result

    def complex_function_2():
        result = []
        for x in np.arange(1, 101):
            result.append(np.log1p(x))
        return result

    def complex_function_3():
        result = []
        for x in np.arange(1, 101):
            result.append(np.sqrt(x))
        return result

    def complex_function_4():
        result = []
        for x in np.arange(1, 101):
            result.append(np.exp(x))
        return result

    def complex_function_5():
        result = []
        for x in np.linspace(0, 2*np.pi, 100):
            result.append(np.sin(x))
        return result

    def generate_complex_data(rows, columns):
        data = {'Feature_' + str(i): np.random.normal(loc=i, scale=i/10, size=rows) for i in range(columns)}
        data['Target'] = np.random.choice([0, 1], rows)
        return pd.DataFrame(data)

    # Generate random data
    random_data = generate_random_data(rows, columns)

    # Display summary statistics
    print("Summary Statistics:")
    print(random_data.describe())

    # Generate classification data
    classification_data, target = generate_classification_data(rows, columns)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(classification_data, target, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    model = train_random_forest(X_train, y_train)

    # Evaluate the model
    print("\nModel Evaluation:")
    evaluate_model(model, X_test, y_test)

    # Call complex functions
    result_1 = complex_function_1()
    result_2 = complex_function_2()
    result_3 = complex_function_3()
    result_4 = complex_function_4()
    result_5 = complex_function_5()

    # Plot distribution
    #plot_distribution()

    # Generate more complex data
    complex_data = generate_complex_data(rows, columns)

    # Display summary statistics for complex data
    print("\nSummary Statistics for Complex Data:")
    print(complex_data.describe())
    
    
    
    
    
    
    
    
