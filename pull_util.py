import pandas as pd

def analyze_given_data(data):
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Display summary statistics for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    print("Summary Statistics:")
    print(numeric_df.describe())

    # Analyze correlation between numeric features
    correlation_matrix = numeric_df.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
