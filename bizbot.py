import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load and preprocess the dataset
def load_data():
    st.title('BizBot - Business Data Analysis & Prediction')
    uploaded_file = st.file_uploader("Upload your CSV file for analysis", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Overview:")
        st.write(data.head())
        return data
    else:
        st.warning("Please upload a CSV file.")
        return None

# Correlation Matrix
def plot_correlation_matrix(data):
    st.subheader('📊 Correlation Matrix')
    numerical_data = data.select_dtypes(include=[np.number])
    if numerical_data.empty:
        st.write("No numerical columns found for correlation analysis.")
        return
    
    corr = numerical_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# Data Distribution
def plot_data_distribution(data):
    st.subheader('📈 Data Distribution')
    numerical_data = data.select_dtypes(include=[np.number])
    if numerical_data.empty:
        st.write("No numerical columns found for distribution analysis.")
        return
    
    for column in numerical_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        st.pyplot(fig)

# Categorical Data Distribution
def plot_categorical_distribution(data):
    st.subheader('📊 Categorical Data Distribution')
    categorical_data = data.select_dtypes(include=[object, 'category'])
    if categorical_data.empty:
        st.write("No categorical columns found for distribution analysis.")
        return
    
    for column in categorical_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data[column], ax=ax)
        ax.set_title(f'Count Plot of {column}')
        st.pyplot(fig)

# Missing Values
def plot_missing_values(data):
    st.subheader('🧹 Missing Values')
    missing_values = data.isnull().sum()
    missing_values_percentage = (missing_values / len(data)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_values_percentage})
    st.write("### Missing Values Report")
    st.write(missing_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title('Missing Values Heatmap')
    st.pyplot(fig)

# Outliers Detection
def plot_outliers(data):
    st.subheader('🚨 Outliers Detection')
    numerical_data = data.select_dtypes(include=[np.number])
    if numerical_data.empty:
        st.write("No numerical columns found for outlier analysis.")
        return
    
    for column in numerical_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=data[column], ax=ax)
        ax.set_title(f'Boxplot of {column}')
        st.pyplot(fig)

# Skewness Analysis
def plot_skewness(data):
    st.subheader('📊 Skewness')
    numerical_data = data.select_dtypes(include=[np.number])
    if numerical_data.empty:
        st.write("No numerical columns found for skewness analysis.")
        return
    
    skewness = numerical_data.skew()
    st.write("### Skewness of Numerical Columns")
    st.write(skewness)

    for column in numerical_data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f'{column} Skewness: {skewness[column]:.2f}')
        st.pyplot(fig)

# Handle missing data
def handle_missing_data(data):
    imputer = SimpleImputer(strategy='mean')
    data_imputed = data.copy()
    data_imputed[data_imputed.columns] = imputer.fit_transform(data_imputed)
    return data_imputed

# Train model
def train_model(X_train, y_train, model_type='LinearRegression'):
    if model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'DecisionTree':
        model = DecisionTreeRegressor()
    elif model_type == 'RandomForest':
        model = RandomForestRegressor()
    elif model_type == 'SVR':
        model = SVR()
    
    model.fit(X_train, y_train)
    return model

# Prediction
def prediction(data):
    st.subheader('💡 Future Predictions for Business')
    data_imputed = handle_missing_data(data)
    target_column = st.selectbox("Select target column for prediction:", data.columns)
    X = data_imputed.drop(columns=[target_column])
    y = data_imputed[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_option = st.selectbox("Choose a model for prediction:", ['LinearRegression', 'DecisionTree', 'RandomForest', 'SVR'])
    model = train_model(X_train, y_train, model_type=model_option)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Actual vs Predicted ({target_column})')
    st.pyplot(fig)

# Statistical Analysis
def display_statistics(data):
    st.subheader('📊 Statistical Insights')
    st.write("### General Statistical Overview")
    st.write(data.describe().T)

# Main function
def main():
    data = load_data()
    if data is not None:
        st.sidebar.title("Navigation")
        options = st.sidebar.radio("Choose an Analysis:", ["Overview", "Correlation Matrix", "Data Distribution", 
                                                           "Categorical Distribution", "Missing Values", 
                                                           "Outliers Detection", "Skewness", "Predictions", "Statistics"])
        if options == "Overview":
            st.write("### Data Overview")
            st.write(data)
        elif options == "Correlation Matrix":
            plot_correlation_matrix(data)
        elif options == "Data Distribution":
            plot_data_distribution(data)
        elif options == "Categorical Distribution":
            plot_categorical_distribution(data)
        elif options == "Missing Values":
            plot_missing_values(data)
        elif options == "Outliers Detection":
            plot_outliers(data)
        elif options == "Skewness":
            plot_skewness(data)
        elif options == "Predictions":
            prediction(data)
        elif options == "Statistics":
            display_statistics(data)

# Run the app
if __name__ == '__main__':
    main()
