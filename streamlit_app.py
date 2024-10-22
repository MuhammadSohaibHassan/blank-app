import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # Simple model for demonstration
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Data Management ---
DATA_FILE = "user_data.csv"

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        # Placeholder data for initial training (replace with real data)
        num_rows = 500
        df = pd.DataFrame({
            'sex': np.random.choice(['Male', 'Female'], num_rows),
            'weight': np.random.randint(40, 120, num_rows),  # Example weight range
            'height': np.random.randint(140, 200, num_rows), # Example height range
            'age': np.random.randint(18, 65, num_rows)      # Example age range
        })
        df.to_csv(DATA_FILE, index=False) 
    return df

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def prepare_data(df):
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1}) # One-hot encode sex
    X = df[['sex', 'weight', 'height']]
    y = df['age']
    return X, y


# --- Model Training and Prediction ---
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_age(model, sex, weight, height):
    input_data = pd.DataFrame({'sex': [sex], 'weight': [weight], 'height': [height]})
    predicted_age = model.predict(input_data)[0]
    return predicted_age


# --- Streamlit App ---
st.title("Age Prediction App")

# Load and prepare data
df = load_data()
X, y = prepare_data(df)
model = train_model(X, y)


# User input
sex = st.selectbox("Select sex:", ["Male", "Female"])
weight = st.number_input("Enter weight (kg):", min_value=30)
height = st.number_input("Enter height (cm):", min_value=100)

if st.button("Predict Age"):
    sex_numeric = 1 if sex == "Female" else 0 # Convert sex to numeric for prediction
    predicted_age = predict_age(model, sex_numeric, weight, height)
    st.write(f"Predicted age: {int(round(predicted_age))}")

    # Get actual age from user
    actual_age = st.number_input("Enter actual age:", min_value=0, max_value=120) 
    if actual_age:
        new_data = pd.DataFrame({'sex': [sex], 'weight': [weight], 'height': [height], 'age': [actual_age]})
        new_data['sex'] = new_data['sex'].map({'Male': 0, 'Female': 1})
        df = pd.concat([df, new_data], ignore_index=True)  # Add new data to DataFrame
        save_data(df)  # Save the updated data

        # Retrain the model
        X, y = prepare_data(df)  
        model = train_model(X, y)
        st.success("Data saved and model retrained!")




# --- Data Visualization and Metrics ---
if st.checkbox("Show Data and Metrics"):
    st.subheader("Training Data")
    st.write(df)

    st.subheader("Model Accuracy")
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    st.write(f"Mean Absolute Error: {mae:.2f}")


    st.subheader("Data Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='height', y='weight', hue='age', data=df, ax=ax)
    st.pyplot(fig)


    # Example Accuracy over time (replace with your logic)
    # In a real app, you would track MAE over time and plot it.
    st.subheader("Accuracy Over Time (Example)")
    st.line_chart({"MAE": [2.5, 2.0, 1.8, 1.5]})
