import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # Improved model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os

# --- Database and Data Management ---
DATABASE = "user_data.db"

def create_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sex INTEGER,
            weight REAL,
            height REAL,
            age INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def load_data():
    create_table()
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query("SELECT * FROM user_data", conn)
    conn.close()

    if df.empty:
        num_rows = 500
        df = pd.DataFrame({
            'sex': np.random.choice([0, 1], num_rows),
            'weight': np.random.randint(40, 120, num_rows),
            'height': np.random.randint(140, 200, num_rows),
            'age': np.random.randint(18, 65, num_rows)
        })
        save_data(df)
    return df

def save_data(df):
    conn = sqlite3.connect(DATABASE)
    df.to_sql('user_data', conn, if_exists='replace', index=False)
    conn.close()


# --- Model Training and Evaluation ---
def train_and_evaluate_model(df):
    X = df[['sex', 'weight', 'height']]
    y = df['age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Train/test split

    model = RandomForestRegressor(n_estimators=100, random_state=42) # More robust model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    accuracy_percentage = r2 * 100
    return model, accuracy_percentage



# --- Streamlit App ---
st.title("Age Prediction App")

# Load data and train the initial model
df = load_data()
model, accuracy = train_and_evaluate_model(df)


# User input
sex = st.selectbox("Select sex:", ["Male", "Female"])
weight = st.number_input("Enter weight (kg):", min_value=30)
height = st.number_input("Enter height (cm):", min_value=100)

if st.button("Predict Age"):
    sex_numeric = 1 if sex == "Female" else 0
    input_data = pd.DataFrame({'sex': [sex_numeric], 'weight': [weight], 'height': [height]})
    predicted_age = model.predict(input_data)[0]
    st.write(f"Predicted age: {int(round(predicted_age))}")

    actual_age = st.number_input("Enter actual age:", min_value=0, max_value=120)
    if actual_age:
        new_data = pd.DataFrame({'sex': [sex_numeric], 'weight': [weight], 'height': [height], 'age': [actual_age]})
        df = pd.concat([df, new_data], ignore_index=True)
        save_data(df)

        # Retrain and re-evaluate the model after adding new data
        df = load_data() # Reload to get updated data
        model, accuracy = train_and_evaluate_model(df)
        st.success("Data saved and model retrained!")

# Display data and metrics
if st.checkbox("Show Data and Metrics"):
    st.subheader("Training Data")
    st.write(df)


    st.subheader("Model Accuracy")
    st.write(f"R-squared (Accuracy): {accuracy:.2f}%")  # Display as percentage


    # Data Visualization
    st.subheader("Data Visualization")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df, x='height', y='weight', hue='age', ax=ax) # Improved visualization
    st.pyplot(fig)
