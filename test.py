import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Streamlit app
st.title("💳 Credit Card Fraud Detection")
st.write("Enter all the features (comma-separated) to predict whether the transaction is **legitimate** or **fraudulent**.")
st.write(f"🔍 Model Accuracy: **{test_acc:.2f}** on test data")

# Input field
input_df = st.text_input('Enter input features (comma-separated):')

# Submit button
submit = st.button("Submit")

if submit:
    try:
        # Clean and convert input to float list
        input_df_lst = input_df.split(',')
        cleaned_input = [float(x.strip().strip('"').strip("'")) for x in input_df_lst]

        # Check if feature count is correct
        if len(cleaned_input) != X.shape[1]:
            st.error(f"Expected {X.shape[1]} features, but got {len(cleaned_input)}.")
        else:
            # Convert to array and reshape
            features = np.array(cleaned_input, dtype=np.float64).reshape(1, -1)
            
            # Prediction
            prediction = model.predict(features)

            # Result
            if prediction[0] == 0:
                st.success("✅ Legitimate transaction")
            else:
                st.error("⚠️ Fraudulent transaction")
    except ValueError:
        st.error("Please enter valid numerical values separated by commas.")
