import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    'Attendance': [90, 75, 60, 85, 50, 95, 40, 70, 80, 55, 65, 92, 30, 88, 77],
    'Previous_Score': [85, 70, 55, 80, 45, 90, 35, 65, 75, 50, 60, 88, 25, 82, 72],
    'Pass': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Attendance', 'Previous_Score']]
y = df['Pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title("🎓 Student Pass Prediction Prototype")
st.write("This tool predicts whether a student will Pass or Fail based on Attendance and Previous Exam Scores.")


attendance_input = st.slider("Attendance (%)", 0, 100, 75)
previous_score_input = st.slider("Previous Exam Score (%)", 0, 100, 70)

if st.button("Predict"):
    new_student = np.array([[attendance_input, previous_score_input]])
    prediction = model.predict(new_student)[0]
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.subheader(f"Prediction: {result}")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Model Accuracy on Test Data:** {accuracy:.2f}")

