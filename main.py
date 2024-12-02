# Importing necessary libraries
import streamlit as st  # For creating the web app
import pandas as pd  # For working with data (tables)
import numpy as np  # For generating random numbers
from sklearn.preprocessing import LabelEncoder  # For converting grades to numeric values


# Step 1: Generate dummy data with better distribution
def generate_dummy_data():
    """Generate dummy data for marks and grades with a better distribution."""
    # Setting a random seed for reproducibility (so you always get the same data)
    np.random.seed(42)

    # Creating the data
    data = {
        # Marks data: Generates 5 values for each grade (A, B, C, D)
        "marks": np.concatenate([
            np.random.randint(90, 101, 5),  # 5 data points for 'A' (marks between 90 and 100)
            np.random.randint(70, 90, 5),  # 5 data points for 'B' (marks between 70 and 90)
            np.random.randint(50, 70, 5),  # 5 data points for 'C' (marks between 50 and 70)
            np.random.randint(30, 50, 5)  # 5 data points for 'D' (marks between 30 and 50)
        ]),

        # Grades data: Assigning grades for each range of marks
        "grades": np.concatenate([
            ['A'] * 5,  # 'A' for marks between 90 and 100
            ['B'] * 5,  # 'B' for marks between 70 and 90
            ['C'] * 5,  # 'C' for marks between 50 and 70
            ['D'] * 5  # 'D' for marks between 30 and 50
        ])
    }

    # Convert the dictionary to a pandas DataFrame (a table)
    df = pd.DataFrame(data)
    return df  # Return the DataFrame with the data


# Step 2: Convert grades to numeric values for easier handling by machine learning models
def preprocess_data(df):
    """Convert grades to numeric values."""
    label_encoder = LabelEncoder()  # Create an encoder to convert 'A', 'B', 'C', 'D' to numbers
    df['grades_numeric'] = label_encoder.fit_transform(df['grades'])  # Transform grades to numbers
    return df, label_encoder  # Return the modified DataFrame and the encoder


# Step 3: Rule-based grade prediction function
def predict_grade(marks):
    """Predict grade based on marks using a thresholding rule."""
    # This function uses a simple threshold to determine the grade based on marks.
    if marks >= 90:
        return 'A'  # If marks are 90 or higher, return 'A'
    elif marks >= 70:
        return 'B'  # If marks are 70 or higher (but less than 90), return 'B'
    elif marks >= 50:
        return 'C'  # If marks are 50 or higher (but less than 70), return 'C'
    elif marks >= 30:
        return 'D'  # If marks are 30 or higher (but less than 50), return 'D'
    else:
        return 'E'  # If marks are less than 30, return 'E'


# Step 4: Build Streamlit web app
def main():
    """Create the Streamlit app interface."""

    # App title
    st.title("Student Grade Prediction App")
    st.write("This app predicts the grade of a student based on their marks.")  # Description of the app

    # Input field for marks: User can enter a value between 0 and 100
    marks = st.number_input("Enter the marks (0-100):", min_value=0, max_value=100, step=1)

    # Button to trigger the prediction when clicked
    if st.button("Predict"):
        # When the button is clicked, show a loading spinner and generate dummy data
        with st.spinner("Generating data..."):
            df = generate_dummy_data()  # Generate the dummy data
            df, label_encoder = preprocess_data(df)  # Preprocess the data (convert grades to numeric)

        # Use the rule-based prediction function to predict the grade based on the input marks
        predicted_grade = predict_grade(marks)

        # Display the predicted grade in the app
        st.success(f"The predicted grade is: {predicted_grade}")

    # Optionally, show the generated dummy data for debugging or learning purposes
    if st.checkbox("Show dummy data"):
        st.write(generate_dummy_data())  # Display the dummy data as a table


# Run the app
if __name__ == "__main__":
    main()  # Call the main function to run the app
