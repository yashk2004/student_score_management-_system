import streamlit as st

# Grade prediction function
def predict_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 70:
        return "B"
    elif marks >= 50:
        return "C"
    elif marks >= 30:
        return "D"
    else:
        return "E"

# Main App
def main():
    st.title("Student Grade Prediction App")
    st.write("This app predicts the grade of a student based on their marks.")

    marks = st.number_input(
        "Enter the marks (0-100):",
        min_value=0,
        max_value=100,
        value=0,
        step=1
    )

    if st.button("Predict"):
        grade = predict_grade(marks)
        st.success(f"The predicted grade is: {grade}")

if __name__ == "__main__":
    main()
