import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score

# Load the pre-trained XGBoost model
best_xgb_model = joblib.load('best_xgb_model.pkl')

def predict(input_data):
    predictions = best_xgb_model.predict(input_data)
    prediction_probabilities = best_xgb_model.predict_proba(input_data)
    return predictions, prediction_probabilities

def main():
    st.title("Loan Default Prediction")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Choose input method", ["Manual Input", "Upload CSV"])

    if options == "Manual Input":
        st.header("Enter Data Manually")

        totaldue = st.number_input("Total Due", value=0.0)
        education_flag = st.number_input("Education Flag", value=0.0)
        employment_flag = st.number_input("Employment flag", value=0.0)
        bnk_acc_typ_flag = st.number_input("Bank Account Type Flag", value=0.0)
        loan_freq = st.number_input("Loan Frequency", value=0.0)
        avg_paybacktime = st.number_input("Average Playback Time", value=0.0)
        clients_avg_loanAMT = st.number_input("Average Loan", value=0.0)
        clients_max_loanAMT = st.number_input("Max Loan", value=0.0)
        total_loans = st.number_input("Total Loans", value=0.0)
        ontime_loans = st.number_input("Loans Paid Ontime", value=0.0)
        ontime_percentage = st.number_input("On Time Percentage", value=0.0)
        pays_late_risk = st.number_input("Pays Late Risk", value=0.0)
        referred_client = st.number_input("Referred Client", value=0.0)
        loan_interest2 = st.number_input("Loan Interest", value=0.0)
        percentage_loan_interest = st.number_input("Feature 15", value=0.0)
        age_categ_youth = st.number_input("Youth", value=0.0)
        age_categ_young_adult = st.number_input("Young Adult", value=0.0)
        age_categ_adult = st.number_input("Adult", value=0.0)
        age_categ_retired = st.number_input("Retired", value=0.0)
        borrower_reliability_score = st.number_input("Borrower Reliability Score", value=0.0)
                
        input_data = np.array([[totaldue, education_flag, employment_flag, bnk_acc_typ_flag, loan_freq, avg_paybacktime, clients_avg_loanAMT, clients_max_loanAMT, total_loans, ontime_loans, ontime_percentage, pays_late_risk, referred_client, loan_interest2, percentage_loan_interest, age_categ_youth, age_categ_young_adult, age_categ_adult, age_categ_retired, borrower_reliability_score]])
        if st.button("Predict"):
            predictions, prediction_probabilities = predict(input_data)
            st.write(f"Prediction: {predictions[0]}")
            st.write("Prediction Probabilities:")
            st.write(prediction_probabilities)

    elif options == "Upload CSV":
        st.header("Upload CSV File")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write("Input Data:")
            st.write(input_df)

            if st.button("Predict"):
                predictions, prediction_probabilities = predict(input_df)
                input_df["Predictions"] = predictions
                st.write("Predictions:")
                st.write(input_df)

                # Prediction Distribution Plot
                st.subheader("Prediction Distribution")
                fig = px.histogram(input_df, x="Predictions")
                st.plotly_chart(fig)

                # Feature Importance Plot
                st.subheader("Feature Importance")
                importance = best_xgb_model.feature_importances_
                feature_names = input_df.columns[:20]  # Adjust based on the number of features
                fig, ax = plt.subplots()
                sns.barplot(x=importance, y=feature_names, ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)

                # Display Prediction Probabilities
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame(prediction_probabilities, columns=[f'Class {i} Probability' for i in range(prediction_probabilities.shape[1])])
                st.write(prob_df)

if __name__ == "__main__":
    main()
