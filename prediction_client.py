import streamlit as st
import pandas as pd
import numpy as np
import joblib

def show():  # 👈 added function definition so prediction.py can call it
    # Load the trained model
    model = joblib.load("models/osteoporosis_risk_model.pkl")

    st.title("🦴 Fragility Fracture Prediction in Lower-Limb Prosthetic Users")
    st.write(
        "This tool assists with evaluating the risk of fragility fractures in lower-limb prosthetic users."
    )

    # ---- User Inputs ----
    st.subheader("Patient Information")

    age = st.number_input("Age", min_value=10, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.subheader("Prosthetic Details")

    prosthetic_difficulty = st.radio("Any prosthetic difficulties?", ["Yes", "No"])

    # 👇 New dynamic section for prosthetic challenges
    if prosthetic_difficulty == "Yes":
        issues = st.multiselect(
            "Select the challenges you experience:",
            [
                "Pain",
                "Discomfort",
                "Loose socket",
                "Tight socket",
                "Alignment issues",
                "Balance issues",
                "Others",
            ],
        )
    else:
        issues = []

    # 👇 Updated amputation levels
    amputation_level = st.selectbox(
        "Level/Type of amputation",
        [
            "Hemipelvectomy",
            "Hip Disarticulation",
            "Transfemoral Amputation",
            "Knee Disarticulation",
            "Transtibial Amputation",
            "Ankle Disarticulation",
            "Syme's Amputation",
            "Boyd's Amputation",
            "Chopart's Amputation",
            "Lis Franc's Amputation",
            "Transmetatarsal Amputation",
            "Transphalangeal Amputation",
        ],
    )

    prosthesis_use_duration = st.number_input(
        "How long have you been using the prosthesis? (in years)", min_value=0, max_value=50
    )

    previous_fracture = st.radio("Any previous fractures?", ["Yes", "No"])
    bone_density = st.selectbox("Bone Density (T-score)", ["Normal", "Osteopenia", "Osteoporosis"])

    # ---- Prepare Input Data ----
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Prosthetic_Difficulty": [prosthetic_difficulty],
            "Issues": [", ".join(issues) if issues else "None"],
            "Amputation_Level": [amputation_level],
            "Prosthesis_Use_Years": [prosthesis_use_duration],
            "Previous_Fracture": [previous_fracture],
            "Bone_Density": [bone_density],
        }
    )

    st.write("### Preview of Input Data")
    st.dataframe(input_data)

    # ---- Prediction Section ----
    if st.button("🔍 Predict Risk"):
        try:
            prediction = model.predict(input_data)
            risk = prediction[0]

            st.subheader("Prediction Result")

            if risk == 1:
                st.warning(
                    "⚠️ High risk of fragility fracture detected. "
                    "Please consult your clinician or prosthetist for further biomechanical assessment and preventive strategies."
                )
            else:
                st.success(
                    "✅ Low risk of fragility fracture detected. "
                    "Maintain a healthy lifestyle and ensure regular prosthetic check-ups."
                )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # ---- Footer ----
    st.caption(
        "©️ 2025 Adiefe Mabel Judith, Jinadu Mahmud Babatunde, Prof. Ekezie Jervas, & Olekanma Chinonso Victoria — All rights reserved."
    )


# 👇 This ensures it still runs directly if you open it alone
if __name__ == "__main__":
    show()
