import streamlit as st
import home
import prediction

# Page configuration
st.set_page_config(
    page_title="Fragility Fracture & Osteoarthritis Prediction – Olekamna Chinonso Victoria",
    page_icon="🦴",
    layout="centered"
)

# App header
st.title("🦴 Fragility Fracture & Osteoarthritis Prediction")
st.caption("Developed by Olekamna Chinonso Victoria")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# Handle navigation from Home buttons
if st.session_state.page == "predict":
    prediction.show()
else:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🧠 Predict"]
    )

    if page == "🏠 Home":
        home.show()
    elif page == "🧠 Predict":
        prediction.show()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>"
    "© 2026 Olekamna Chinonso Victoria</p>",
    unsafe_allow_html=True
)
