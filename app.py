import streamlit as st
import home
import prediction

# Page configuration
st.set_page_config(
    page_title="Fragility Fracture & Osteoarthritis Prediction – Olekanma Chinonso Victoria",
    page_icon="🦴",
    layout="centered"
)

# App header
st.title("🦴 Fragility Fracture & Osteoarthritis Prediction")
st.caption("Developed by Olekanma Chinonso Victoria")

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
    "© 2025 Adiefe Mabel Judith & Jinadu Mahmud Babatunde — Academic research content<br>"
    "© 2026 Olekanma Chinonso Victoria — Software, AI models, and application implementation"
    "</p>",
    unsafe_allow_html=True
)
