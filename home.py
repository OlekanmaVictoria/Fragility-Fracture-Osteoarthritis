import streamlit as st
from PIL import Image
import requests
from io import BytesIO

def show():
    # Title
    st.markdown(
        "<h1 style='text-align:center; color:#3366cc;'>🦿 Fragility Fracture & Osteoarthritis Prediction</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size:18px; color:#7f8c8d;'>"
        "Empowering bone health decisions for prosthetic users and the elderly"
        "</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # Load and Show Image
    try:
        url = "https://images.unsplash.com/photo-1611782373786-2c723b1b531b?auto=format&fit=crop&w=800&q=80"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    except:
        img = Image.open("assets/knee.webp")

    st.image(img, caption="🦴 Joint Health Matters", use_container_width=True)

    st.divider()

    # Copyright Notice
    with st.container():
        st.markdown("### 📜 Copyright Notice")
        st.markdown("""
**© 2025 Adiefe Mabel Judith & Jinadu Mahmud Babatunde — Academic research content**  
**© 2025 Olekanma Chinonso Victoria — Software, AI models, and application implementation**

This project was carried out as part of undergraduate research by **Adiefe Mabel Judith** and 
**Jinadu Mahmud Babatunde**, students of the Department of **Prosthetics & Orthotics** at the 
**Federal University of Technology Owerri (FUTO)**, under the academic supervision of 
**Prof. Ekezie Jervas**, in partial fulfilment of the requirements for the award of 
**Bachelor of Health Technology (B.Tech)**.
""")

    st.divider()

    # About Section
    with st.container():
        st.markdown("### 🧠 About This App")
        st.markdown("""
This application is a **proof-of-concept (POC)** predictive health tool designed to explore 
AI-assisted risk assessment in lower-limb prosthetic users and elderly populations.

It includes:
- **Fragility Fracture Risk Assessment / Prediction** in lower-limb prosthetic users — *Adiefe Mabel Judith*  
- **Osteoarthritis Risk Prediction** in lower-limb prosthetic users — *Jinadu Mahmud Babatunde*

**Technical Development & System Implementation:**  
The AI pipelines, data preprocessing workflows, and Streamlit application architecture were 
**designed and implemented by Olekanma Chinonso Victoria** in collaboration with the student researchers.

Click the button below to begin your personalized assessment.
""")

    st.write("")
    if st.button("🚀 Continue to Prediction", use_container_width=True):
        st.session_state.page = "predict"
        st.rerun()
