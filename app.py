import streamlit as st
from groq_client import generate_parallel_code

st.set_page_config(page_title="CAPC Python Parallelizer", layout="centered")
st.title("🚀 CAPC Python Parallelizer (Groq + LLaMA 3/4 Scout)")

st.markdown("Paste your serial Python function below. It will be optimized using Numba or NumPy.")

code_input = st.text_area("🔢 Serial Python Code", height=300)

if st.button("⚙️ Generate Optimized Code"):
    if code_input.strip():
        with st.spinner("Sending code to Groq..."):
            try:
                output = generate_parallel_code(code_input)
                st.subheader("✅ Optimized Code and Explanation")
                st.code(output, language='python')
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please paste some Python code first.")
