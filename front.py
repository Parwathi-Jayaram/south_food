import streamlit as st
from back import predict_dish

st.set_page_config(page_title="Recipe Detector", page_icon="🍳")

st.title("🍳 Recipe Detector")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg","png"])
camera_file = st.camera_input("Or take a photo")

input_file = uploaded_file if uploaded_file else camera_file

st.title("Recipes")

if input_file:
    st.image(input_file, caption="Uploaded food image")  # Show preview
    with st.spinner("Analyzing your dish..."):
        try:
            results = predict_dish(input_file)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    if not results:
        st.warning("No recipes found for this dish.")
    else:
        for idx, result in enumerate(results, start=1):
            st.subheader(f"Recipe {idx}: {result['dish']}")

            st.markdown("### 🥕 Ingredients")
            for ing in result["ingredients"].split(","):
                if ing.strip():
                    st.write(f"- {ing.strip()}")

            st.markdown("### 📖 Steps")
            for i, step in enumerate(result["instructions"].split("."), start=1):
                if step.strip():
                    st.write(f"**Step {i}:** {step.strip()}")