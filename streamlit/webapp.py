import streamlit as st
from io import BytesIO
import requests



FASTAPI_URL = "http://localhost:8000/predict"

st.title("Facial Expression Recognition")

def predict_expression(file):
    files = {"file": file}
    response = requests.post(FASTAPI_URL, files=files)
    return response



def main():
    st.info(__doc__)
    image = st.file_uploader("Upload Image",type = ["jpg","png"])
    show_image = st.empty()

    if not image:
        formats = " ".join(["jpg", "png", "jpeg"])
        st.info(f"Please Upload Image: {formats}")
        return
    content = image.read()
    image.seek(0)

    if isinstance(image,BytesIO):
        show_image.image(image)

    response = predict_expression(content)

    if response.status_code == 200:
        predicted_class = response.headers["predicted_class"]
        st.success(f"Predicted Expression: {predicted_class}")
    else:
        st.error("Error occurred during prediction.")
    image.close()

main()

