from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import openai
import cv2
import os
import requests
from streamlit_lottie import st_lottie

st.title("FireWatch AI")

#openai api
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPEN_AI_KEY')
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

#Predict Function
def predict_with_yolov8(img_bytes):
    # Load the YOLOv8 model
    model = YOLO('best.pt')

    # Convert the image bytes to PIL image
    pil_image = Image.open(img_bytes)

    # Run YOLOv8 segmentation on the image
    results = model.predict(pil_image, imgsz=600,conf=0.3, iou=0.5)
    # Get the path of the new image saved by YOLOv8
    # Assuming inference[0] is the Results object
    res_plotted = results[0].plot()[:, :, ::-1]
    pred= results[0].boxes.cls
    
    return res_plotted,pred

#Loading GIF files
def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()
lotties1= load_lottieurl("https://lottie.host/3d230f30-56d2-4eb1-9796-5e364e310fe1/iaVkIihz06.json")
lotties2= load_lottieurl("https://lottie.host/7c19b488-2ff6-4692-8e45-d5c35d2247df/3wl3vilrwD.json")
lotties3= load_lottieurl("https://lottie.host/c5cc8f5f-ccea-44aa-9062-6390537109b1/j3QsRUcyYJ.json")

# File uploader to get the image from the user
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Predict with YOLOv8 and Openai API
if uploaded_file is not None:
        tab1, tab2,tab3 = st.tabs(["Smoke Detection", "News Article","Awarness"])
        out_img, out_name = predict_with_yolov8(uploaded_file)

        with tab1:
            st.image(out_img, use_column_width=True,caption="Image")
            
            if out_name.numel()==0:
                st.markdown("**No Smoke Detected.**")
                st_lottie(lotties3,height=200, key="coding3")
                
            else:
                st.markdown("**Smoke Detected!**")
                st_lottie(lotties1,height=200, key="coding4")
                
        
        with tab2:
            if out_name.numel() == 0:
                st.markdown("**Empty**")
                st_lottie(lotties2,height=300, key="coding")  
            else:
             with st.container():
                col1, col2 = st.columns([2, 1])
                with col1:
                        prompt=f"""Write me a News article for the wildfire smoke in Canada.
                        Make the News article short."""

                        response = get_completion(prompt)
                        st.markdown(response) #Get News Article from Openai API
                with col2:
                    st.image(out_img, use_column_width=True,caption="Wildfire Smoke in Canada")
        
        with tab3:
            if out_name.numel() == 0:
                st.write("**Empty**")
                st_lottie(lotties2,height=300, key="coding2")
            else:
             prompt=f"""Give me following awarness information for wildfire smoke:
             -educational content
             -tips
             -best practices for wildfire prevention
             -safety measures
             
             Make sure to give only few keywords as possible for it """
             response = get_completion(prompt)
             st.markdown(response) #Get Awarness from Openai API

             