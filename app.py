import streamlit as st
import tensorflow as tf
import requests
import PIL.Image
import os
from keras.utils.image_utils import load_img, img_to_array
import numpy as np


######################  Streamlit Settings  ######################
st.set_option('deprecation.showfileUploaderEncoding', False)
# Set the layout of the page
st.set_page_config(page_title='Covid Chest X-ray Detection', page_icon=':microbe:', layout='wide')



######################  Models and Classes  ######################
# Load the selected model
model_paths = {
    "CNN Model": "Model/CNN_Model.h5",
}
# Creating a Sidebar for the Model selection
model_name = st.sidebar.selectbox("Select the model to use", list(model_paths.keys()), index=0)
model = tf.keras.models.load_model(model_paths[model_name])

# Define the class labels
class_labels = ['covid', 'virus', 'normal']

# Test images path
image_filenames = os.listdir('Images/Test_imgs/')




# Define the function to make predictions
def predict(image_data):
    # Load the image and preprocess it
    img = load_img(image_data, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make the prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]

    return class_name



# Create the streamlit app
def app():

    ######################  User Interface  ######################
    # Set up the sidebar
    st.sidebar.title("COVID-19 X-ray Classification")
    st.sidebar.write('Please select an option:')
    app_mode = st.sidebar.selectbox("",
        ["Show instructions", "Use test X-ray image", "Upload an X-ray image"])
    # Define the layout of the Streamlit app
    logo = PIL.Image.open('Images/UI_Imgs/Reverb_logo.png')
    st.image(logo, width=100)
    # Set the title of the page
    st.title("COVID-19 Chest X-ray Classification")
    # Description
    st.write('This app can predict whether an X-ray image shows evidence of COVID-19 or not. Please select an option from the sidebar.')
    st.write('Upload an X-ray image to detect if it is Covid, Virus, or Normal')
    # ShAI Picture
    shai_img = PIL.Image.open("Images/UI_Imgs/SHAI-P3.png")
    # display image using streamlit
    st.image(shai_img, width=800, caption="Project Image")


    ######################  Show instructions  ######################
    if app_mode == "Show instructions":
        st.write("""
        **Instructions**
        - To use a test X-ray image, select an option from the dropdown menu.
        - To upload your own X-ray image, select "Upload an X-ray image" and upload an image in PNG, JPEG, or JPG format.
        - Adjust the confidence threshold as needed. If the model predicts that an X-ray image shows evidence of COVID-19 with a confidence level below the threshold, the result will be displayed as "Virus".
        - If the model predicts that an X-ray image shows evidence of COVID-19 with a confidence level above the threshold, the result will be displayed as "COVID-19" and a warning message will be displayed.
        """)


    ######################  Use test X-ray image  ######################
    if app_mode == "Use test X-ray image":
        st.sidebar.title('Select a test image')
        selected_image_name = st.sidebar.selectbox('Please select a test image:', image_filenames)
        st.write('Selected test image:', selected_image_name)
        selected_image_path = 'Images/Test_imgs/' + selected_image_name
        selected_image = PIL.Image.open(selected_image_path)
        selected_image = selected_image.resize((500, 500))
        # Display the image
        st.image(selected_image,width=300, caption='Selected test image', use_column_width=True)
        # Make a prediction 
        prediction = predict(selected_image_path)
        # Display the results
        if prediction == "covid":
           st.warning("Warning: This X-ray image shows evidence of COVID-19.")
        elif prediction == "virus":
            st.warning("Warning: This X-ray image shows evidence of a Virus.")
        elif prediction == "normal":
            st.success("Success: This X-ray image shows no evidence of any virus.")


    ######################  Upload an X-ray image  ######################
    elif app_mode == "Upload an X-ray image":
        st.warning("Warning: Please upload a JPEG or PNG file with a size less than 1 MB.")
        # Prompt the user to upload an image
        uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

        # If the user has uploaded a file, make a prediction
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded X-ray', use_column_width=True)

            # Make a prediction
            prediction = predict(uploaded_file)
            # Display the results
            if prediction == "covid":
                st.warning("Warning: This X-ray image shows evidence of COVID-19.")
            elif prediction == "virus":
                st.warning("Warning: This X-ray image shows evidence of a Virus.")
            elif prediction == "normal":
                st.success("Success: This X-ray image shows no evidence of any virus.")
if __name__ == "__main__":
    app()
