import streamlit as st
from PIL import Image
import Model_load as Ml
import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import glob

# Set page config
st.set_page_config(
    page_title="MNIST Digit Classifier", 
    page_icon="🔢", 
    layout="centered"
)

# Title and description
st.title("🔢 MNIST Classifier")
st.caption("Upload or select a handwritten digit image for prediction")

# Custom CSS to make UI more compact
st.markdown("""
<style>
    .stApp > header {
        background-color: transparent;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 800px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 35px;
        padding: 0px 12px;
    }
    .stMetric {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    .stMetric > div {
        background-color: transparent !important;
        border: none !important;
    }
    .stDataFrame {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📤 Image")
    
    # Add tabs for upload vs local library
    tab1, tab2 = st.tabs(["Library", "Upload"])
    
    selected_image_path = None
    uploaded_file = None
    
    with tab1:
        # Get images from local library
        sample_images_dir = "sample_images"
        if os.path.exists(sample_images_dir):
            image_files = glob.glob(os.path.join(sample_images_dir, "*"))
            image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            if image_files:
                # Create a selectbox for choosing images
                image_names = [os.path.basename(f) for f in image_files]
                selected_image_name = st.selectbox(
                    "Sample images:",
                    ["Select..."] + image_names,
                    label_visibility="collapsed"
                )
                
                if selected_image_name != "Select...":
                    selected_image_path = os.path.join(sample_images_dir, selected_image_name)
                    
                    # Display the selected image
                    image = Image.open(selected_image_path)
                    st.image(image, caption=selected_image_name, width=200)
                    
                    # Show image info in smaller text
                    st.caption(f"Size: {image.size}")
            else:
                st.warning("No sample images found.")
        else:
            st.warning("Sample images folder not found.")
    
    with tab2:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose image file", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a handwritten digit image"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded", width=200)
            
            # Save uploaded file temporarily for prediction
            with open("temp_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_image_path = "temp_image.png"
with col2:
    st.subheader("🎯 Predictions")
    
    if selected_image_path is not None:
        # Get predictions
        with st.spinner("Analyzing..."):
            prediction_result = Ml.predict_mnist_probabilities(selected_image_path)
        
        if prediction_result.startswith("Error"):
            st.error(prediction_result)
        else:
            # Parse the prediction results
            lines = prediction_result.strip().split('\n')
            digits = []
            probabilities = []
            
            for line in lines:
                digit, prob = line.split(': ')
                digits.append(int(digit))
                probabilities.append(float(prob))
            
            # Find the predicted digit
            predicted_digit = digits[np.argmax(probabilities)]
            max_probability = max(probabilities)
            
            # Display prediction in compact format
            st.success(f"**Predicted Digit: {predicted_digit}** (Confidence: {max_probability:.1%})")
            
            # Create and display smaller probability chart
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(digits, probabilities, color='lightblue', alpha=0.8)
            
            # Highlight the predicted digit
            bars[predicted_digit].set_color('red')
            
            ax.set_xlabel('Digit', fontsize=10)
            ax.set_ylabel('Probability', fontsize=10)
            ax.set_title('Prediction Probabilities', fontsize=11)
            ax.set_xticks(digits)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add probability values on top of bars (smaller font)
            for i, prob in enumerate(probabilities):
                if prob > 0.05:  # Only show labels for significant probabilities
                    ax.text(i, prob + 0.01, f'{prob:.2f}', ha='center', va='bottom', fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Display compact probabilities table
            with st.expander("📊 Detailed Probabilities", expanded=False):
                prob_data = {
                    'Digit': digits,
                    'Probability': [f"{prob:.3f}" for prob in probabilities],
                    'Percentage': [f"{prob:.1%}" for prob in probabilities]
                }
                st.dataframe(prob_data, use_container_width=True, hide_index=True)
    
    else:
        st.info("👆 Select an image to see predictions")

# Compact tips section
with st.expander("💡 Usage Tips", expanded=False):
    st.markdown("""
    **For better results:**
    - Use clear, high-contrast digit images
    - Single digits work best
    - White backgrounds with dark digits preferred
    - Images are resized to 28x28 pixels
    
    **Adding sample images:**
    - Place images in `sample_images` folder
    - Refresh app to see new images
    - Use descriptive filenames
    """)

# Clean up temporary file
import os
if os.path.exists("temp_image.png"):
    try:
        os.remove("temp_image.png")
    except:
        pass


    