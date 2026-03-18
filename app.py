import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd  # Added pandas for the clean data table
from collections import Counter

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Marine Litter Classifier", page_icon="🌊")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return YOLO('yolov8.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}. Did you put 'best.pt' in the same folder?")
        return None

model = load_model()

# --- 3. UI LAYOUT ---
st.title("Marine Litter Detection & Classification")
st.write("Upload an image of marine debris to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the original image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
        #st.info("Processing...")

    # --- 4. PREDICTION LOGIC ---
    results = model(image)
    
    # --- 5. EXTRACT RESULTS ---
    detected_classes = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            detected_classes.append(class_name)

    # --- 6. DISPLAY RESULTS (UPDATED) ---
    with col2:
        if len(detected_classes) > 0:
            st.success("Analysis Complete!")
            
            # Calculate counts for every class found
            class_counts = Counter(detected_classes)
            
                       # Create a clean DataFrame for display
            df = pd.DataFrame(class_counts.items(), columns=['Class Type', 'Count'])
            
            # Append a total row showing total detected objects
            total_count = df['Count'].sum()
            total_row = pd.DataFrame([['Total', total_count]], columns=df.columns)
            df = pd.concat([df, total_row], ignore_index=True)
            
            # Display the Summary Table
            st.write("### Detection Summary")
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Plot the image with bounding boxes
            res_plotted = results[0].plot()
            st.image(res_plotted, caption='AI Detection View', channels="BGR", use_container_width=True)
        else:
            st.warning("No litter detected. Try another image.")