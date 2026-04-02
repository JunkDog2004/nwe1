import streamlit as st
import pandas as pd
import joblib
import json
import os

def load_artifacts():
    """
    Loads the trained model and the metadata JSON file from the outputs folder.
    """
    model_path = "outputs/best_model.pkl"
    meta_path = "outputs/model_meta.json"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model or metadata not found. Please build the pipeline first.")
        
    # Load the FLAML model
    model = joblib.load(model_path)
    
    # Load the metadata (features, task type, etc.)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    return model, metadata

def render_page():
    """
    The main UI function called by app.py. Renders the dynamic input form.
    """
    st.subheader("🔮 Live Prediction Endpoint")
    st.markdown("Enter the feature values below to get a real-time prediction from your trained model.")
    
    try:
        model, metadata = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return
        
    features = metadata.get("features", [])
    task_type = metadata.get("task_type", "Unknown")
    
    st.info(f"**Model Type:** {metadata.get('best_estimator', 'Unknown')} | **Task:** {task_type.capitalize()}")
    
    # Dictionary to hold the user's inputs
    user_inputs = {}
    
    # Create a Streamlit Form so the app doesn't refresh on every single keystroke
    with st.form("prediction_form"):
        st.write("### Input Features")
        
        # Dynamically generate an input box for every feature the model expects
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(features):
            # Alternate placing inputs in column 1 and column 2 to save screen space
            if i % 2 == 0:
                with col1:
                    # Using text_input to handle both numbers and categorical strings safely
                    user_inputs[feature] = st.text_input(f"{feature}", placeholder="Enter value...")
            else:
                with col2:
                    user_inputs[feature] = st.text_input(f"{feature}", placeholder="Enter value...")
                    
        # The submit button for the form
        submit_button = st.form_submit_button(label="Generate Prediction", type="primary")
        
    # When the user clicks the button...
    if submit_button:
        try:
            # 1. Convert the dictionary of inputs into a Pandas DataFrame (1 row)
            input_df = pd.DataFrame([user_inputs])
            
            # 2. Attempt to convert strings to numbers where appropriate
            # (If a user types "5", it becomes an integer. If they type "Male", it stays a string)
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='ignore')
                
            # 3. Make the prediction!
            prediction = model.predict(input_df)
            
            # 4. Display the result beautifully
            st.divider()
            st.subheader("🎯 Prediction Result")
            
            if task_type.lower() == "classification":
                st.success(f"The model classifies this as: **{prediction[0]}**")
            else:
                # For regression, round the number to 2 decimal places for neatness
                st.success(f"The model predicts a value of: **{prediction[0]:.2f}**")
                
        except Exception as e:
            st.error(f"An error occurred during prediction. Did you leave a required field blank? \n\n Error details: {str(e)}")
