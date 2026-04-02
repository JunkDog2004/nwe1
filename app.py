import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Optional: Import your actual scripts here once they are ready
# from flaml import AutoML 
# import agent 
# import pipeline

# --- Page Configuration ---
st.set_page_config(page_title="AutoML App", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation 🧭")
st.sidebar.write("Switch between building your model and using it.")
st.sidebar.write("Go to:")
page = st.sidebar.radio(
    "Navigation",
    ["🛠️ Build Pipeline", "🚀 Predictor UI"],
    label_visibility="collapsed"
)

# ==========================================
# PAGE 1: BUILD PIPELINE
# ==========================================
if page == "🛠️ Build Pipeline":
    st.title("🛠️ Build Pipeline")
    st.write("Upload your data, clean it with Gemini, and train it with FLAML.")
    
    st.divider()

    # --- 1. Data Upload & EDA ---
    st.subheader("Data Upload & Exploration")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "txt", "xlsx"])

    if uploaded_file is not None:
        # Read the file based on its extension
        file_extension = uploaded_file.name.split('.')[-1]
        
        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "txt":
                df = pd.read_csv(uploaded_file, sep='\t') 
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file)
                
            st.success("Data loaded successfully!")
            st.dataframe(df.head())
            
            with st.expander("📊 View Exploratory Data Analysis (EDA)"):
                st.write("**Statistical Summary:**")
                st.dataframe(df.describe())
                
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                if not numeric_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Numeric Feature Distributions**")
                        st.line_chart(numeric_df)
                    
                    with col2:
                        st.write("**Correlation Matrix**")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", ax=ax, fmt=".2f")
                        fig.patch.set_facecolor('none') 
                        ax.set_facecolor('none')
                        st.pyplot(fig)
                        
            # Save dataframe to session state so the AI and AutoML tools can access it later
            st.session_state['dataset'] = df

        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.divider()

    # --- 2. Action Buttons Layout (Matching your screenshot) ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 1. AI Task & Cleaning")
        if st.button("Ask Gemini for Analysis"):
            with st.spinner("Agent is thinking..."):
                # TODO: Connect your agent.py here
                # example: response = agent.run_analysis(st.session_state['dataset'])
                time.sleep(2) # Simulating processing time
                st.info("Gemini analysis complete! (Placeholder: Connect agent.py to show real results here)")

    with col2:
        st.subheader("🤖 2. AutoML Training")
        if st.button("Run FLAML Training"):
            if 'dataset' not in st.session_state:
                st.error("Please upload a dataset first!")
            else:
                with st.spinner("FLAML is training..."):
                    # TODO: Connect your pipeline.py or FLAML code here
                    time.sleep(3) # Simulating training time
                    
                    # Dummy model creation for demonstration purposes
                    # Replace this with your actual fitted FLAML model: `model = automl`
                    dummy_model = {"model_name": "FLAML_Classifier", "status": "trained"} 
                    
                    # Save the model
                    with open("trained_model.pkl", "wb") as f:
                        pickle.dump(dummy_model, f)
                        
                    st.success("Training complete! Model saved successfully.")

# ==========================================
# PAGE 2: PREDICTOR UI
# ==========================================
elif page == "🚀 Predictor UI":
    st.title("🚀 Model Deployment & Prediction")
    st.write("Use your trained model to make predictions on new data.")
    
    st.divider()

    try:
        # Attempt to load the model
        with open("trained_model.pkl", "rb") as f:
            model = pickle.load(f)
            
        st.success("Trained model found and loaded successfully!")
        
        # --- Prediction Interface ---
        st.subheader("Make a Prediction")
        st.write("Enter the values for your features below:")
        
        # Example input fields (You will need to adjust these based on your dataset's columns)
        feature_1 = st.number_input("Feature 1", value=0.0)
        feature_2 = st.number_input("Feature 2", value=0.0)
        
        if st.button("Predict"):
            # TODO: Run the actual prediction
            # prediction = model.predict([[feature_1, feature_2]])
            st.write(f"**Prediction Result:** [Connect model.predict() here]")
            
    except FileNotFoundError:
        # If the file isn't there, show your exact warning
        st.warning("No trained model found. Please go to the 'Build Pipeline' page and train a model first.")
