import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configure the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

genai.configure(api_key=api_key)

# Initialize the Gemini model (Using gemini-1.5-pro for complex reasoning/coding)
model = genai.GenerativeModel('gemini-1.5-pro')

def get_dataframe_summary(df):
    """Utility function to create a text summary of the dataframe for the LLM."""
    summary = f"Columns and Data Types:\n{df.dtypes.to_string()}\n\n"
    summary += f"Missing Values:\n{df.isnull().sum().to_string()}\n\n"
    summary += f"First 3 Rows:\n{df.head(3).to_string()}"
    return summary

def detect_task_type(df, target_column):
    """Asks Gemini to determine if this is Classification or Regression."""
    df_summary = get_dataframe_summary(df)
    
    prompt = f"""
    You are an expert Data Scientist. Review the following dataset summary.
    The user wants to predict the target column: '{target_column}'.
    
    Dataset Summary:
    {df_summary}
    
    Based on the data type and values of the target column, is this a 'Classification' or 'Regression' problem?
    Respond with EXACTLY one word: either Classification or Regression.
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()

def get_cleaning_suggestions(df):
    """Asks Gemini for human-readable data cleaning advice."""
    df_summary = get_dataframe_summary(df)
    
    prompt = f"""
    You are an expert Data Engineer. Review the following dataset summary:
    
    {df_summary}
    
    Provide 3 to 5 clear, bulleted suggestions on how to clean this dataset before training a machine learning model.
    Focus on handling missing values, encoding categorical variables, or dropping useless columns.
    Keep the response concise and easy to read.
    """
    
    response = model.generate_content(prompt)
    return response.text

def generate_cleaning_code(df):
    """Asks Gemini to write the actual Pandas Python code to clean the data."""
    df_summary = get_dataframe_summary(df)
    
    prompt = f"""
    You are a Python Data Engineering assistant. 
    Write a Python function named `clean_data(df)` that takes a pandas DataFrame 'df' and cleans it based on best practices.
    
    Dataset Summary:
    {df_summary}
    
    Requirements:
    1. Handle missing values appropriately (impute or drop).
    2. Convert categorical string columns to numeric (using LabelEncoder or dummy variables).
    3. Return the cleaned dataframe.
    4. ONLY output valid Python code inside a markdown code block (```python ... ```). 
    Do not include any explanations or conversational text.
    """
    
    response = model.generate_content(prompt)
    
    # Extract just the code from the markdown block
    code_text = response.text
    if "```python" in code_text:
        code_text = code_text.split("```python")[1].split("```")[0].strip()
    elif "```" in code_text:
        code_text = code_text.split("```")[1].split("```")[0].strip()
        
    return code_text
