import streamlit as st
import pandas as pd
# Assuming api_handler and report_generator are your modules
from api_handler import get_api_data  
from report_generator import generate_html_report  

# Function to load Excel data
def load_excel_data(file):
    df = pd.read_excel(file)
    return df

# Function to compare data
def compare_data(excel_data):
    mismatches = {}
    complete_matches = []

    excel_fields = excel_data.columns.tolist()

    for index, row in excel_data.iterrows():
        user_id = row['id']
        user_api_data = get_api_data(user_id)
        if user_api_data:
            user_mismatches = []

            for field in excel_fields:
                if field in user_api_data:
                    excel_value = str(row.get(field, '')).strip()
                    api_value = str(user_api_data.get(field, '')).strip()

                    if excel_value and api_value and excel_value != api_value:
                        user_mismatches.append(f"Field '{field}' mismatch: Excel '{excel_value}' vs API '{api_value}'")

            if not user_mismatches:
                complete_matches.append(user_id)
            else:
                mismatches[user_id] = user_mismatches
        else:
            mismatches[user_id] = ['No API data available']

    return complete_matches, mismatches

# Main function for Streamlit app
def main():
    st.title("API Data Validation Report Generator")

    uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

    if uploaded_file is not None:
        excel_data = load_excel_data(uploaded_file)
        complete_matches, mismatches = compare_data(excel_data)

        st.write("## Complete Matches")
        st.write(complete_matches)

        st.write("## Mismatches")
        st.write(mismatches)

        # Generate and display report
    try:
        report_content = generate_html_report(complete_matches, mismatches, excel_data)
        st.markdown(report_content, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while generating the report: {e}")

if name == "main":
    main()