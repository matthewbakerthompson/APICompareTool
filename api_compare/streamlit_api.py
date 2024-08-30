import streamlit as st
import pandas as pd
import base64
from io import BytesIO

# Function to download data in CSV format
def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return output

# Function to download data in Excel format
def to_excel(df):
    output = BytesIO()
    # Using a context manager to handle the Excel file creation
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        # No need for writer.save()
    output.seek(0)
    return output


def main():
    st.title("API Comparison Report Generator")

    # Sample Data (replace this with your actual data processing logic)
    data = {
        "Category": ["A", "B", "C"],
        "Value": [10, 20, 30]
    }
    df = pd.DataFrame(data)

    # Display the report in Streamlit
    st.write("## Report")
    st.dataframe(df)

    # Generate download link for the report
    st.write("## Download Report")

    csv = to_csv(df)  # Get the CSV file
    st.download_button(label="Download as CSV", data=csv, file_name="report.csv", mime='text/csv')

    excel = to_excel(df)  # Get the Excel file
    st.download_button(label="Download as Excel", data=excel, file_name="report.xlsx", mime='application/vnd.ms-excel')

if __name__ == "__main__":
    main()
