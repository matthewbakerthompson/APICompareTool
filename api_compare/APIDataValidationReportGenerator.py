"""
File: APIDataValidationReportGenerator.py
Author: Matthew Thompson
Created on: 11/21/2023
Last updated: 12/8/2023

Description:
    This script processes an uploaded Excel file, performs data validation, generates an HTML report,
    and uploads the report to Azure Blob Storage.

Contact:
    matthewbakerthompson@gmail.com
    mbthompson@albany.edu
"""


import pandas as pd
from .api_handler import get_api_data
from .report_generator import generate_html_report_with_sas
from django.conf import settings
import uuid

def load_excel_data(file):
    df = pd.read_excel(file)
    return df

def compare_data(excel_data):
    mismatches = {}
    complete_matches = []

    # Get the list of fields from the Excel data columns
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
                print(f"Complete match found for User ID: {user_id}")
            else:
                mismatches[user_id] = user_mismatches
                print(f"Mismatches for User ID {user_id}: {user_mismatches}")
        else:
            mismatches[user_id] = ['No API data available']
            print(f"No API data available for User ID: {user_id}")

    print(f"Complete Matches: {complete_matches}")
    print(f"Partial Mismatches: {mismatches}")
    return complete_matches, mismatches


def generate_report_from_uploaded_file(uploaded_file):
    excel_data = load_excel_data(uploaded_file)
    complete_matches, mismatches = compare_data(excel_data)

    report_url_with_sas = generate_html_report_with_sas(complete_matches, mismatches, excel_data)
    has_mismatches = bool(mismatches)

    return report_url_with_sas, has_mismatches

 