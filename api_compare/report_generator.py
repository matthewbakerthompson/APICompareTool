"""
File: reporrt_generator.py
Author: Matthew Thompson
Created on: 11/21/2023
Last updated: 12/8/2023

Description:
    This script takes results of a comparison done in APIDataValidationReportGenerator.py and passes them to 
    report_template.html for a downloadable report artifact for the user. 

Contact:
    matthewbakerthompson@gmail.com
    mbthompson@albany.edu
"""

import json
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from django.conf import settings
from azure.storage.blob import BlobServiceClient
import os
import uuid

def generate_html_report_with_sas(complete_matches, partial_mismatches, excel_data_list):
    """
    Generates an HTML report from the provided data and uploads it to Azure Blob Storage.

    Args:
        complete_matches (list): A list of IDs that completely matched between the Excel file and API data.
        partial_mismatches (dict): A dictionary containing IDs and their corresponding mismatched fields.
        excel_data_list (list): A list of data extracted from the uploaded Excel file.

    Returns:
        str: The URL of the uploaded HTML report with a SAS token for access.

    Raises:
        ValueError: If the Azure SAS URL is not set in the settings.
    """

    # Retrieve Azure Blob Storage settings
    azure_sas_url = settings.AZURE_BLOB_SERVICE_SAS_URL
    if not azure_sas_url:
        raise ValueError("Azure SAS URL is not set")
    azure_container_name = settings.AZURE_CONTAINER_NAME
    
    # Generate a unique filename for the report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"API_Compare_Report_{timestamp}.html"
    
    # Calculate statistics for the report
    total_ids = len(excel_data_list)
    passed_ids = len(complete_matches)
    failed_ids = len(partial_mismatches)
    pass_percentage = (passed_ids / total_ids) * 100 if total_ids else 0
    fail_percentage = (failed_ids / total_ids) * 100 if total_ids else 0

    # Prepare data for the report template
    data = {
        "total_ids": total_ids,
        "passed_ids": passed_ids,
        "failed_ids": failed_ids,
        "pass_percentage": pass_percentage,
        "fail_percentage": fail_percentage,
        "complete_matches": complete_matches,   
        "partial_mismatches": partial_mismatches,
        "timestamp": timestamp
    }
 
    # Convert data to JSON for JavaScript use in the report
    data_for_js = json.dumps({
        "pass_percentage": pass_percentage,
        "fail_percentage": fail_percentage,
    })
    print(f"complete matches in report_generator.py: {complete_matches}")
    # Load the HTML report template and render it with data
    templates_directory = os.path.join(settings.BASE_DIR, 'app', 'templates', 'app')
    env = Environment(loader=FileSystemLoader(templates_directory))
    template = env.get_template("report_template.html")
    html_content = template.render(data=data, data_for_js=data_for_js, complete_matches=complete_matches)

    # Initialize BlobServiceClient with SAS URL and upload the report
    blob_service_client = BlobServiceClient(account_url=azure_sas_url)
    blob_client = blob_service_client.get_blob_client(container=azure_container_name, blob=report_filename)
    try:
        blob_client.upload_blob(html_content.encode('utf-8'), blob_type="BlockBlob")
    except Exception as e:
        print(f"An error occurred during blob upload: {e}")
        raise e

    # Construct the report URL with SAS token
    report_url_with_sas = f"{blob_client.url}"
    print(f"Report uploaded: {report_url_with_sas}")

    return report_url_with_sas
