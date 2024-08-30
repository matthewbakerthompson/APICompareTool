"""
File: api_handler.py
Author: Matthew Thompson
Created on: 11/21/2023
Last updated: 12/8/2023

Description:
    This module contains the function to handle API requests for the APIDataValidationReportGenerator.
    It retrieves data from a specified API endpoint and handles potential errors during the request.

Contact:
    matthewbakerthompson@gmail.com
    mbthompson@albany.edu
"""

import requests
import logging

logger = logging.getLogger(__name__)

def get_api_data(user_id):
    api_url = f'https://jsonplaceholder.typicode.com/users/{user_id}'
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Flatten nested fields
        flat_data = {
            'id': data.get('id'),
            'name': data.get('name'),
            'username': data.get('username'),
            'email': data.get('email'),
            'street': data.get('address', {}).get('street'),
            'suite': data.get('address', {}).get('suite'),
            'city': data.get('address', {}).get('city'),
            'zipcode': data.get('address', {}).get('zipcode'),
            'lat': data.get('address', {}).get('geo', {}).get('lat'),
            'lng': data.get('address', {}).get('geo', {}).get('lng')
            # Add other fields as needed
        }
        return flat_data
    except Exception as e:
        logger.error(f"An error occurred while fetching API data: {e}")
        return None
