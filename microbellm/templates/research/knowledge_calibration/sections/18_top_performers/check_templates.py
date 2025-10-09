#!/usr/bin/env python3
"""Check what template names are in the API data."""

import json
import urllib.request
import os

api_url = 'http://localhost:5050'
endpoint = api_url.rstrip('/') + '/api/knowledge_analysis_data'

try:
    with urllib.request.urlopen(endpoint, timeout=10) as resp:
        if resp.status == 200:
            data = json.loads(resp.read().decode('utf-8'))

            # Extract unique template names
            templates = set()
            for file_name, file_data in data.get('knowledge_analysis', {}).items():
                if file_data.get('types'):
                    for input_type, template_data in file_data['types'].items():
                        for template_name in template_data.keys():
                            templates.add(template_name)

            print("Unique template names found:")
            for t in sorted(templates):
                print(f"  - {t}")

except Exception as e:
    print(f"Error: {e}")