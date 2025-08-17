import json
import glob
import os

# Find all JSON files in the current directory
json_files = glob.glob('*.json')

if not json_files:
    print("No JSON files found in the current directory.")
else:
    for filepath in json_files:
        try:
            # Read the JSON file
            with open(filepath, 'r') as file:
                data = json.load(file)

            # Sort the dictionary by UUID (keys)
            sorted_data = dict(sorted(data.items()))

            # Write the sorted data back to the same JSON file
            with open(filepath, 'w') as file:
                json.dump(sorted_data, file, indent=2)
            print(f"Successfully sorted {filepath}")

        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except json.JSONDecodeError:
            print(f"Error: File '{filepath}' contains invalid JSON.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{filepath}': {str(e)}")