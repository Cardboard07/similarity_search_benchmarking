import json

with open(r"C:\Users\hrish\Projects\research_projects\duplicate_detection\fingerprint_dataset\parsed_data\test\q1_1_2.json",'r') as f:
    data=json.load(f)
print(len(data['relative_tokens']))