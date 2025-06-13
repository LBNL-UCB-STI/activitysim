import csv
import sys
import yaml
from pathlib import Path

def csvs_to_yaml(csv_paths, output_yaml):
    result = {'tunable_coefficients': {}}
    for csv_path in csv_paths:
        csv_name = Path(csv_path).name
        result['tunable_coefficients'][csv_name] = {}
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coef = row['coefficient_name']
                value = float(row['value'])
                result['tunable_coefficients'][csv_name][coef] = {
                    'initial_value': value,
                    'bounds': [-2.0, 2.0]
                }
    with open(output_yaml, 'w') as f:
        yaml.dump(result, f, sort_keys=False)

if __name__ == '__main__':
    # Usage: python script.py output.yaml file1.csv file2.csv ...
    if len(sys.argv) < 3:
        print("Usage: python script.py output.yaml file1.csv file2.csv ...")
        sys.exit(1)
    output_yaml = sys.argv[1]
    csv_files = sys.argv[2:]
    csvs_to_yaml(csv_files, output_yaml)