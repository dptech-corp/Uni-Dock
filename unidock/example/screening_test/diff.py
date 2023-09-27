import json
import os
import argparse
def float_equal(a, b, epsilon=1e-5):
    return abs(a - b) < epsilon

def compare_dicts(dict1, dict2, path="", focus_on=None):
    for key in dict1.keys():
        if key not in dict2:
            return False, f"Key '{key}' not found in second dictionary at path {path}"
        
        new_path = f"{path}.{key}" if path else key
        
        # Skip the keys that are not in focus if focus_on is specified
        if focus_on and new_path not in focus_on:
            continue
        
        value1, value2 = dict1[key], dict2[key]
        
        if type(value1) != type(value2):
            return False, f"Type mismatch at {new_path}: {type(value1).__name__} vs {type(value2).__name__}"
        
        if isinstance(value1, dict):
            equal, reason = compare_dicts(value1, value2, new_path, focus_on)
            if not equal:
                return equal, reason
        
        elif isinstance(value1, list):
            if len(value1) != len(value2):
                return False, f"List length mismatch at {new_path}: {len(value1)} vs {len(value2)}"
            
            for i, (item1, item2) in enumerate(zip(value1, value2)):
                list_path = f"{new_path}[{i}]"
                
                if focus_on and list_path not in focus_on:
                    print(list_path)
                    continue
                
                if isinstance(item1, dict):
                    equal, reason = compare_dicts(item1, item2, list_path, focus_on)
                    if not equal:
                        return equal, reason
                
                elif isinstance(item1, float):
                    if not float_equal(item1, item2):
                        return False, f"Float value mismatch at {list_path}: {item1} vs {item2}"
                        
                elif item1 != item2:
                    return False, f"Value mismatch at {list_path}: {item1} vs {item2}"
        
        else:
            if isinstance(value1, float):
                if not float_equal(value1, value2):
                    return False, f"Float value mismatch at {new_path}: {value1} vs {value2}"
            elif value1 != value2:
                return False, f"Value mismatch at {new_path}: {value1} vs {value2}"

    return True, "Dicts are equal"


def compare_json_files(file1, file2, focus_on=None):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        dict1 = json.load(f1)
        dict2 = json.load(f2)
    
    return compare_dicts(dict1, dict2, focus_on=focus_on)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result diff.")
    parser.add_argument("-f1", "--file1", help="file1", required=True)
    parser.add_argument("-f2", "--file2", help="file2",required=True)
    parser.add_argument("-focus","--focus_on", help="Comma-separated list of keys to focus on", default=None)
    args = parser.parse_args()
    file1 = args.file1
    file2 = args.file2
    args = parser.parse_args()
    focus_on = set(args.focus_on.split(',')) if args.focus_on else None
    print(focus_on)
    equal, reason = compare_json_files(file1, file2, focus_on)
    if equal:
        print("The JSON files are equal.")
    else:
        print(f"The JSON files are not equal. Reason: {reason}")
