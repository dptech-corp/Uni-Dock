import os
import json
import argparse
def parse_pdbqt(pdbqt_path):
    with open(pdbqt_path, 'r') as f:
        lines = f.readlines()

    models_data = []
    model_data = {}
    vina_results = []
    atom_x = []
    atom_y = []
    atom_z = []
    atom_type = []
    model_idx = 0

    for line in lines:
        line = line.strip()
        if line.startswith("MODEL"):
            model_data = {}
            model_idx += 1
            vina_results = []
            atom_x = []
            atom_y = []
            atom_z = []
            atom_type = []
        elif line.startswith("REMARK VINA RESULT:"):
            vina_results.append(float(line.split()[3]))
            vina_results.append(float(line.split()[4]))
            vina_results.append(float(line.split()[5]))
        elif line.startswith("ATOM"):
            atom_info = line.split()
            atom_x.append(float(atom_info[5]))
            atom_y.append(float(atom_info[6]))
            atom_z.append(float(atom_info[7]))
            atom_type.append(atom_info[-1])
        elif line.startswith("ENDMDL"):
            model_data['VINA Results'] = vina_results
            model_data['Atom Coordinates X'] = atom_x
            model_data['Atom Coordinates Y'] = atom_y
            model_data['Atom Coordinates Z'] = atom_z
            model_data['Atom Types'] = atom_type
            models_data.append({"Model " + str(model_idx): model_data})

    return models_data

# Directory with pdbqt files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get result.")
    parser.add_argument("-files", "--input_folder", help="pdbqt_files_dir", required=True)
    parser.add_argument("-out", "--result_out",  help="result_out",required=True)

    args = parser.parse_args()
    # Files to parse
    pdbqt_files_dir = "result/def"
    pdbqt_files_dir = args.input_folder
    pdbqt_files = []  
    for filename in os.listdir(pdbqt_files_dir):
        if filename.endswith(".pdbqt"):
            # pdbqt_files.append( os.path.join(pdbqt_files_dir, filename))
            pdbqt_files.append( filename)
    print(pdbqt_files)

    parsed_data = {}

    # Parse each pdbqt file
    for pdbqt_file in pdbqt_files:
        pdbqt_path = os.path.join(pdbqt_files_dir, pdbqt_file)
        parsed_data[pdbqt_file] = parse_pdbqt(pdbqt_path)

    # Save the data as a JSON file
    json_path = 'result.json'
    json_path = args.result_out
    with open(json_path, 'w') as f:
        json.dump(parsed_data, f, separators=(',', ':'))




