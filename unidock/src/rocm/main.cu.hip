/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>,
		   The Olson Lab,
		   The Scripps Research Institute

*/

#include <iostream>
#include <string>
#include <vector> // ligand paths
#include <exception>
#include <boost/program_options.hpp>
#include "vina.h"
#include "utils.h"
#include "scoring_function.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
// #include <device_launch_parameters.h>

struct usage_error : public std::runtime_error {
	usage_error(const std::string& message) : std::runtime_error(message) {}
};

struct options_occurrence {
	bool some;
	bool all;
	options_occurrence() : some(false), all(true) {} // convenience
	options_occurrence& operator+=(const options_occurrence& x) {
		some = some || x.some;
		all  = all  && x.all;
		return *this;
	}
};

options_occurrence get_occurrence(boost::program_options::variables_map& vm, boost::program_options::options_description& d) {
	options_occurrence tmp;
	VINA_FOR_IN(i, d.options())
		if(vm.count((*d.options()[i]).long_name()))
			tmp.some = true;
		else
			tmp.all = false;
	return tmp;
}

void check_occurrence(boost::program_options::variables_map& vm, boost::program_options::options_description& d) {
	VINA_FOR_IN(i, d.options()) {
		const std::string& str = (*d.options()[i]).long_name();
		if(!vm.count(str))
			std::cerr << "Required parameter --" << str << " is missing!\n";
	}
}

int predict_peak_memory(int batch_size, int exhaustiveness, int all_atom2_numbers, bool use_v100 = true, bool multi_bias = false){
	if (!multi_bias){
		if (use_v100){
			return 1.214869*batch_size + .0038522*exhaustiveness*batch_size + .011978*all_atom2_numbers + 20017.72; // this is based on V100, 32G
		}
		else {
			return 1.166067*batch_size + .0038676*exhaustiveness*batch_size + .0119598*all_atom2_numbers + 5313.848; // this is based on T4, 16G
		}
	}
	else{
		if (use_v100){
			return 65.214869*batch_size + .0038522*exhaustiveness*batch_size + .011978*all_atom2_numbers + 20017.72; // this is based on V100, 32G
		}
		else {
			return 65.166067*batch_size + .0038676*exhaustiveness*batch_size + .0119598*all_atom2_numbers + 5313.848; // this is based on T4, 16G
		}
	}
	return 0;
}

int main(int argc, char* argv[]) {
	using namespace boost::program_options;
	const std::string git_version = VERSION;
	const std::string version_string = "Uni-Dock " + git_version;
	const std::string error_message = "\n\n\
Please report bugs through the Issue Tracker on GitHub \n\
(https://github.com/dptech-corp/Uni-Dock/issues)., so\n\
that this problem can be resolved. The reproducibility of the\n\
error may be vital, so please remember to include the following in\n\
your problem report:\n\
* the EXACT error message,\n\
* your version of the program,\n\
* the type of computer system you are running it on,\n\
* all command line options,\n\
* configuration file (if used),\n\
* ligand file as PDBQT,\n\
* receptor file as PDBQT,\n\
* flexible side chains file as PDBQT (if used),\n\
* output file as PDBQT (if any),\n\
* input (if possible),\n\
* random seed the program used (this is printed when the program starts).\n\
\n\
Thank you!\n";

	const std::string cite_message = "\
#################################################################\n\
# If you used Uni-Dock in your work, please cite:               #\n\
#                                                               #\n\
# Uni-Dock, xxxx, (2022) DOI xxxx							    #\n\
#                                                               #\n\
#                                                               #\n\
# J. Eberhardt, D. Santos-Martins, A. F. Tillack, and S. Forli  #\n\
# AutoDock Vina 1.2.0: New Docking Methods, Expanded Force      #\n\
# Field, and Python Bindings, J. Chem. Inf. Model. (2021)       #\n\
# DOI 10.1021/acs.jcim.1c00203                                  #\n\
#                                                               #\n\
# O. Trott, A. J. Olson,                                        #\n\
# AutoDock Vina: improving the speed and accuracy of docking    #\n\
# with a new scoring function, efficient optimization and       #\n\
# multithreading, J. Comp. Chem. (2010)                         #\n\
# DOI 10.1002/jcc.21334                                         #\n\
#                                                               #\n\
# Please see https://github.com/dptech-corp/Uni-Dock/ for       #\n\
# more information.                                             #\n\
#################################################################\n";

	try {
		std::string rigid_name;
		std::string flex_name;
		std::string config_name;
		std::string out_name;
		std::vector<std::string> gpu_out_name;
		std::string out_dir;
		std::string out_maps;
		std::vector<std::string> ligand_names;
		std::string ligand_index; // path to a text file, containing paths to ligands files
		std::vector<std::string> batch_ligand_names;
		std::vector<std::string> gpu_batch_ligand_names;
		std::string maps;
		std::string sf_name = "vina";
		std::string search_mode;
		double center_x;
		double center_y;
		double center_z;
		double size_x;
		double size_y;
		double size_z;
		int cpu = 0;
		int seed = 0;
		int exhaustiveness = 8;
		int max_evals = 0;
		int verbosity = 1;
		int num_modes = 9;
		double min_rmsd = 1.0;
		double energy_range = 3.0;
		double grid_spacing = 0.375;
		double buffer_size = 4;
		int max_step = 0;
		int max_gpu_memory = 0;
		int refine_step = 5;

		// autodock4.2 weights
		double weight_ad4_vdw   = 0.1662;
		double weight_ad4_hb    = 0.1209;
		double weight_ad4_elec  = 0.1406;
		double weight_ad4_dsolv = 0.1322;
		double weight_ad4_rot   = 0.2983;

		// vina weights
		double weight_gauss1      = -0.035579;
		double weight_gauss2      = -0.005156;
		double weight_repulsion   =  0.840245;
		double weight_hydrophobic = -0.035069;
		double weight_hydrogen    = -0.587439;
		double weight_rot         =  0.05846;

		// vinardo weights
		double weight_vinardo_gauss1 = -0.045;
		double weight_vinardo_repulsion = 0.8;
		double weight_vinardo_hydrophobic = -0.035;
		double weight_vinardo_hydrogen = -0.600;

		// macrocycle closure
		double weight_glue        = 50.000000; // linear attraction

		bool score_only = false;
		bool local_only = false;
		bool no_refine = false;
		bool force_even_voxels = false;
		bool randomize_only = false;
		bool help = false;
		bool help_advanced = false;
		bool version = false; // FIXME
		bool autobox = false;
		variables_map vm;

		// bias
		std::string bias_file;
		bool multi_bias;

		positional_options_description positional; // remains empty

		options_description inputs("Input");
		inputs.add_options()
			("receptor", value<std::string>(&rigid_name), "rigid part of the receptor (PDBQT)")
			("flex", value<std::string>(&flex_name), "flexible side chains, if any (PDBQT)")
			("ligand", value< std::vector<std::string> >(&ligand_names)->multitoken(), "ligand (PDBQT)")
			("ligand_index",value<std::string>(&ligand_index),"file containing paths to ligands")
			("batch", value< std::vector<std::string> >(&batch_ligand_names)->multitoken(), "batch ligand (PDBQT)")
			("gpu_batch", value< std::vector<std::string> >(&gpu_batch_ligand_names)->multitoken(), "gpu batch ligand (PDBQT)")
			("scoring", value<std::string>(&sf_name)->default_value(sf_name), "scoring function (ad4, vina or vinardo)")
		;
		//options_description search_area("Search area (required, except with --score_only)");
		options_description search_area("Search space (required)");
		search_area.add_options()
			("maps", value<std::string>(&maps), "affinity maps for the autodock4.2 (ad4) or vina scoring function")
			("center_x", value<double>(&center_x), "X coordinate of the center (Angstrom)")
			("center_y", value<double>(&center_y), "Y coordinate of the center (Angstrom)")
			("center_z", value<double>(&center_z), "Z coordinate of the center (Angstrom)")
			("size_x", value<double>(&size_x), "size in the X dimension (Angstrom)")
			("size_y", value<double>(&size_y), "size in the Y dimension (Angstrom)")
			("size_z", value<double>(&size_z), "size in the Z dimension (Angstrom)")
			("autobox", bool_switch(&autobox), "set maps dimensions based on input ligand(s) (for --score_only and --local_only)")
		;
		//options_description outputs("Output prefixes (optional - by default, input names are stripped of .pdbqt\nare used as prefixes. _001.pdbqt, _002.pdbqt, etc. are appended to the prefixes to produce the output names");
		options_description outputs("Output (optional)");
		outputs.add_options()
			("out", value<std::string>(&out_name), "output models (PDBQT), the default is chosen based on the ligand file name")
			("dir", value<std::string>(&out_dir), "output directory for batch mode")
			("write_maps", value<std::string>(&out_maps), "output filename (directory + prefix name) for maps. Option --force_even_voxels may be needed to comply with .map format")
		;
		options_description advanced("Advanced options (see the manual)");
		advanced.add_options()
			("score_only",     bool_switch(&score_only),     "score only - search space can be omitted")
			("local_only",     bool_switch(&local_only),     "do local search only")
			("no_refine", bool_switch(&no_refine),  "when --receptor is provided, do not use explicit receptor atoms (instead of precalculated grids) for: (1) local optimization and scoring after docking, (2) --local_only jobs, and (3) --score_only jobs")
			("force_even_voxels", bool_switch(&force_even_voxels),  "calculated grid maps will have an even number of voxels (intervals) in each dimension (odd number of grid points)")
			("randomize_only", bool_switch(&randomize_only), "randomize input, attempting to avoid clashes")

			("weight_gauss1", value<double>(&weight_gauss1)->default_value(weight_gauss1),                "gauss_1 weight")
			("weight_gauss2", value<double>(&weight_gauss2)->default_value(weight_gauss2),                "gauss_2 weight")
			("weight_repulsion", value<double>(&weight_repulsion)->default_value(weight_repulsion),       "repulsion weight")
			("weight_hydrophobic", value<double>(&weight_hydrophobic)->default_value(weight_hydrophobic), "hydrophobic weight")
			("weight_hydrogen", value<double>(&weight_hydrogen)->default_value(weight_hydrogen),          "Hydrogen bond weight")
			("weight_rot", value<double>(&weight_rot)->default_value(weight_rot),                         "N_rot weight")

			("weight_vinardo_gauss1", value<double>(&weight_vinardo_gauss1)->default_value(weight_vinardo_gauss1), "Vinardo gauss_1 weight")
			("weight_vinardo_repulsion", value<double>(&weight_vinardo_repulsion)->default_value(weight_vinardo_repulsion), "Vinardo repulsion weight")
			("weight_vinardo_hydrophobic", value<double>(&weight_vinardo_hydrophobic)->default_value(weight_vinardo_hydrophobic), "Vinardo hydrophobic weight")
			("weight_vinardo_hydrogen", value<double>(&weight_vinardo_hydrogen)->default_value(weight_vinardo_hydrogen), "Vinardo Hydrogen bond weight")
			("weight_vinardo_rot", value<double>(&weight_rot)->default_value(weight_rot), "Vinardo N_rot weight")

			("weight_ad4_vdw", value<double>(&weight_ad4_vdw)->default_value(weight_ad4_vdw), "ad4_vdw weight")
			("weight_ad4_hb", value<double>(&weight_ad4_hb)->default_value(weight_ad4_hb), "ad4_hb weight")
			("weight_ad4_elec", value<double>(&weight_ad4_elec)->default_value(weight_ad4_elec), "ad4_elec weight")
			("weight_ad4_dsolv", value<double>(&weight_ad4_dsolv)->default_value(weight_ad4_dsolv), "ad4_dsolv weight")
			("weight_ad4_rot", value<double>(&weight_ad4_rot)->default_value(weight_ad4_rot), "ad4_rot weight")

			("weight_glue", value<double>(&weight_glue)->default_value(weight_glue),                      "macrocycle glue weight")
			("bias", value<std::string>(&bias_file), "bias configuration file name, content similar to BPF in AutoDock-bias")
			("multi_bias", bool_switch(&multi_bias), "add ligand bias {ligand_name}.bpf for every input ligand {ligand_name}.pdbqt in batch, content similar to BPF in AutoDock-bias")
;
		options_description misc("Misc (optional)");
		misc.add_options()
			("cpu", value<int>(&cpu)->default_value(0), "the number of CPUs to use (the default is to try to detect the number of CPUs or, failing that, use 1)")
			("seed", value<int>(&seed)->default_value(0), "explicit random seed")
			("exhaustiveness", value<int>(&exhaustiveness)->default_value(8), "exhaustiveness of the global search (roughly proportional to time): 1+")
			("max_evals", value<int>(&max_evals)->default_value(0), "number of evaluations in each MC run (if zero, which is the default, the number of MC steps is based on heuristics)")
			("num_modes", value<int>(&num_modes)->default_value(9), "maximum number of binding modes to generate")
			("min_rmsd", value<double>(&min_rmsd)->default_value(1.0), "minimum RMSD between output poses")
			("energy_range", value<double>(&energy_range)->default_value(3.0), "maximum energy difference between the best binding mode and the worst one displayed (kcal/mol)")
			("spacing", value<double>(&grid_spacing)->default_value(0.375), "grid spacing (Angstrom)")
			("verbosity", value<int>(&verbosity)->default_value(1), "verbosity (0=no output, 1=normal, 2=verbose)")
			("max_step", value<int>(&max_step)->default_value(0), "maximum number of steps in each MC run (if zero, which is the default, the number of MC steps is based on heuristics)")
			("refine_step", value<int>(&refine_step)->default_value(5), "number of steps in refinement, default=5")
			("max_gpu_memory", value<int>(&max_gpu_memory)->default_value(0), "maximum gpu memory to use (default=0, use all available GPU memory to optain maximum batch size)")
			("search_mode", value<std::string>(&search_mode), "search mode of vina (fast, balance, detail), using recommended settings of exhaustiveness and search steps; the higher the computational complexity, the higher the accuracy, but the larger the computational cost")

		;
		options_description config("Configuration file (optional)");
		config.add_options()
			("config", value<std::string>(&config_name), "the above options can be put here")
		;
		options_description info("Information (optional)");
		info.add_options()
			("help",          bool_switch(&help), "display usage summary")
			("help_advanced", bool_switch(&help_advanced), "display usage summary with advanced options")
			("version",       bool_switch(&version), "display program version")
		;
		options_description desc, desc_config, desc_simple;
		desc       .add(inputs).add(search_area).add(outputs).add(advanced).add(misc).add(config).add(info);
		desc_config.add(inputs).add(search_area).add(outputs).add(advanced).add(misc);
		desc_simple.add(inputs).add(search_area).add(outputs).add(misc).add(config).add(info);

		std::cout << version_string << '\n';
		try {
			//store(parse_command_line(argc, argv, desc, command_line_style::default_style ^ command_line_style::allow_guessing), vm);
			store(command_line_parser(argc, argv)
				.options(desc)
				.style(command_line_style::default_style ^ command_line_style::allow_guessing)
				.positional(positional)
				.run(),
				vm);
			notify(vm);
		} catch(boost::program_options::error& e) {
			std::cerr << "Command line parse error: " << e.what() << '\n' << "\nCorrect usage:\n" << desc_simple << '\n';
			return 1;
		}

		if (vm.count("config")) {
			try {
				path name = make_path(config_name);
				ifile config_stream(name);
				store(parse_config_file(config_stream, desc_config), vm);
				notify(vm);
			}
			catch(boost::program_options::error& e) {
				std::cerr << "Configuration file parse error: " << e.what() << '\n' << "\nCorrect usage:\n" << desc_simple << '\n';
				return 1;
			}
		}

		if (help) {
			std::cout << desc_simple << '\n';
			return 0;
		}

		if (help_advanced) {
			std::cout << desc << '\n';
			return 0;
		}

		if (version) {
			return 0;
		}

		if (verbosity > 0) {
			std::cout << cite_message << '\n';
		}

		if (vm.count("receptor") && vm.count("maps")) {
			std::cerr << "ERROR: Cannot specify both receptor and affinity maps at the same time, --flex argument is allowed with receptor or maps.\n";
			exit(EXIT_FAILURE);
		}

		if (vm.count("search_mode")){
			if (search_mode.compare("balance") == 0 || search_mode.compare("balanced") == 0){
				exhaustiveness = 384;
				max_step = 40;
			}
			if (search_mode.compare("fast") == 0){
				exhaustiveness = 128;
				max_step = 20;
			}
			if (search_mode.compare("detail") == 0 || search_mode.compare("detailed") == 0){
				exhaustiveness = 512;
				max_step = 40;
			}
		}

		if (sf_name.compare("vina") == 0 || sf_name.compare("vinardo") == 0) {
			if (!vm.count("receptor") && !vm.count("maps")) {
				std::cerr << desc_simple << "ERROR: The receptor or affinity maps must be specified.\n";
				exit(EXIT_FAILURE);
			}
		} else if (sf_name.compare("ad4") == 0) {
			if (vm.count("receptor")) {
				std::cerr << "ERROR: No receptor allowed, only --flex argument with the AD4 scoring function.\n";
				exit(EXIT_FAILURE);
			}
			if (!vm.count("maps")) {
				std::cerr << desc_simple << "\n\nERROR: Affinity maps are missing.\n";
				exit(EXIT_FAILURE);
			}
		} else {
			std::cerr << desc_simple << "Scoring function " << sf_name << " unknown.\n";
			exit(EXIT_FAILURE);
		}

		if (!vm.count("ligand") && !vm.count("batch") && !vm.count("gpu_batch") && !vm.count("ligand_index")) {
			std::cerr << desc_simple << "\n\nERROR: Missing ligand(s).\n";
			exit(EXIT_FAILURE);
		} else if (vm.count("ligand") && (vm.count("batch") || vm.count("gpu_batch"))) {
			std::cerr << desc_simple << "\n\nERROR: Can't use both --ligand and --batch arguments simultaneously.\n";
			exit(EXIT_FAILURE);
		} else if ((vm.count("batch") || vm.count("gpu_batch")) && !vm.count("dir")) {
			std::cerr << desc_simple << "\n\nERROR: Need to specify an output directory for batch mode.\n";
			exit(EXIT_FAILURE);
		} else if (vm.count("dir")) {
			if (!is_directory(out_dir)) {
				std::cerr << "ERROR: Directory " << out_dir << " does not exist.\n";
				exit(EXIT_FAILURE);
			}
		} else if (vm.count("ligand") && vm.count("dir")) {
			std::cerr << "WARNING: In ligand mode, --dir argument is ignored.\n";
		}

		if (!score_only) {
			if (!vm.count("out") && ligand_names.size() == 1) {
				out_name = default_output(ligand_names[0]);
				std::cout << "Output will be " << out_name << '\n';
			} else if (!vm.count("out") && ligand_names.size() >= 1) {
				std::cerr << desc_simple << "\n\nERROR: Output name must be defined when docking simultaneously multiple ligands.\n";
				exit(EXIT_FAILURE);
			}
		}

		// read ligands from index file
		// will append to `batch` if used together
		if (vm.count("ligand_index")) {
			std::ifstream index_file(ligand_index);
			if (!index_file.is_open()) {
				throw file_error(ligand_index, true);
			}
			std::string ligand_name;
			while (index_file >> ligand_name) {
				gpu_batch_ligand_names.push_back(ligand_name); // FIXME: not compatiable with CPU batch mode
			}
			index_file.close();
		}

		if (verbosity > 0) {
			std::cout << "Scoring function : " << sf_name << "\n";
			if (vm.count("receptor"))
				std::cout << "Rigid receptor: " << rigid_name << "\n";
			if (vm.count("flex"))
				std::cout << "Flex receptor: " << flex_name << "\n";
			if (ligand_names.size() == 1) {
				std::cout << "Ligand: " << ligand_names[0] << "\n";
			} else if (ligand_names.size() > 1) {
				std::cout << "Ligands:\n";
				VINA_RANGE(i, 0, ligand_names.size()) {
					std::cout << "  - " << ligand_names[i] << "\n";
				}
			} else if (batch_ligand_names.size() > 1) {
				std::cout << "Ligands (batch mode): " << batch_ligand_names.size() << " molecules\n";
			}
			if (!vm.count("maps") && !autobox) {
				std::cout << "Grid center: X " << center_x << " Y " << center_y << " Z " << center_z << "\n";
				std::cout << "Grid size  : X " << size_x << " Y " << size_y << " Z " << size_z << "\n";
				std::cout << "Grid space : " << grid_spacing << "\n";
			} else if (autobox) {
				std::cout << "Grid center: ligand center (autobox)\n";
				std::cout << "Grid size  : ligand size + " << buffer_size << " A in each dimension (autobox)\n";
				std::cout << "Grid space : " << grid_spacing << "\n";
			}
			std::cout << "Exhaustiveness: " << exhaustiveness << "\n";
			std::cout << "CPU: " << cpu << "\n";
			if (!vm.count("seed"))
				std::cout << "Seed: " << seed << "\n";
			std::cout << "Verbosity: " << verbosity << "\n";
			std::cout << "\n";
		}

		Vina v(sf_name, cpu, seed, verbosity, no_refine);

		// rigid_name variable can be ignored for AD4
		if (vm.count("receptor") || vm.count("flex"))
			v.set_receptor(rigid_name, flex_name);

		if (vm.count("bias")){
			std::ifstream bias_file_content(bias_file);
			if (!bias_file_content.is_open()){
				throw file_error(bias_file, true);
			}

			// initialize bias object
			v.set_bias(bias_file_content);

			bias_file_content.close();
			
		}

		v.multi_bias = false;
		if (multi_bias){
			if (!(vm.count("gpu_batch") || vm.count("ligand_index"))){
				std::cerr << "ERROR: Batch bias must be set in batch mode.\n";
				exit(EXIT_FAILURE);
			}
			if (vm.count("bias")){
				std::cerr << "ERROR: Batch bias is incompatible with receptor bias.\n";
				exit(EXIT_FAILURE);
			}
			v.multi_bias = true;
		}
		
		// Technically we don't have to initialize weights,
		// because they are initialized during the Vina object creation with the default weights
		// but we still do it in case the user decided to change them
		if (sf_name.compare("vina") == 0) {
			v.set_vina_weights(weight_gauss1, weight_gauss2, weight_repulsion,
							   weight_hydrophobic, weight_hydrogen, weight_glue, weight_rot);
		} else if (sf_name.compare("vinardo") == 0) {
			v.set_vinardo_weights(weight_vinardo_gauss1, weight_vinardo_repulsion,
								  weight_vinardo_hydrophobic, weight_vinardo_hydrogen, weight_glue, weight_rot);
		} else {
			v.set_ad4_weights(weight_ad4_vdw, weight_ad4_hb, weight_ad4_elec,
							  weight_ad4_dsolv, weight_glue, weight_ad4_rot);
			v.load_maps(maps);

			// It works, but why would you do this?!
			if (vm.count("write_maps"))
				v.write_maps(out_maps);
		}

		if (vm.count("ligand")) {
			v.set_ligand_from_file(ligand_names);

			if (sf_name.compare("vina") == 0 || sf_name.compare("vinardo") == 0) {
				if (vm.count("maps")) {
					v.load_maps(maps);
				} else {
					// Will compute maps only for Vina atom types in the ligand(s)
					// In the case users ask for score and local only with the autobox arg, we compute the optimal box size for it/them.
					if ((score_only || local_only) && autobox) {
						std::vector<double> dim = v.grid_dimensions_from_ligand(buffer_size);
						v.compute_vina_maps(dim[0], dim[1], dim[2], dim[3], dim[4], dim[5], grid_spacing, force_even_voxels);
					} else {
						v.compute_vina_maps(center_x, center_y, center_z, size_x, size_y, size_z, grid_spacing, force_even_voxels);
					}

					if (vm.count("write_maps"))
						v.write_maps(out_maps);
				}
			}

			if (randomize_only) {
				v.randomize();
				v.write_pose(out_name);
			} else if (score_only) {
				std::vector<double> energies;
				energies = v.score();
				v.show_score(energies);
			} else if (local_only) {
				std::vector<double> energies;
				energies = v.optimize();
				v.write_pose(out_name);
				v.show_score(energies);
			} else {
				// search one ligand on cpu
				v.global_search(exhaustiveness, num_modes, min_rmsd, max_evals);
				v.write_poses(out_name, num_modes, energy_range);
			}
		} else if (vm.count("gpu_batch") || vm.count("ligand_index")) {
			if (randomize_only || score_only || local_only ){
				printf("Not available under gpu_batch mode.\n");
				return 0;
			}
			v.enable_gpu();
			if (sf_name.compare("vina") == 0 || sf_name.compare("vinardo") == 0) {
				if (vm.count("maps")) {
					v.load_maps(maps);
				} else {
					// Will compute maps for all Vina atom types
					v.compute_vina_maps(center_x, center_y, center_z, size_x, size_y, size_z, grid_spacing, force_even_voxels);

					if (vm.count("write_maps"))
						v.write_maps(out_maps);
				}
			}
			// Vina worker[2]{v,v}; // Do CPU works on one worker while GPU works on another
			// bool index = 0; // indicate which worker occupies GPU
			std::vector<std::string> ligand_names {std::move(gpu_batch_ligand_names)};
			std::cout << "Total ligands: " << ligand_names.size() << std::endl;

			int receptor_atom_numbers = v.m_receptor.get_atoms().size();
			int deviceCount=0;
			size_t avail;
			size_t total;
			float max_memory = 32000;
			bool use_v100 = true;
			bool ad4 = false;
			if (sf_name.compare("ad4") == 0) ad4 = true;
			hipGetDeviceCount(&deviceCount);
			if (deviceCount > 0){
				hipSetDevice(0);
				hipMemGetInfo(&avail, &total);
				printf("Available Memory = %dMiB   Total Memory = %dMiB\n", int(avail/1024/1024), int(total / 1024 / 1024));
				max_memory = avail / 1024 / 1024 * 0.95; // leave 5% to prevent error
			}
			if (max_memory < 17000){
				// using T4 or other 16G global memory GPU
				use_v100 = false;
			}
			if (max_gpu_memory > 0 && max_gpu_memory < max_memory){
				max_memory = (float)max_gpu_memory;
			}

			typedef std::pair<std::string,model> named_model;
			std::vector<named_model> all_ligands;
			const int ligand_batch_limit = 1e6; // ~20GB for 100,000 lig obj
			int batch_index=0;
			while(batch_index<ligand_names.size()){
			all_ligands.clear();
			int next_batch_index = std::min(int(batch_index+ligand_batch_limit),int(ligand_names.size()));
			#pragma omp parallel for
			for (int ligand_count=batch_index;ligand_count<next_batch_index;++ligand_count)
			{
				auto& ligand=ligand_names[ligand_count];
				auto l = parse_ligand_pdbqt_from_file_no_failure(
						ligand, v.m_scoring_function->get_atom_typing());
				#pragma omp critical
				all_ligands.emplace_back(std::make_pair(ligand,l));
			}
			batch_index=next_batch_index;
			/*
			std::sort(all_ligands.begin(), all_ligands.end(),
				  [](named_model a, named_model b)
				  { return a.second.get_atoms().size() < b.second.get_atoms().size(); });
			*/
			DEBUG_PRINTF("%d\n",next_batch_index);
			int processed_ligands = 0;
			int batch_id = 0;
			while (processed_ligands < all_ligands.size()) {
				++batch_id;
				auto start = std::chrono::system_clock::now();
				Vina v1(v); // reuse init'ed maps
				int batch_size = 0;
				int all_atom2_numbers = 0; // total number of atom^2 in current batch
				std::vector<model> batch_ligands; // ligands in current batch
				v1.bias_batch_list.clear();
				while (predict_peak_memory(batch_size, exhaustiveness, all_atom2_numbers, use_v100, v.multi_bias) < max_memory &&
					 processed_ligands + batch_size < all_ligands.size())
				{
					batch_ligands.emplace_back(all_ligands[processed_ligands + batch_size].second);
					int next_atom_numbers = batch_ligands.back()
												.get_atoms()
												.size()
											+ receptor_atom_numbers;
					int next_atom2_numbers = next_atom_numbers * next_atom_numbers; // Memory ~ atom numbers^2
					all_atom2_numbers += next_atom2_numbers;
					batch_size++;
				}
				DEBUG_PRINTF("batch size=%d, all_atom2_numbers=%d\n", batch_size, all_atom2_numbers);

				std::cout << "Batch " << batch_id << " size: " << batch_size << std::endl;
				std::vector<std::string> batch_ligand_names;
				for (int i = processed_ligands; i < processed_ligands + batch_size; i++)
				{
					batch_ligand_names.push_back(all_ligands[i].first);
				}
				processed_ligands += batch_size;
				gpu_out_name = {};
				VINA_RANGE(i, 0, batch_ligand_names.size())
				{
					gpu_out_name.push_back(default_output(get_filename(batch_ligand_names[i]), out_dir));
					if (v1.multi_bias){
						std::ifstream bias_file_content(get_biasname(batch_ligand_names[i]));
						if (!bias_file_content.is_open()){
							throw file_error(bias_file, true);
						}

						// initialize bias object
						v1.set_batch_bias(bias_file_content);
						bias_file_content.close();
					}
				}
				v1.set_ligand_from_object_gpu(batch_ligands);
				v1.global_search_gpu(exhaustiveness, num_modes, min_rmsd, max_evals, max_step, batch_ligand_names.size(), (unsigned long long)seed, refine_step);
				v1.write_poses_gpu(gpu_out_name, num_modes, energy_range);
				auto end = std::chrono::system_clock::now();
				std::cout << "Batch " << batch_id << " running time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
			}
		}
		}
	}

	catch(file_error& e) {
		std::cerr << "\n\nError: could not open \"" << e.name.string() << "\" for " << (e.in ? "reading" : "writing") << ".\n";
		return 1;
	}
	catch(boost::filesystem::filesystem_error& e) {
		std::cerr << "\n\nFile system error: " << e.what() << '\n';
		return 1;
	}
	catch(usage_error& e) {
		std::cerr << "\n\nUsage error: " << e.what() << ".\n";
		return 1;
	}
#ifdef NDEBUG // don't catch in debug mode
	catch (std::bad_alloc &)
	{
		std::cerr << "\n\nError: insufficient memory!\n";
		return 1;
	}

	// Errors that shouldn't happen:
	catch(std::exception& e) {
		std::cerr << "\n\nAn error occurred: " << e.what() << ". " << error_message;
		return 1;
	}
	catch(internal_error& e) {
		std::cerr << "\n\nAn internal error occurred in " << e.file << "(" << e.line << "). " << error_message;
		return 1;
	}
	catch(...) {
		std::cerr << "\n\nAn unknown error occurred. " << error_message;
		return 1;
	}
#endif // NDEBUG
}
