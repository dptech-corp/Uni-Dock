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

// Test program for vina cuda worker with CUDA streams
// Runs n 1:1 ligand operations in parallel, and compare with running in sequence

#include <thread>
#include <chrono>
#include <boost/filesystem.hpp>
#include <sstream>
#include "simulation_container.h"

// Perform 1:1 docking with CUDA streams, but does it one at a time.
// This code follows the same sequence of operations as original code.
// The original non-CUDA-Stream unidock code can be run by using global_search_gpu()
// instead of the prime/run/obtain sequence in the code below

std::string util_random_string(std::size_t length)
{
    const std::string CHARACTERS = "iamafunnydogthatlaughsindeterministically";

    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_int_distribution<> distribution(0, CHARACTERS.size() - 1);

    std::string rstring;

    for (std::size_t i = 0; i < length; ++i)
    {
        rstring += CHARACTERS[distribution(generator)];
    }

    return rstring;
}

// Dock a single complex at time
// Can use either streaming or non-streaming modes

bool dock_one(
            bool use_mc_non_streaming,
            complex_property prop, 
            bool local_only,
            std::string workdir,
            std::string input_dir,
            std::string out_phrase,
            int batch_size, 
            int max_eval_steps)            
{
    int exhaustiveness = 512;
    int num_modes = 1;
    int min_rmsd = 0;
    int max_evals = 0;
    int max_step = 40;   
    int seed = 5;
    int refine_step = 5;
    double energy_range = 3.0;
    bool keep_H = true;
    std::string sf_name = "vina";
    int cpu = 0;
    int verbosity = 1;
    bool no_refine = false;
    int box_size = 25;
    int size_x = box_size;
    int size_y = box_size;
    int size_z = box_size;
    double grid_spacing = 0.375;
    bool force_even_voxels = false;
    // vina weights
    double weight_gauss1 = -0.035579;
    double weight_gauss2 = -0.005156;
    double weight_repulsion = 0.840245;
    double weight_hydrophobic = -0.035069;
    double weight_hydrogen = -0.587439;
    double weight_rot = 0.05846;    
    // macrocycle closure
    double weight_glue = 50.000000;  // linear attraction
    monte_carlo mc;

    double center_x = prop.center_x;
    double center_y = prop.center_y;
    double center_z = prop.center_z;
    std::string complex_name = prop.complex_name;

    std::string ligand_name(workdir + "/" + input_dir + "/" + complex_name + "_ligand.pdbqt");
    if (! boost::filesystem::exists( ligand_name ) )
    {
        std::cout << "Input ligand file does not exist\n";        
        return false;
    }

    std::string out_dir(workdir + "/" + out_phrase);



    std::vector<std::string> gpu_out_name;
    gpu_out_name.emplace_back(default_output(get_filename(ligand_name), out_dir));
    if (!boost::filesystem::exists(out_dir))
    {
        std::cout << "Creating output dir" << out_dir << "\n";
        boost::filesystem::create_directory(out_dir);
    }


    if (use_mc_non_streaming) //original code that runs a non-CUDA Stream kernel
    {
        // Create the vina object
        Vina v(sf_name, cpu, seed, verbosity, no_refine);

        v.bias_batch_list.clear();
        v.multi_bias = false;
        v.set_vina_weights(weight_gauss1, weight_gauss2, weight_repulsion, weight_hydrophobic,
                            weight_hydrogen, weight_glue, weight_rot);

        // rigid_name variable can be ignored for AD4
        std::string flex;
        std::string rigid(workdir + "/" + input_dir + "/" + complex_name + "_protein.pdbqt");
        if (! boost::filesystem::exists( rigid ) )
        {
            std::cout << "Input (rigid) protein file does not exist\n";        
            return false;
        }    
        v.set_receptor(rigid, flex);

        std::vector<model> batch_ligands;  // ligands in current batch    
        auto parsed_ligand = parse_ligand_from_file_no_failure(
            ligand_name, v.m_scoring_function.get_atom_typing(), keep_H);
        batch_ligands.emplace_back(parsed_ligand);

        v.set_ligand_from_object_gpu(batch_ligands);
        v.enable_gpu(); // Has to be done before computing vina maps
        v.compute_vina_maps(center_x, center_y, center_z, size_x, size_y, size_z,
                                                grid_spacing, force_even_voxels);
                                                        
        v.global_search_gpu(exhaustiveness, num_modes, min_rmsd, max_evals, max_step,
                                1, (unsigned long long)seed,
                                refine_step, local_only);
        v.write_poses_gpu(gpu_out_name, num_modes, energy_range);                                      
    }
    else // Run with CUDA streams
    {
        vina_cuda_worker v(center_x,
            center_y, 
            center_z, 
            complex_name,
            local_only,
            box_size,
            max_step,
            workdir,
            input_dir,
            out_phrase);

        v.bias_batch_list.clear();
        v.multi_bias = false;
        v.set_vina_weights(weight_gauss1, weight_gauss2, weight_repulsion, weight_hydrophobic,
                            weight_hydrogen, weight_glue, weight_rot);

        // rigid_name variable can be ignored for AD4
        std::string flex;
        std::string rigid(workdir + "/" + input_dir + "/" + complex_name + "_protein.pdbqt");
        if (! boost::filesystem::exists( rigid ) )
        {
            std::cout << "Input (rigid) protein file does not exist\n";        
            return false;
        }    
        v.set_receptor(rigid, flex);

        std::vector<model> batch_ligands;  // ligands in current batch    
        auto parsed_ligand = parse_ligand_from_file_no_failure(
            ligand_name, v.m_scoring_function.get_atom_typing(), keep_H);
        batch_ligands.emplace_back(parsed_ligand);

        v.set_ligand_from_object_gpu(batch_ligands);
        v.enable_gpu(); // Has to be done before computing vina maps
        v.compute_vina_maps(center_x, center_y, center_z, size_x, size_y, size_z,
                                                grid_spacing, force_even_voxels);
                                                        
        mc = v.global_search_gpu_prime(
                                exhaustiveness, num_modes, min_rmsd, max_evals, max_step,
                                1, (unsigned long long)seed,
                                local_only);   
        v.global_search_gpu_run(mc);                            
        v.global_search_gpu_obtain(mc, 1, refine_step);
        v.write_poses_gpu(gpu_out_name, num_modes, energy_range);   

    }
    

    return true;                       
}

// Exercises the CUDA-Stream accelerated build 
// Runs batched (5 sets together) and non-batched (5 individual), compares timing
int dock_many_non_batched(
                std::string work_dir, 
                std::string input_dir, 
                std::string out_dir,
                int batch_size, 
                bool local_only, 
                int max_eval_steps)
{
    std::string out_phrase = std::to_string(batch_size) + "_" + std::to_string (int(local_only)) +
            "_" + std::to_string (max_eval_steps) + "_" + util_random_string(5);
    std::string non_batched_out_dir = "out_non_batched_" + out_phrase;

    complex_property cp1(-0.487667, 24.0228,-11.1546, "7TUO_KL9");
    complex_property cp2(92.7454, 8.79115, 30.7175, "6VTA_AKN");
    complex_property cp3(-22.1801, 13.4045, 27.4542, "5S8I_2LY");
    complex_property cp4(54.9792, -21.0535, -10.7179, "6VS3_R6V");
    complex_property cp5(-15.0006, -23.6868, 149.842, "7VJT_7IJ");

    auto start_one_by_one = std::chrono::steady_clock::now();
    //-9.7
    dock_one(true, cp1, local_only, work_dir, input_dir, non_batched_out_dir, batch_size, max_eval_steps);
    // -7.1
    dock_one(true, cp2, local_only, work_dir, input_dir, non_batched_out_dir, batch_size, max_eval_steps);
    // -5.8
    dock_one(true, cp3, local_only, work_dir, input_dir, non_batched_out_dir, batch_size, max_eval_steps);
    // -9.3149
    dock_one(true, cp4, local_only, work_dir, input_dir, non_batched_out_dir, batch_size, max_eval_steps);
    // -12.5
    dock_one(true, cp5, local_only, work_dir, input_dir, non_batched_out_dir, batch_size, max_eval_steps);

    auto end_one_by_one = std::chrono::steady_clock::now();
    auto milliseconds_one_by_one = std::chrono::duration_cast<std::chrono::milliseconds>(end_one_by_one - start_one_by_one).count();
    
    std::cout << "One by one - Time elapsed milliSeconds = " << milliseconds_one_by_one << "\n";

    return 0;
}

void fast_usage()
{
    std::string usage = "\
\n\
Usage:\n\
app <work_dir> <input_dir_relative> <batch_size> <max_limit> <local_only> <max_eval_steps>\n\
// work_dir           = Full path to a working directory\n\
//   The outdir will be created with random name in the workdir\n\
// input_dir_relative = Path (relative to workdir) that contains the pdbqt files for ligand and protein\n\
//   The below files are required for each complex\n\
//      <complex_name>_ligand.pdbqt\n\
//      <complex_name>_protein.pdbqt\n\
//      <complex_name>_ligand_config.txt (containing the center_x, center_y, center_z)\n\
// batch_size     = Size of each batch\n\
// max_limit      = Limits number of complexs to be analysed\n\
// local_only     = 1, for localonly using computed map for given atoms, or 0, for randomized search\n\
// max_eval_steps = Number of steps in each MC evaluation\n\
\
\n";

    std::cout << usage;
}

void parse_args(char* argv[],
                std::string & work_dir, 
                std::string & input_dir, 
                std::string & out_dir,
                int & batch_size, 
                int & max_limit, 
                bool & local_only, 
                int & max_eval_steps)
{
    
    work_dir = argv[1];
    input_dir = argv[2];
    batch_size = std::stoi(argv[3]);
    max_limit = std::stoi(argv[4]);

    int local_ = std::stoi(argv[5]);
    local_only = !!local_;
    max_eval_steps = std::stoi(argv[6]);

    std::string out_phrase = std::to_string(batch_size) + "_" + std::to_string (local_) + 
            "_" + std::to_string (max_eval_steps) + "_" + util_random_string(5);
    out_dir = "out_" + out_phrase;
}


// Exercises the CUDA-Stream accelerated build
// Runs batched operations using cmdline arguments
//
// Arguments:
//
// workdir   = Full path to a working directory where input and output folders can reside
// input_dir = Path (relative to workdir) that contains the pdbqt files for ligand and protein
//   The below files are required for each complex
//      <complex_name>_ligand.pdbqt
//      <complex_name>_protein.pdbqt
//      <complex_name>_ligand_config.txt (containing the center_x, center_y, center_z)
// batch_size     = Size of each batch
// max_limit      = Limits number of complexs to be analysed
// local_only     = 1, for localonly using computed map for given atoms, or 0, for randomized search
// max_eval_steps = Number of steps in each MC evaluation

int main(int argc, char* argv[])
{

    if (argc < 7)
    {
        fast_usage();
        exit(-1);
    }

    std::string input_path;
    std::string work_dir;
    std::string out_phrase;

    // These are sample values
    int batch_size = 5;
    int max_limit = 1000; //max number of ligand:protein complexes to be run (ex control max time to run)
    bool local_only = false;
    int max_eval_steps = 60;

    parse_args(argv, work_dir, input_path, out_phrase, batch_size, max_limit, local_only, max_eval_steps);

    simulation_container sc(work_dir, input_path, out_phrase, batch_size, 25, local_only, max_eval_steps, max_limit);

    if (int res = sc.prime())
    {
        std::cout << "Error priming [" << res << "]\n";
        return res;
    }

    auto start = std::chrono::steady_clock::now();

    sc.launch();

    auto end = std::chrono::steady_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Completed Batched Operations in " << milliseconds << " mS\n";

    // For comparison - use original non-streamed code
    dock_many_non_batched (work_dir, input_path, out_phrase, batch_size, local_only, max_eval_steps);

    return 0;
}
