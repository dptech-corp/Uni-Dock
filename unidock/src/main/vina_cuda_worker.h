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
#pragma once

#include <iostream>
#include <string>
#include <vector>  // ligand paths
#include <exception>
#include <boost/program_options.hpp>

//#define DEBUG

#include "vina.h"
#include "utils.h"
#include "scoring_function.h"

#include <thread>
#include <chrono>
#include <iterator>
#include <cstddef>

#include "complex_property.h"
// Use vina sf, and accelerate operations with CUDA streams

class vina_cuda_worker : public Vina
{
    int exhaustiveness = 512;
    int num_modes = 1;
    int min_rmsd = 0;
    int max_evals = 0;
    int max_step = 60;   
    int seed = 5;
    int refine_step = 3;
    bool local_only = false;
    double energy_range = 3.0;
    bool keep_H = true;
    std::string sf_name = "vina";
    int cpu = 0;
    bool no_refine = false;
    double size_x = 25;
    double size_y = 25;
    double size_z = 25;
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
    std::vector<std::string> gpu_out_name;
    std::string workdir;
    std::string input_dir;
    std::string out_dir;
    std::vector<model> batch_ligands; 
    double center_x;
    double center_y; 
    double center_z; 
    std::string protein_name;
    std::string ligand_name;
    void init(std::string out_phrase)
    {
        out_dir = workdir + "/" + out_phrase;
        if (!boost::filesystem::exists(out_dir))
        {
            boost::filesystem::create_directory(out_dir);
        }
	    m_seed = seed;
    }
public:            
    vina_cuda_worker(
            int seed,
            int num_modes,
            int refine_steps,
            double center_x, 
            double center_y, 
            double center_z, 
            std::string protein_name,
            std::string ligand_name,
            bool local_only,
            std::vector<double> box_size_xyz,
            int max_step,
            int verbosity,
            int exh,
            std::string workdir,
            std::string input_dir,
            std::string out_phrase):

            seed(seed),
            num_modes(num_modes),
            refine_step(refine_steps),
            workdir(workdir),
            input_dir(input_dir),
            center_x(center_x),
            center_y(center_y),
            center_z(center_z),
            size_x(box_size_xyz[0]),
            size_y(box_size_xyz[1]),
            size_z(box_size_xyz[2]),
            max_step(max_step),
            exhaustiveness(exh),
            protein_name(protein_name),
            ligand_name(ligand_name),
            local_only(local_only),
            out_dir(out_phrase),
            Vina{"vina", 0, seed, verbosity, false, NULL}
    {
        init(out_phrase);
    }

    ~vina_cuda_worker()
    {

    }


    // Performs CUDA Stream based docking of 1 ligand and 1 protein

    int launch()
    {
        multi_bias = false;
	    bias_batch_list.clear();
        
        set_vina_weights(weight_gauss1, weight_gauss2, weight_repulsion, weight_hydrophobic,
                                weight_hydrogen, weight_glue, weight_rot);        
        std::string flex;
        std::string rigid(protein_name);

        if (! boost::filesystem::exists( ligand_name ) )
        {
            std::cout << "Input ligand file does not exist ("  << ligand_name << ")\n";
            return -1;
        }        
        if (! boost::filesystem::exists( rigid ) )
        {
            std::cout << "Input (rigid) protein file does not exist (" << rigid << ")\n";
            return -1;
        }    
 
        set_receptor(rigid, flex);                                

        enable_gpu();
        compute_vina_maps(center_x, center_y, center_z, size_x, size_y, size_z,
                                                grid_spacing, force_even_voxels);

        auto parsed_ligand = parse_ligand_from_file_no_failure(
            ligand_name, m_scoring_function->get_atom_typing(), keep_H);
        batch_ligands.emplace_back(parsed_ligand);

        set_ligand_from_object_gpu(batch_ligands);

        bool create_new_stream = true;
        global_search_gpu(  exhaustiveness, num_modes, min_rmsd, max_evals, max_step,
                            1, (unsigned long long)seed, refine_step,
                            local_only, create_new_stream);

        std::vector<std::string> gpu_out_name;
        gpu_out_name.push_back(
            default_output(get_filename(ligand_name), out_dir));
        write_poses_gpu(gpu_out_name, num_modes, energy_range);

        return 0;
    }

    // Protectors
    vina_cuda_worker (const vina_cuda_worker&);
    vina_cuda_worker& operator=(const vina_cuda_worker&);
};

