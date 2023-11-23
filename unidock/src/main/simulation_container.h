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
#include "vina.h"
#include "utils.h"
#include "scoring_function.h"

#include <thread>
#include <chrono>
#include <iterator>
#include <cstddef>

#include "vina_cuda_worker.h"


// Information about current simulation
struct simulation_container
{
    std::string m_work_dir;
    std::string m_input_path;
    std::string m_out_phrase;
    int m_batch_size;
    int m_box_size;
    bool m_local_only;
    int m_max_limits;
    int m_max_global_steps;
    int m_verbosity;
    bool m_isGPU;

    std::vector<std::string> m_complex_names;
    std::vector<boost::filesystem::directory_entry> m_ligand_paths;
    std::vector<boost::filesystem::directory_entry> m_ligand_config_paths;
    std::vector<boost::filesystem::directory_entry> m_protein_paths;
    complex_property_holder * m_ptr_complex_property_holder;

    simulation_container(std::string work_dir, std::string input_path, std::string out_phrase, 
                int batch_size, int box_size, bool local_only, int max_eval_steps, int max_limits, int verbosity, bool isGPU):
        m_work_dir(work_dir),
        m_input_path(input_path),
        m_out_phrase(out_phrase),
        m_batch_size(batch_size),
        m_box_size(box_size),
        m_local_only(local_only),
        m_max_global_steps(max_eval_steps),
        m_max_limits(max_limits),
        m_verbosity(verbosity),
        m_isGPU(isGPU)
    {}

    std::string trim_eol(std::string line)
    {
        std::string newString;

        for (char ch : line)
        {
            if (ch == '\n' || ch == '\r')
                continue;
            newString += ch;
        }
        return newString;
    }

    void fill_config(complex_property & cp, std::string path, std::string protein_name, std::string ligand_name)
    {
        std::ifstream ifs(path);
        std::string line;
        double vals[3];
        int id = 0;
        while (std::getline(ifs, line))
        {       
            std::string trimmed(trim_eol(line));
            int pos = trimmed.find('=');
            vals[id] = std::stod(trimmed.substr(pos+1,  std::string::npos));       
            id ++;
        }

        cp.center_x = vals[0];
        cp.center_y = vals[1];
        cp.center_z = vals[2];

        // Default to provided, update if in config file
        cp.box_x = cp.box_y = cp.box_z = m_box_size;
        if (id > 3)
        {
            cp.box_x = vals[3];
            cp.box_y = vals[4];
            cp.box_z = vals[5];
        }

        cp.protein_name = protein_name;
        cp.ligand_name = ligand_name;

        ifs.close();

    }

    void add_rank_combinations(std::string effective_path)
    {
        int curr_entry_size = 0;
        //search for complex_rank<n>.pdbqt for ranked ligands
        for (boost::filesystem::directory_entry& entry : boost::filesystem::recursive_directory_iterator(effective_path))
        {
            int pos_rank = entry.path().string().find("_rank");
            int pos_config = entry.path().stem().string().find("_config");
            int pos_pdbqt = entry.path().extension().string().find(".pdbqt");

            if (pos_rank != std::string::npos &&
                pos_pdbqt != std::string::npos &&
                pos_config == std::string::npos)
            {
                int pos_complex = entry.path().stem().string().find("_rank");
                std::string complex = entry.path().stem().string().substr(0, pos_complex);
                m_complex_names.emplace_back(complex);
                m_ligand_paths.emplace_back(entry.path());
                m_protein_paths.emplace_back(entry.path().parent_path() / boost::filesystem::path(complex + "_protein.pdbqt"));
                m_ligand_config_paths.emplace_back(entry.path().parent_path() / boost::filesystem::path(entry.path().stem().string() + "_config.txt"));

                curr_entry_size ++;
                if (curr_entry_size >= m_max_limits)
                {
                    std::cout << "Limiting number of ranked samples to max limits " << m_max_limits << "\n";
                    break;
                }
            }
        }
    }
    void add_combinations(std::string effective_path)
    {
        int curr_entry_size = 0;
        for (boost::filesystem::directory_entry& entry : boost::filesystem::recursive_directory_iterator(effective_path))
        {
            int pos = entry.path().string().find("_protein.pdbqt");

            if (pos != std::string::npos)
            {
                int pos_complex = entry.path().stem().string().find("_protein");
                std::string complex = entry.path().stem().string().substr(0, pos_complex);
                
                m_complex_names.emplace_back(complex);
                m_ligand_paths.emplace_back(entry.path());
                m_protein_paths.emplace_back(entry.path().parent_path() / boost::filesystem::path(complex + "_protein.pdbqt"));
                m_ligand_config_paths.emplace_back(entry.path().parent_path() / boost::filesystem::path(complex + "_ligand_config.txt"));

                curr_entry_size ++;
                if (curr_entry_size >= m_max_limits)
                {
                    std::cout << "Limiting number of samples to max limits " << m_max_limits << "\n";
                    break;
                }
            }
        }
    }

    int prime()
    {
        std::string effective_path = m_work_dir + '/' + m_input_path;

        if (!boost::filesystem::exists(effective_path))
        {
            std::cout << "Error: Input path " << effective_path << " does not exist\n";
            return -1;
        }

        add_rank_combinations(effective_path);
        //add_combinations(effective_path);

        std::cout << "Found " << m_complex_names.size() << "\n";

        m_ptr_complex_property_holder = new complex_property_holder(m_complex_names.size());

        int id = 0;
        for (complex_property & cp: *m_ptr_complex_property_holder)
        {
            fill_config(cp, m_ligand_config_paths[id].path().string(), m_protein_paths[id].path().string(), m_ligand_paths[id].path().string());
            id ++;
        }

        return 0;
    }   
    // Launch simulations 
    int launch()
    {
        int batches = m_complex_names.size()/m_batch_size;
        std::cout << "To do [" << batches << "] batches, box = " << m_box_size << " max_eval_steps global = " << m_max_global_steps << "\n";
        std::cout << "Batched output to " << m_out_phrase << "\n";

        std::vector<complex_property> cp;
        for (int i = 0;i < batches;i ++)
        {            
            for (int curr = 0;curr < m_batch_size;curr ++)
            {
                int index = i*m_batch_size + curr;
                cp.emplace_back(m_ptr_complex_property_holder->m_properties[index]);
                std::cout << "Processing " << m_ptr_complex_property_holder->m_properties[index].ligand_name << "\n";
            }
            // run
            batch_dock_with_worker(cp, m_local_only, m_work_dir, m_input_path, m_out_phrase);  
            std::cout << "Batch [" << i << "] completed.\n"; 
            cp.clear();
        }
        // Remaining if any
        int remaining = m_complex_names.size() - batches * m_batch_size;
        if (remaining > 0)
        {
            for (int i = 0;i < remaining;i ++)
            {
                int index = i + batches * m_batch_size;
                cp.emplace_back(m_ptr_complex_property_holder->m_properties[index]);
            }
            batch_dock_with_worker(cp, m_local_only, m_work_dir, m_input_path, m_out_phrase);
            cp.clear();
        }
        std::cout << "Remaining [" << remaining << "] completed.\n";

        return 0;
    };

// Launches a batch of vcw workers in separate threads
// to perform 1:1 docking.
// Each launch uses CUDA stream for concurrent operation of the batches

    void batch_dock_with_worker(
                std::vector<complex_property> props, 
                bool local_only,
                std::string workdir,
                std::string input_dir,
                std::string out_phrase)
    {
        std::vector<std::thread> worker_threads;

        for (int i = 0;i < props.size();i ++)
        {
            worker_threads.emplace_back(std::thread(
                [=]()
                {
                    vina_cuda_worker vcw(props[i].center_x, props[i].center_y, 
                            props[i].center_z, props[i].protein_name,props[i].ligand_name,
                            local_only, m_box_size, m_max_global_steps, m_verbosity,
                            workdir, input_dir, out_phrase);
                    try
                    {
                        vcw.launch();
                        vcw.wait_for_completion();
                    }
                    catch(...)
                    {
                        std::cerr << "Exception processing " << props[i].ligand_name;
                    }
                }
            )
            );
        }
        for (int i = 0;i < props.size();i ++)
        {
            worker_threads[i].join();
        }
    }    

    bool dock_one_cpu_local(complex_property prop)
    {
        int exhaustiveness = 512;
        int num_modes = 1;
        int min_rmsd = 0;
        int max_evals = 0;
        int seed = 5;
        int refine_step = 5;
        double energy_range = 3.0;
        bool keep_H = true;
        std::string sf_name = "vina";
        int cpu = 0;
        int verbosity = 1;
        bool no_refine = false;
        int size_x = m_box_size;
        int size_y = m_box_size;
        int size_z = m_box_size;
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

        double center_x = prop.center_x;
        double center_y = prop.center_y;
        double center_z = prop.center_z;

        std::string ligand_name(m_work_dir + "/" + m_input_path + "/" + prop.ligand_name + ".pdbqt");
        if (! boost::filesystem::exists( ligand_name ) )
        {
            std::cout << "Input ligand file does not exist\n";        
            return false;
        }

        std::string out_dir(m_work_dir + "/" + m_out_phrase);

        std::vector<std::string> gpu_out_name;
        gpu_out_name.emplace_back(default_output(get_filename(ligand_name), out_dir));
        if (!boost::filesystem::exists(out_dir))
        {
            std::cout << "Creating output dir" << out_dir << "\n";
            boost::filesystem::create_directory(out_dir);
        }

        // Create the vina object
        Vina v(sf_name, cpu, seed, verbosity, no_refine);

        v.bias_batch_list.clear();
        v.multi_bias = false;
        v.set_vina_weights(weight_gauss1, weight_gauss2, weight_repulsion, weight_hydrophobic,
                            weight_hydrogen, weight_glue, weight_rot);

        std::string flex;
        std::string rigid(m_work_dir + "/" + m_input_path + "/" + prop.protein_name + ".pdbqt");
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

        v.set_ligand_from_object(batch_ligands);

#if 0 //autobox
        double buffer_size = 4;
        std::vector<double> dim = v.grid_dimensions_from_ligand(buffer_size);
        v.compute_vina_maps(dim[0], dim[1], dim[2], dim[3], dim[4], dim[5],
                            grid_spacing, force_even_voxels);
#else
        v.compute_vina_maps(center_x, center_y, center_z, size_x, size_y, size_z,
                            grid_spacing, force_even_voxels);
#endif
        std::vector<double> energies;
        energies = v.optimize();
        v.write_pose(default_output(get_filename(ligand_name), out_dir));
        v.show_score(energies);

        return true;                       
    }

    void launch_cpu()
    {
        std::cout << "WARN: Launching CPU docking\n";

        for (int i = 0;i < m_complex_names.size();i ++)
        {
            dock_one_cpu_local(m_ptr_complex_property_holder->m_properties[i]);
        }
    }


}; // simulation_container

