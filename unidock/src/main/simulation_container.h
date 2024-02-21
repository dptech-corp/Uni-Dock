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
#include <atomic>

#include "vina_cuda_worker.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


// Information about current simulation
struct simulation_container
{
    std::string m_work_dir;
    std::string m_input_path;
    std::string m_out_phrase;
    int m_batch_size;
    std::vector<double> m_box_size;
    bool m_local_only;
    int m_max_limits = 5000;
    int m_max_global_steps;
    int m_verbosity;
    int m_exhaustiveness = 512;
    bool m_isGPU;
    int m_seed = 5;
    int m_num_modes = 9;
    int m_refine_steps = 3;
    int m_device_id = 0;

    std::vector<std::string> m_complex_names;
    std::string m_config_json_path;
    std::vector<boost::filesystem::directory_entry> m_ligand_paths;
    std::vector<boost::filesystem::directory_entry> m_ligand_config_paths;
    std::vector<boost::filesystem::directory_entry> m_protein_paths;
    complex_property_holder * m_ptr_complex_property_holder;
    int m_successful_property_count;

    simulation_container(
        int seed,
        int num_modes,
        int refine_steps,
        std::string out_dir,
        std::string config_json_path,
        int paired_batch_size,
        std::vector<double> box_size_xyz,
        int local_only,
        int max_step,
        int verbosity,
        int exh,
        int device_id):

        m_seed(seed),
        m_num_modes(num_modes),
        m_refine_steps(refine_steps),
        m_work_dir(out_dir),
        m_config_json_path(config_json_path),
        m_batch_size(paired_batch_size),
        m_box_size(box_size_xyz),
        m_local_only(local_only),
        m_max_global_steps(max_step),
        m_verbosity(verbosity),
        m_exhaustiveness(exh),
        m_isGPU(true),
        m_successful_property_count (0),
        m_device_id(device_id)
     {
        //m_out_phrase = util_random_string(5);
     }
     ~simulation_container()
     {
        if (m_ptr_complex_property_holder)
        {
            delete m_ptr_complex_property_holder;
            m_ptr_complex_property_holder = 0;
        }
     };


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

    void add_rank_combinations_from_json(std::string filename)
    {
        int curr_entry_size = 0;
        boost::property_tree::ptree tree_root;
        boost::property_tree::read_json(filename, tree_root);

        using boost::property_tree::ptree;
        ptree::const_iterator end = tree_root.end();

        for (ptree::const_iterator it = tree_root.begin(); it != end; ++it) {

            m_complex_names.emplace_back(it->first);

            for (ptree::const_iterator it_entries = it->second.begin(); it_entries != it->second.end(); ++it_entries)
            {
                if (it_entries->first == "ligand")
                {
                    m_ligand_paths.emplace_back(it_entries->second.get_value<std::string>());
                }
                if (it_entries->first == "protein")
                {
                    m_protein_paths.emplace_back(it_entries->second.get_value<std::string>());
                }
                if (it_entries->first == "ligand_config")
                {
                    m_ligand_config_paths.emplace_back(it_entries->second.get_value<std::string>());
                }
            }
            curr_entry_size ++;
            if (curr_entry_size >= m_max_limits)
            {
                std::cout << "Limiting number of ranked samples to max limits " << m_max_limits << "\n";
                break;
            }
        }
    }

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

    int fill_config_from_json(complex_property & cp, std::string path, std::string protein_name, std::string ligand_name)
    {
        boost::property_tree::ptree tree_root;
        boost::property_tree::read_json(path, tree_root);

        // Default to provided box, update if in config file
        cp.box_x = m_box_size[0];
        cp.box_y = m_box_size[1];
        cp.box_z = m_box_size[2];

        try
        {
            cp.center_x = tree_root.get<float>("center_x");
            cp.center_y = tree_root.get<float>("center_y");
            cp.center_z = tree_root.get<float>("center_z");
            cp.box_x = tree_root.get<float>("size_x");
            cp.box_y = tree_root.get<float>("size_y");
            cp.box_z = tree_root.get<float>("size_z");
        }
        catch(...)
        {
            std::cout << "Error parsing config json " << path << "\n";
            return -1;
        }

        cp.protein_name = protein_name;
        cp.ligand_name = ligand_name;

        return 0;
    }

    int_least32_t fill_config(complex_property & cp, std::string path, std::string protein_name, std::string ligand_name)
    {
        // Default to provided box, update if in config file
        cp.box_x = m_box_size[0];
        cp.box_y = m_box_size[1];
        cp.box_z = m_box_size[2];

        if (path.empty())
        {
            return -1;
        }
        else
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

            if (id > 3)
            {
                cp.box_x = vals[3];
                cp.box_y = vals[4];
                cp.box_z = vals[5];
            }
            ifs.close();
        }

        cp.protein_name = protein_name;
        cp.ligand_name = ligand_name;

        return 0;
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
                m_protein_paths.emplace_back(entry.path());
                m_ligand_paths.emplace_back(entry.path().parent_path() / boost::filesystem::path(complex + "_ligand.pdbqt"));
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

    void add_combinations(std::vector<std::string> ligand_names)
    {
        int curr_entry_size = 0;
        for (std::string& path : ligand_names)
        {
            int pos = path.find("_ligand.pdbqt");

            if (pos != std::string::npos)
            {
                int pos_complex = path.find("_ligand");
                std::string complex = path.substr(0, pos_complex);

                m_complex_names.emplace_back(complex);
                m_protein_paths.emplace_back(complex + "_protein.pdbqt");
                m_ligand_paths.emplace_back(path);
                m_ligand_config_paths.emplace_back(complex + "_ligand_config.txt");

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
        if (m_config_json_path.empty())
        {
            std::cout << "Found nothing to prime.\n";
            return -1;
        }
        else
        {
            try
            {
                add_rank_combinations_from_json(m_config_json_path);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                std::cout << e.what() << '\n';
                return -1;
            }
        }

        std::cout << "Found " << m_complex_names.size() << " to be primed.\n";

        m_ptr_complex_property_holder = new complex_property_holder(m_complex_names.size());

        for (int id = 0;id < m_complex_names.size();id ++)
        {
            int success_filled = -1;

            complex_property& cp = m_ptr_complex_property_holder->m_properties[m_successful_property_count];

            if (boost::filesystem::extension(m_ligand_config_paths[id].path().string()) == ".json")
            {
                try
                {
                    success_filled = fill_config_from_json(cp, m_ligand_config_paths[id].path().string(), m_protein_paths[id].path().string(), m_ligand_paths[id].path().string());
                }
                catch(const std::exception& e)
                {
                    std::cout << "Error reading config json " << e.what() << "\n";                   
                    success_filled = -1;                    
                }
            }
            else
            {
                success_filled = fill_config(cp, m_ligand_config_paths[id].path().string(), m_protein_paths[id].path().string(), m_ligand_paths[id].path().string());
            }

            if (0 == success_filled)
            {
                m_successful_property_count ++;
            }
        }
        std::cout << "Filled " << m_successful_property_count << " properties successfully.\n";
        return m_successful_property_count;
    }   
    // Launch simulations 
    int launch()
    {
        if (0 == m_successful_property_count)
        {
            std::cout << "m_successful_property_count = 0\n";
            return -1;
        }
        int batches = m_successful_property_count/m_batch_size;
        std::cout << "Parameters: exh = " << m_exhaustiveness << \
            ", box[0] = " << m_box_size[0] << \
            ", max_eval_steps global = " << m_max_global_steps << \
            ", num_modes = " << m_num_modes << \
            ", refine_steps = " << m_refine_steps << "\n";

        std::cout << "To do [" << batches << "] batches\n";
        std::cout << "Batched output to " << m_work_dir << "\n";

        if (!boost::filesystem::exists(m_work_dir))
        {
            std::cout << "Creating work dir " << m_work_dir  << "\n";
            boost::filesystem::create_directory(m_work_dir);
        }


        std::vector<complex_property> cp;
	    int total_err_count = 0;
        for (int i = 0;i < batches;i ++)
        {            
            for (int curr = 0;curr < m_batch_size;curr ++)
            {
                int index = i*m_batch_size + curr;
                cp.emplace_back(m_ptr_complex_property_holder->m_properties[index]);
                std::cout << "Processing " << m_ptr_complex_property_holder->m_properties[index].ligand_name << "\n";
            }
            // run
            int err_count = batch_dock_with_worker(cp, m_local_only, m_work_dir, m_input_path, m_out_phrase);  
            std::cout << "Batch [" << i+1 << "/" << batches << "] completed. " << err_count << " errors.\n";
	        total_err_count += err_count;
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
            int err_count = batch_dock_with_worker(cp, m_local_only, m_work_dir, m_input_path, m_out_phrase);
            total_err_count += err_count;
	        cp.clear();
        }
        std::cout << "Remaining [" << remaining << "/" << m_complex_names.size() << "] completed.\n" << total_err_count << " Total errors\n";

        return total_err_count;
    };

    struct err_counter
    {
        std::atomic<int> err_count;
        void update()
        {
            err_count ++;
        }
        int get()
        {
            return err_count;
        }
        void clear()
        {
            err_count = 0;
        }
    };
    err_counter counter;

// Launches a batch of vcw workers in separate threads
// to perform 1:1 docking.
// Each launch uses CUDA stream for concurrent operation of the batches

    int batch_dock_with_worker(
                std::vector<complex_property> props, 
                bool local_only,
                std::string workdir,
                std::string input_dir,
                std::string out_phrase)
    {
        std::vector<std::thread> worker_threads;

        counter.clear();

        for (int i = 0;i < props.size();i ++)
        {
            worker_threads.emplace_back(std::thread(
                [=]()
                {
                    vina_cuda_worker vcw(m_seed, m_num_modes, m_refine_steps, props[i].center_x, props[i].center_y, 
                            props[i].center_z, props[i].protein_name,props[i].ligand_name,
                            local_only, std::vector<double>{props[i].box_x, props[i].box_y, props[i].box_z}, m_max_global_steps, m_verbosity,
                            m_exhaustiveness, workdir, input_dir, out_phrase, m_device_id);
                    try
                    {
                        int ret = vcw.launch();
                        if (ret)
                        {
                            counter.update();
                        }
                    }
                    catch(const std::exception& e)
                    {
                        std::cerr << "Exception processing " << props[i].ligand_name << ", " << e.what() << "\n";
			            counter.update();
                    }
                }
            )
            );
        }
        for (int i = 0;i < props.size();i ++)
        {
            worker_threads[i].join();
        }
	    return counter.get();
    }
}; // simulation_container
