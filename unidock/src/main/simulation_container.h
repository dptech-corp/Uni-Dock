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


    std::vector<std::string> m_complex_names;
    std::vector<boost::filesystem::directory_entry> m_ligand_paths;
    std::vector<boost::filesystem::directory_entry> m_ligand_config_paths;
    std::vector<boost::filesystem::directory_entry> m_protein_paths;
    complex_property_holder * m_ptr_complex_property_holder;

    simulation_container(std::string work_dir, std::string input_path, std::string out_phrase, 
                int batch_size, int box_size, bool local_only, int max_eval_steps, int max_limits):
        m_work_dir(work_dir),
        m_input_path(input_path),
        m_out_phrase(out_phrase),
        m_batch_size(batch_size),
        m_box_size(box_size),
        m_local_only(local_only),
        m_max_global_steps(max_eval_steps),
        m_max_limits(max_limits)
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

    void fill_config(complex_property & cp, std::string path, std::string name)
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

        cp.complex_name = name;

        ifs.close();

    }

    int prime()
    {
        int curr_entry_size = 0;
        std::string effective_path = m_work_dir + '/' + m_input_path;

        if (!boost::filesystem::exists(effective_path))
        {
            std::cout << "Error: Input path " << effective_path << " does not exist\n";
            return -1;
        }

        for (boost::filesystem::directory_entry& entry : boost::filesystem::directory_iterator(effective_path))
        {
            int pos = entry.path().string().find("_ligand.pdbqt");

            if (pos != std::string::npos)
            {
                int pos_complex = entry.path().stem().string().find("_ligand");
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
        std::cout << "Found " << m_complex_names.size() << "\n";

        m_ptr_complex_property_holder = new complex_property_holder(m_complex_names.size());

        int id = 0;
        for (complex_property & cp: *m_ptr_complex_property_holder)
        {
            fill_config(cp, m_ligand_config_paths[id].path().string(), m_complex_names[id]);
            id ++;
        }

        return 0;
    }   
    // Launch simulations 
    int launch()
    {
        int batches = m_complex_names.size()/m_batch_size;
        std::cout << "To do [" << batches << "] batches, box = " << m_box_size << " max_eval_steps global = " << m_max_global_steps << "\n";

        std::vector<complex_property> cp;
        for (int i = 0;i < batches;i ++)
        {            
            for (int curr = 0;curr < m_batch_size;curr ++)
            {
                int index = i*m_batch_size + curr;
                cp.emplace_back(m_ptr_complex_property_holder->m_properties[index]);
                std::cout << "Processing " << m_ptr_complex_property_holder->m_properties[index].complex_name << "\n";
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
                            props[i].center_z, props[i].complex_name,
                            local_only, m_box_size, m_max_global_steps,
                            workdir, input_dir, out_phrase);
                    vcw.launch();
                    vcw.wait_for_completion();                        
                }
            )
            );
        }
        for (int i = 0;i < props.size();i ++)
        {
            worker_threads[i].join();
        }
    }    
}; // simulation_container

