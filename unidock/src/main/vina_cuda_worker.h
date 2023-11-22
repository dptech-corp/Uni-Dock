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
    int refine_step = 5;
    bool local_only = false;
    double energy_range = 3.0;
    bool keep_H = true;
    std::string sf_name = "vina";
    int cpu = 0;
    bool no_refine = false;
    int size_x = 25;
    int size_y = 25;
    int size_z = 25;
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
    }
public:            
    vina_cuda_worker(
            double center_x, 
            double center_y, 
            double center_z, 
            std::string protein_name,
            std::string ligand_name,
            bool local_only,
            int box_size,
            int max_step,
            int verbosity,
            std::string workdir,
            std::string input_dir,
            std::string out_phrase):

            workdir(workdir),
            input_dir(input_dir),
            center_x(center_x),
            center_y(center_y),
            center_z(center_z),
            size_x(box_size),
            size_y(box_size),
            size_z(box_size),
            max_step(max_step),
            protein_name(protein_name),
            ligand_name(ligand_name),
            local_only(local_only),
            out_dir(out_phrase),
            Vina{"vina", 0, seed, verbosity, false, NULL}
    {
        init(out_phrase);
    }
    vina_cuda_worker(
            double center_x, 
            double center_y, 
            double center_z, 
            std::string protein_name,
            std::string ligand_name,
            bool local_only,
            int box_size,
            int max_step,
            int verbosity,
            std::string workdir,
            std::string input_dir,
            std::string out_phrase,
            const Vina & v):

            workdir(workdir),
            input_dir(input_dir),
            center_x(center_x),
            center_y(center_y),
            center_z(center_z),
            size_x(box_size),
            size_y(box_size),
            size_z(box_size),
            max_step(max_step),
            protein_name(protein_name),
            ligand_name(ligand_name),
            local_only(local_only),
            out_dir(out_phrase),
            Vina(v)
    {
        init(out_phrase);
    }        

    ~vina_cuda_worker()
    {

    }


    // Performs CUDA Stream based docking of 1 ligand and 1 protein

    void launch()
    {
        multi_bias = false;
        
        set_vina_weights(weight_gauss1, weight_gauss2, weight_repulsion, weight_hydrophobic,
                                weight_hydrogen, weight_glue, weight_rot);        
        std::string flex;
        std::string rigid(protein_name);

        if (! boost::filesystem::exists( ligand_name ) )
        {
            std::cout << "Input ligand file does not exist ("  << ligand_name << ")\n";        
            return;
        }        
        if (! boost::filesystem::exists( rigid ) )
        {
            std::cout << "Input (rigid) protein file does not exist" << rigid << "\n";        
            return;
        }    
                
        set_receptor(rigid, flex);                                

        enable_gpu();
        compute_vina_maps(center_x, center_y, center_z, size_x, size_y, size_z,
                                                grid_spacing, force_even_voxels);

        auto parsed_ligand = parse_ligand_from_file_no_failure(
            ligand_name, m_scoring_function.get_atom_typing(), keep_H);
        batch_ligands.emplace_back(parsed_ligand);

        set_ligand_from_object_gpu(batch_ligands);

        global_search_gpu_prime(
                                exhaustiveness, num_modes, min_rmsd, max_evals, max_step,
                                1, (unsigned long long)seed,
                                local_only);
        global_search_gpu_run();
    }

    void wait_for_completion()
    {
        global_search_gpu_obtain(1, refine_step);
        std::vector<std::string> gpu_out_name;
        gpu_out_name.emplace_back(default_output(get_filename(ligand_name), out_dir));
        write_poses_gpu(gpu_out_name, num_modes, energy_range);
    }

    void global_search_gpu_prime(
                                const int exhaustiveness = 8, const int n_poses = 20,
                           const double min_rmsd = 1.0, const int max_evals = 0,
                           const int max_step = 0, int num_of_ligands = 1,
                           unsigned long long seed = 181129,
                           const bool local_only = false) {
        // Vina search (Monte-carlo and local optimization)
        // Check if ff, box and ligand were initialized
        if (!m_ligand_initialized) {
            std::cerr << "ERROR: Cannot do the global search. Ligand(s) was(ere) not initialized.\n";
            exit(EXIT_FAILURE);
        } else if (!m_map_initialized) {
            std::cerr << "ERROR: Cannot do the global search. Affinity maps were not initialized.\n";
            exit(EXIT_FAILURE);
        } else if (exhaustiveness < 1) {
            std::cerr << "ERROR: Exhaustiveness must be 1 or greater";
            exit(EXIT_FAILURE);
        }

        if (exhaustiveness < m_cpu) {
            std::cerr << "WARNING: At low exhaustiveness, it may be impossible to utilize all CPUs.\n";
        }

        double e = 0;
        double intramolecular_energy = 0;
        const vec authentic_v(1000, 1000, 1000);
        model best_model;
        boost::optional<model> ref;

        std::stringstream sstm;
        rng generator(static_cast<rng::result_type>(m_seed));

        // set global_steps with cutoff, maximun for the first version
        sz heuristic = 0;
        for (int i = 0; i < num_of_ligands; ++i) {
            heuristic
                = std::max(heuristic, m_model_gpu[i].num_movable_atoms()
                                        + 10 * m_model_gpu[i].get_size().num_degrees_of_freedom());
            mc.local_steps = unsigned((25 + m_model_gpu[i].num_movable_atoms()) / 3);
        }
        mc.global_steps = unsigned(70 * 3 * (50 + heuristic) / 2);  // 2 * 70 -> 8 * 20 // FIXME
        // DEBUG_PRINTF("mc.global_steps = %u, max_step = %d, ��unsigned)max_step=%u\n",
        // mc.global_steps, max_step, (unsigned)max_step);
        if (max_step > 0 && mc.global_steps > (unsigned)max_step) {
            mc.global_steps = (unsigned)max_step;
        }
        // DEBUG_PRINTF("final mc.global_steps = %u\n", mc.global_steps);
        mc.max_evals = max_evals;
        mc.min_rmsd = min_rmsd;
        mc.num_saved_mins = n_poses;
        mc.hunt_cap = vec(10, 10, 10);
        mc.threads_per_ligand = exhaustiveness;
        mc.num_of_ligands = num_of_ligands;
        mc.thread = exhaustiveness * num_of_ligands;
        mc.local_only = local_only;

        // Docking search

        if (m_sf_choice == SF_VINA) {
            mc.ptr_gpu_state = mc.gpu_prime(m_model_gpu, m_precalculated_byatom_gpu, m_data_list_gpu, 
                m_grid,
            m_grid.corner1(), m_grid.corner2(), generator, m_verbosity, seed, bias_batch_list);
        }
    }

    void global_search_gpu_run()
    {
        std::stringstream sstm;

        sstm << "Performing docking (random seed: " << m_seed << ")";

        mc.gpu_run_kernel(mc.ptr_gpu_state, m_seed);

        done(m_verbosity, 1);
    }

    void global_search_gpu_obtain(
                        int num_of_ligands, 
                            const int refine_step = 5)
    {
        std::vector<output_container> poses_gpu;
        output_container poses;  // temp output_container
        poses_gpu.resize(num_of_ligands, poses);

        model best_model;
        boost::optional<model> ref;
        double intramolecular_energy = 0;
        const vec authentic_v(1000, 1000, 1000);

        // Obtain GPU results from mc object
        mc.gpu_obtain(mc.ptr_gpu_state, poses_gpu, m_precalculated_byatom_gpu, m_data_list_gpu  );

        // Docking post-processing and rescoring
        non_cache m_non_cache_tmp = m_non_cache;

        for (int l = 0; l < num_of_ligands; ++l) {
            DEBUG_PRINTF("num_output_poses before remove=%lu\n", poses_gpu[l].size());
            poses = remove_redundant(poses_gpu[l], mc.min_rmsd);
            DEBUG_PRINTF("num_output_poses=%lu\n", poses.size());

            if (!poses.empty()) {
                DEBUG_PRINTF("energy=%lf\n", poses[0].e);
                DEBUG_PRINTF("vina: poses not empty, poses.size()=%lu\n", poses.size());
                // For the Vina scoring function, we take the intramolecular energy from the best pose
                // the order must not change because of non-decreasing g (see paper), but we'll re-sort
                // in case g is non strictly increasing
                if (m_sf_choice == SF_VINA || m_sf_choice == SF_VINARDO) {
                    // Refine poses if no_refine is false and got receptor
                    if (!m_no_refine & m_receptor_initialized) {
                        change g(m_model_gpu[l].get_size());
                        quasi_newton quasi_newton_par;
                        const vec authentic_v(1000, 1000, 1000);
                        int evalcount = 0;
                        const fl slope = 1e6;
                        m_non_cache = m_non_cache_tmp;
                        m_non_cache.slope = slope;
                        quasi_newton_par.max_steps
                            = unsigned((25 + m_model_gpu[l].num_movable_atoms()) / 3);
                        VINA_FOR_IN(i, poses) {
                            // DEBUG_PRINTF("poses i score=%lf\n", poses[i].e);
                            const fl slope_orig = m_non_cache.slope;
                            VINA_FOR(p, refine_step) {
                                m_non_cache.slope = 100 * std::pow(10.0, 2.0 * p);
                                quasi_newton_par(m_model_gpu[l], m_precalculated_byatom_gpu[l],
                                                m_non_cache, poses[i], g, authentic_v, evalcount);
                                if (m_non_cache.within(m_model_gpu[l])) break;
                            }
                            poses[i].coords = m_model_gpu[l].get_heavy_atom_movable_coords();
                            if (!m_non_cache.within(m_model_gpu[l])) poses[i].e = max_fl;
                            m_non_cache.slope = slope;
                        }
                    }
                    poses.sort();
                    // probably for bug very negative score
                    m_model_gpu[l].set(poses[0].c);

                    if (m_no_refine || !m_receptor_initialized)
                        intramolecular_energy = m_model_gpu[l].eval_intramolecular(
                            m_precalculated_byatom_gpu[l], m_grid, authentic_v);
                    else
                        intramolecular_energy = m_model_gpu[l].eval_intramolecular(
                            m_precalculated_byatom_gpu[l], m_non_cache, authentic_v);
                }

                for (int i = 0; i < poses.size(); ++i) {
                    if (m_verbosity > 1) std::cout << "ENERGY FROM SEARCH: " << poses[i].e << "\n";

                    m_model_gpu[l].set(poses[i].c);

                    // For AD42 intramolecular_energy is equal to 0
                    // m_model = m_model_gpu[l]; // Vina::score() will use m_model and
                    // m_precalculated_byatom m_precalculated_byatom = m_precalculated_byatom_gpu[l];
                    DEBUG_PRINTF("intramolecular_energy=%f\n", intramolecular_energy);
                    std::vector<double> energies = score_gpu(l, intramolecular_energy);
                    // DEBUG_PRINTF("energies.size()=%d\n", energies.size());
                    // Store energy components in current pose
                    poses[i].e = energies[0];  // specific to each scoring function
                    poses[i].inter = energies[1] + energies[2];
                    poses[i].intra = energies[3] + energies[4] + energies[5];
                    poses[i].total = poses[i].inter + poses[i].intra;  // cost function for optimization
                    poses[i].conf_independent = energies[6];           // "torsion"
                    poses[i].unbound = energies[7];  // specific to each scoring function

                    if (m_verbosity > 1) {
                        std::cout << "FINAL ENERGY: \n";
                        show_score(energies);
                    }
                }

                // Since pose.e contains the final energy, we have to sort them again
                poses.sort();

                // Now compute RMSD from the best model
                // Necessary to do it in two pass for AD4 scoring function
                m_model_gpu[l].set(poses[0].c);
                best_model = m_model_gpu[l];

                if (m_verbosity > 0) {
                    std::cout << '\n';
                    std::cout << "mode |   affinity | dist from best mode\n";
                    std::cout << "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n";
                    std::cout << "-----+------------+----------+----------\n";
                }

                VINA_FOR_IN(i, poses) {
                    m_model_gpu[l].set(poses[i].c);

                    // Get RMSD between current pose and best_model
                    const model& r = ref ? ref.get() : best_model;
                    poses[i].lb = m_model_gpu[l].rmsd_lower_bound(r);
                    poses[i].ub = m_model_gpu[l].rmsd_upper_bound(r);

                    if (m_verbosity > 0) {
                        std::cout << std::setw(4) << i + 1 << "    " << std::setw(9)
                                << std::setprecision(4) << poses[i].e;
                        std::cout << "  " << std::setw(9) << std::setprecision(4) << poses[i].lb;
                        std::cout << "  " << std::setw(9) << std::setprecision(4) << poses[i].ub
                                << "\n";
                    }
                }

                // Clean up by putting back the best pose in model
                m_model_gpu[l].set(poses[0].c);
            } else {
                std::cerr << "WARNING: Could not find any conformations completely within the search "
                            "space.\n";
                std::cerr << "WARNING: Check that it is large enough for all movable atoms, including "
                            "those in the flexible side chains.\n";
                std::cerr << "WARNING: Or could not successfully parse PDBQT input file of ligand #"
                        << l << std::endl;
            }
            // Store results in Vina object
            m_poses_gpu[l] = poses;
        }    
    }


    // Protectors
    vina_cuda_worker (const vina_cuda_worker&);
    vina_cuda_worker& operator=(const vina_cuda_worker&);
};
