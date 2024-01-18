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

#ifndef VINA_MONTE_CARLO_H
#define VINA_MONTE_CARLO_H

#include "incrementable.h"
#include "model.h"
#include "kernel.h"
#include "grid.h"
#include "precalculate.h"

struct monte_carlo {
    unsigned max_evals;
    unsigned global_steps;
    fl temperature;
    vec hunt_cap;
    fl min_rmsd;
    sz num_saved_mins;
    fl mutation_amplitude;
    unsigned local_steps;
    unsigned threads_per_ligand;
    unsigned num_of_ligands;
    bool local_only;
    unsigned thread = 2048;  // for CUDA parallel option, num_of_ligands * threads_per_ligand
    // T = 600K, R = 2cal/(K*mol) -> temperature = RT = 1.2;  global_steps = 50*lig_atoms = 2500
    monte_carlo()
        : max_evals(0),
          global_steps(2500),
          threads_per_ligand(2048),
          temperature(1.2),
          hunt_cap(10, 1.5, 10),
          min_rmsd(0.5),
          num_saved_mins(50),
          mutation_amplitude(2) {}

    output_type operator()(model& m, const precalculate_byatom& p, const igrid& ig,
                           const vec& corner1, const vec& corner2, rng& generator) const;
    // out is sorted
    void operator()(model& m, output_container& out, const precalculate_byatom& p, const igrid& ig,
                    const vec& corner1, const vec& corner2, rng& generator) const;
    void operator()(std::vector<model>& m, std::vector<output_container>& out,
                    std::vector<precalculate_byatom>& p, triangular_matrix_cuda_t* m_data_list_gpu,
                    const igrid& ig, const vec& corner1, const vec& corner2, rng& generator,
                    int verbosity, unsigned long long seed,
                    std::vector<std::vector<bias_element> >& bias_batch_list) const;
    void mc_stream(std::vector<model>& m, std::vector<output_container>& out,
                    std::vector<precalculate_byatom>& p, triangular_matrix_cuda_t* m_data_list_gpu,
                    const igrid& ig, const vec& corner1, const vec& corner2, rng& generator,
                    int verbosity, unsigned long long seed,
                    std::vector<std::vector<bias_element> >& bias_batch_list) const;

    std::vector<output_type> cuda_to_vina(output_type_cuda_t* results_p, int thread) const;
};

#endif
