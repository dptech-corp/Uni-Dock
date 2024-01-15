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

#include "common.cuh"
#include "curand_kernel.h"
#include "kernel.h"
#include "math.h"
#include "warp_ops.cuh"
#include <cmath>
#include <vector>
/* Original Include files */
#include "ad4cache.h"
#include "cache.h"
#include "coords.h"
#include "model.h"
#include "monte_carlo.h"
#include "mutate.h"
#include "precalculate.h"
#include "quasi_newton.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

/* Below is monte-carlo kernel, based on kernel.cl*/

__device__ __forceinline__ void get_heavy_atom_movable_coords(output_type_cuda_t *tmp,
                                                              const m_cuda_t *m_cuda_gpu) {
    int counter = 0;
    for (int i = 0; i < m_cuda_gpu->m_num_movable_atoms; i++) {
        if (m_cuda_gpu->atoms[i].types[0] != EL_TYPE_H) {
            for (int j = 0; j < 3; j++) tmp->coords[counter][j] = m_cuda_gpu->m_coords.coords[i][j];
            counter++;
        } else {
            // DEBUG_PRINTF("\n P2: removed H atom coords in
            // get_heavy_atom_movable_coords()!");
        }
    }
    /* assign 0 for others */
    for (int i = counter; i < MAX_NUM_OF_ATOMS; i++) {
        for (int j = 0; j < 3; j++) tmp->coords[i][j] = 0;
    }
}

__device__ __forceinline__ float generate_n(const float *pi_map, const int step) {
    return fabs(pi_map[step]) / M_PI_F;
}

__device__ __forceinline__ bool metropolis_accept(float old_f, float new_f, float temperature,
                                                  float n) {
    if (new_f < old_f) return true;
    const float acceptance_probability = exp((old_f - new_f) / temperature);
    return n < acceptance_probability;
}

__device__ __forceinline__ void write_back(output_type_cuda_t *results,
                                           const output_type_cuda_t *best_out) {
    for (int i = 0; i < 3; i++) results->position[i] = best_out->position[i];
    for (int i = 0; i < 4; i++) results->orientation[i] = best_out->orientation[i];
    for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)
        results->lig_torsion[i] = best_out->lig_torsion[i];
    for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)
        results->flex_torsion[i] = best_out->flex_torsion[i];
    results->lig_torsion_size = best_out->lig_torsion_size;
    results->e = best_out->e;
    for (int i = 0; i < MAX_NUM_OF_ATOMS; i++) {
        for (int j = 0; j < 3; j++) {
            results->coords[i][j] = best_out->coords[i][j];
        }
    }
}
// MAX_THREADS_PER_BLOCK and MIN_BLOCKS_PER_MP should be adjusted according to
// the profiling results
#define MAX_THREADS_PER_BLOCK 32
#define MIN_BLOCKS_PER_MP 32
template <unsigned int TileSize = 32>
__global__ __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) void kernel(
    m_cuda_t *m_cuda_global, ig_cuda_t *ig_cuda_gpu, p_cuda_t *p_cuda_gpu,
    float *rand_molec_struc_gpu, int bfgs_max_steps, float mutation_amplitude,
    curandStatePhilox4_32_10_t *states, unsigned long long seed, float epsilon_fl,
    float *hunt_cap_gpu, float *authentic_v_gpu, output_type_cuda_t *results,
    output_type_cuda_t *output_aux, change_cuda_t *change_aux, pot_cuda_t *pot_aux,
    matrix_d *h_cuda_gpu, m_cuda_t *m_cuda_gpu, int search_depth, int num_of_ligands,
    int threads_per_ligand, bool multi_bias) {
    int bid = blockIdx.x, tid = threadIdx.x;
    int pose_id = (bid * blockDim.x + tid) / TileSize;
    if (m_cuda_global[pose_id / threads_per_ligand].m_num_movable_atoms == -1) {
        return;
    }

    auto tb = cg::this_thread_block();
    cg::thread_block_tile<TileSize> tile = cg::tiled_partition<TileSize>(tb);

    float best_e = INFINITY;
    output_type_cuda_t &tmp = output_aux[pose_id * 5];
    output_type_cuda_t &best_out = output_aux[pose_id * 5 + 1];
    output_type_cuda_t &candidate = output_aux[pose_id * 5 + 2];
    output_type_cuda_t &x_new = output_aux[pose_id * 5 + 3];
    output_type_cuda_t &x_orig = output_aux[pose_id * 5 + 4];

    change_cuda_t &g = change_aux[pose_id * 6];
    change_cuda_t &tmp1 = change_aux[pose_id * 6 + 1];
    change_cuda_t &tmp2 = change_aux[pose_id * 6 + 2];
    change_cuda_t &tmp3 = change_aux[pose_id * 6 + 3];
    change_cuda_t &tmp4 = change_aux[pose_id * 6 + 4];
    change_cuda_t &tmp5 = change_aux[pose_id * 6 + 5];

    if (pose_id < num_of_ligands * threads_per_ligand) {
        output_type_cuda_init_warp(
            tile, &tmp, rand_molec_struc_gpu + pose_id * (SIZE_OF_MOLEC_STRUC / sizeof(float)));

        m_cuda_init_with_m_cuda_warp(tile, &m_cuda_global[pose_id / threads_per_ligand],
                                     &m_cuda_gpu[pose_id]);

        if (tile.thread_rank() == 0) {
            curand_init(seed, pose_id, 0, &states[pose_id]);
            g.lig_torsion_size = tmp.lig_torsion_size;
        }
        tile.sync();

        if (multi_bias) {
            ig_cuda_gpu += pose_id / threads_per_ligand;
        }

        pot_aux += pose_id;
        p_cuda_gpu += pose_id / threads_per_ligand;

        // BFGS
        for (int step = 0; step < search_depth; step++) {
            output_type_cuda_init_with_output_warp(tile, &candidate, &tmp);

            if (tile.thread_rank() == 0)
                mutate_conf_cuda(bfgs_max_steps, &candidate, &states[pose_id],
                                 m_cuda_gpu[pose_id].ligand.begin, m_cuda_gpu[pose_id].ligand.end,
                                 m_cuda_gpu[pose_id].atoms, &m_cuda_gpu[pose_id].m_coords,
                                 m_cuda_gpu[pose_id].ligand.rigid.origin[0], epsilon_fl,
                                 mutation_amplitude);
            tile.sync();

            bfgs_warp(tile, &candidate, &x_new, &x_orig, &g, &tmp1, &tmp2, &tmp3, &tmp4, &tmp5,
                      &h_cuda_gpu[pose_id], &m_cuda_gpu[pose_id], p_cuda_gpu, ig_cuda_gpu, pot_aux,
                      hunt_cap_gpu, epsilon_fl, bfgs_max_steps);

            bool accepted;
            if (tile.thread_rank() == 0) {
                // n ~ U[0,1]
                float n = curand_uniform(&states[pose_id]);
                accepted = metropolis_accept(tmp.e, candidate.e, 1.2, n);
            }
            accepted = tile.shfl(accepted, 0);

            if (step == 0 || accepted) {
                output_type_cuda_init_with_output_warp(tile, &tmp, &candidate);

                if (tile.thread_rank() == 0) {
                    set(&tmp, &m_cuda_gpu[pose_id].ligand.rigid, &m_cuda_gpu[pose_id].m_coords,
                        m_cuda_gpu[pose_id].atoms, m_cuda_gpu[pose_id].m_num_movable_atoms,
                        epsilon_fl);
                }
                tile.sync();

                if (tmp.e < best_e) {
                    bfgs_warp(tile, &tmp, &x_new, &x_orig, &g, &tmp1, &tmp2, &tmp3, &tmp4, &tmp5,
                              &h_cuda_gpu[pose_id], &m_cuda_gpu[pose_id], p_cuda_gpu, ig_cuda_gpu,
                              pot_aux, authentic_v_gpu, epsilon_fl, bfgs_max_steps);

                    // set
                    if (tmp.e < best_e) {
                        if (tile.thread_rank() == 0)
                            set(&tmp, &m_cuda_gpu[pose_id].ligand.rigid,
                                &m_cuda_gpu[pose_id].m_coords, m_cuda_gpu[pose_id].atoms,
                                m_cuda_gpu[pose_id].m_num_movable_atoms, epsilon_fl);
                        tile.sync();

                        output_type_cuda_init_with_output_warp(tile, &best_out, &tmp);

                        if (tile.thread_rank() == 0) {
                            get_heavy_atom_movable_coords(&best_out,
                                                          &m_cuda_gpu[pose_id]);  // get coords
                        }
                        tile.sync();

                        best_e = tmp.e;
                    }
                }
            }
        }

        // write the best conformation back to CPU // FIX?? should add more
        write_back_warp(tile, results + pose_id, &best_out);
    }
}

/* Above based on kernel.cl */


__host__ void monte_carlo::mc_stream(
    std::vector<model> &m_gpu, std::vector<output_container> &out_gpu,
    std::vector<precalculate_byatom> &p_gpu, triangular_matrix_cuda_t *m_data_list_gpu,
    const igrid &ig, const vec &corner1, const vec &corner2, rng &generator, int verbosity,
    unsigned long long seed, std::vector<std::vector<bias_element>> &bias_batch_list) const {
    /* Definitions from vina1.2 */
    DEBUG_PRINTF("entering CUDA monte_carlo search\n");  // debug

    cudaStream_t curr_stream = 0;
    checkCUDA(cudaStreamCreate ( &curr_stream));
    DEBUG_PRINTF("Stream created [0x%p]\n", curr_stream);

    vec authentic_v(1000, 1000,
                    1000);  // FIXME? this is here to avoid max_fl/max_fl

    quasi_newton quasi_newton_par;
    const int quasi_newton_par_max_steps = local_steps;  // no need to decrease step

    /* Allocate CPU memory and define new data structure */
    DEBUG_PRINTF("Allocating CPU memory\n");  // debug
    m_cuda_t *m_cuda;
    checkCUDA(cudaMallocHost(&m_cuda, sizeof(m_cuda_t)));
    memset(m_cuda, 0, sizeof(m_cuda_t));

    output_type_cuda_t *rand_molec_struc_tmp;
    checkCUDA(cudaMallocHost(&rand_molec_struc_tmp, sizeof(output_type_cuda_t)));
    memset(rand_molec_struc_tmp, 0, sizeof(output_type_cuda_t));

    ig_cuda_t *ig_cuda_ptr;
    checkCUDA(cudaMallocHost(&ig_cuda_ptr, sizeof(ig_cuda_t)));
    memset(ig_cuda_ptr, 0, sizeof(ig_cuda_t));

    p_cuda_t_cpu *p_cuda;
    checkCUDA(cudaMallocHost(&p_cuda, sizeof(p_cuda_t_cpu)));
    memset(p_cuda, 0, sizeof(p_cuda_t_cpu));

    /* End CPU allocation */

    /* Allocate GPU memory */
    DEBUG_PRINTF("Allocating GPU memory\n");
    size_t m_cuda_size = sizeof(m_cuda_t);
    DEBUG_PRINTF("m_cuda_size=%lu\n", m_cuda_size);
    size_t ig_cuda_size = sizeof(ig_cuda_t);
    DEBUG_PRINTF("ig_cuda_size=%lu\n", ig_cuda_size);
    DEBUG_PRINTF("p_cuda_size_cpu=%lu\n", sizeof(p_cuda_t_cpu));

    size_t p_cuda_size_gpu = sizeof(p_cuda_t);
    DEBUG_PRINTF("p_cuda_size_gpu=%lu\n", p_cuda_size_gpu);

    // rand_molec_struc_gpu
    float *rand_molec_struc_gpu;
    checkCUDA(cudaMalloc(&rand_molec_struc_gpu, thread * SIZE_OF_MOLEC_STRUC));
    checkCUDA(cudaMemsetAsync(rand_molec_struc_gpu, 0, thread * SIZE_OF_MOLEC_STRUC, curr_stream));

    float epsilon_fl_float = static_cast<float>(epsilon_fl);

    // use cuRand to generate random values on GPU
    curandStatePhilox4_32_10_t *states;
    DEBUG_PRINTF("random states size=%lu\n", sizeof(curandStatePhilox4_32_10_t) * thread);
    checkCUDA(cudaMalloc(&states, sizeof(curandStatePhilox4_32_10_t) * thread));
    checkCUDA(cudaMemsetAsync(states, 0, sizeof(curandStatePhilox4_32_10_t) * thread, curr_stream));

    // hunt_cap_gpu
    float *hunt_cap_gpu;
    float hunt_cap_float[3] = {static_cast<float>(hunt_cap[0]), static_cast<float>(hunt_cap[1]),
                               static_cast<float>(hunt_cap[2])};

    checkCUDA(cudaMalloc(&hunt_cap_gpu, 3 * sizeof(float)));
    checkCUDA(cudaMemsetAsync(hunt_cap_gpu, 0, 3 * sizeof(float), curr_stream));
    // Preparing m related data
    m_cuda_t *m_cuda_gpu;
    DEBUG_PRINTF("m_cuda_size=%lu", m_cuda_size);
    checkCUDA(cudaMalloc(&m_cuda_gpu, num_of_ligands * m_cuda_size));
    checkCUDA(cudaMemsetAsync(m_cuda_gpu, 0, num_of_ligands * m_cuda_size, curr_stream));
    // Preparing p related data

    p_cuda_t *p_cuda_gpu;
    checkCUDA(cudaMalloc(&p_cuda_gpu, num_of_ligands * p_cuda_size_gpu));
    checkCUDA(cudaMemsetAsync(p_cuda_gpu, 0, num_of_ligands * p_cuda_size_gpu, curr_stream));
    DEBUG_PRINTF("p_cuda_gpu=%p\n", p_cuda_gpu);
    // Preparing ig related data (cache related data)
    ig_cuda_t *ig_cuda_gpu;

    float *authentic_v_gpu;
    float authentic_v_float[3]
        = {static_cast<float>(authentic_v[0]), static_cast<float>(authentic_v[1]),
           static_cast<float>(authentic_v[2])};

    checkCUDA(cudaMalloc(&authentic_v_gpu, sizeof(authentic_v_float)));
    checkCUDA(cudaMemsetAsync(authentic_v_gpu, 0, sizeof(authentic_v_float), curr_stream));
    // Preparing result data
    output_type_cuda_t *results_gpu;
    checkCUDA(cudaMalloc(&results_gpu, thread * sizeof(output_type_cuda_t)));
    checkCUDA(cudaMemsetAsync(results_gpu, 0, thread * sizeof(output_type_cuda_t), curr_stream));

    m_cuda_t *m_cuda_global;
    checkCUDA(cudaMalloc(&m_cuda_global, thread * sizeof(m_cuda_t)));
    checkCUDA(cudaMemsetAsync(m_cuda_global, 0, thread * sizeof(m_cuda_t), curr_stream));

    matrix_d *h_cuda_global;
    checkCUDA(cudaMalloc(&h_cuda_global, thread * sizeof(matrix_d)));
    checkCUDA(cudaMemsetAsync(h_cuda_global, 0, thread * sizeof(matrix_d), curr_stream));

    /* End Allocating GPU Memory */

    assert(num_of_ligands <= MAX_LIGAND_NUM);
    assert(thread <= MAX_THREAD);

    struct tmp_struct {
        int start_index = 0;
        int parent_index = 0;
        void store_node(tree<segment> &child_ptr, rigid_cuda_t &rigid) {
            start_index++;  // start with index 1, index 0 is root node
            rigid.parent[start_index] = parent_index;
            rigid.atom_range[start_index][0] = child_ptr.node.begin;
            rigid.atom_range[start_index][1] = child_ptr.node.end;
            for (int i = 0; i < 9; i++)
                rigid.orientation_m[start_index][i] = child_ptr.node.get_orientation_m().data[i];
            rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
            rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
            rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
            rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
            for (int i = 0; i < 3; i++) {
                rigid.origin[start_index][i] = child_ptr.node.get_origin()[i];
                rigid.axis[start_index][i] = child_ptr.node.get_axis()[i];
                rigid.relative_axis[start_index][i] = child_ptr.node.relative_axis[i];
                rigid.relative_origin[start_index][i] = child_ptr.node.relative_origin[i];
            }
            if (child_ptr.children.size() == 0)
                return;
            else {
                assert(start_index < MAX_NUM_OF_RIGID);
                int parent_index_tmp = start_index;
                for (int i = 0; i < child_ptr.children.size(); i++) {
                    this->parent_index = parent_index_tmp;  // Update parent index
                    this->store_node(child_ptr.children[i], rigid);
                }
            }
        }
    };

    for (int l = 0; l < num_of_ligands; ++l) {
        model &m = m_gpu[l];
        const precalculate_byatom &p = p_gpu[l];

        /* Prepare m related data */
        conf_size s = m.get_size();
        change g(s);
        output_type tmp(s, 0);
        tmp.c = m.get_initial_conf();

        assert(m.atoms.size() < MAX_NUM_OF_ATOMS);

        // Preparing ligand data
        DEBUG_PRINTF("prepare ligand data\n");
        assert(m.num_other_pairs() == 0);  // m.other_pairs is not supported!
        assert(m.ligands.size() <= 1);     // Only one ligand supported!

        if (m.ligands.size() == 0) {  // ligand parsing error
            m_cuda->m_num_movable_atoms = -1;
            DEBUG_PRINTF("copy m_cuda to gpu, size=%lu\n", sizeof(m_cuda_t));
            checkCUDA(cudaMemcpyAsync(m_cuda_gpu + l, m_cuda, sizeof(m_cuda_t), cudaMemcpyHostToDevice, curr_stream));
        } else {
            for (int i = 0; i < m.atoms.size(); i++) {
                m_cuda->atoms[i].types[0]
                    = m.atoms[i].el;  // To store 4 atoms types (el, ad, xs, sy)
                m_cuda->atoms[i].types[1] = m.atoms[i].ad;
                m_cuda->atoms[i].types[2] = m.atoms[i].xs;
                m_cuda->atoms[i].types[3] = m.atoms[i].sy;
                for (int j = 0; j < 3; j++) {
                    m_cuda->atoms[i].coords[j] = m.atoms[i].coords[j];  // To store atom coords
                }
            }

            // To store atoms coords
            for (int i = 0; i < m.coords.size(); i++) {
                for (int j = 0; j < 3; j++) {
                    m_cuda->m_coords.coords[i][j] = m.coords[i].data[j];
                }
            }

            // To store minus forces
            for (int i = 0; i < m.coords.size(); i++) {
                for (int j = 0; j < 3; j++) {
                    m_cuda->minus_forces.coords[i][j] = m.minus_forces[i].data[j];
                }
            }

            m_cuda->ligand.pairs.num_pairs = m.ligands[0].pairs.size();
            for (int i = 0; i < m_cuda->ligand.pairs.num_pairs; i++) {
                m_cuda->ligand.pairs.type_pair_index[i] = m.ligands[0].pairs[i].type_pair_index;
                m_cuda->ligand.pairs.a[i] = m.ligands[0].pairs[i].a;
                m_cuda->ligand.pairs.b[i] = m.ligands[0].pairs[i].b;
            }
            m_cuda->ligand.begin = m.ligands[0].begin;  // 0
            m_cuda->ligand.end = m.ligands[0].end;      // 29
            ligand &m_ligand = m.ligands[0];            // Only support one ligand
            DEBUG_PRINTF("m_ligand.end=%lu, MAX_NUM_OF_ATOMS=%d\n", m_ligand.end, MAX_NUM_OF_ATOMS);
            assert(m_ligand.end < MAX_NUM_OF_ATOMS);

            // Store root node
            m_cuda->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
            m_cuda->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
            for (int i = 0; i < 3; i++)
                m_cuda->ligand.rigid.origin[0][i] = m_ligand.node.get_origin()[i];
            for (int i = 0; i < 9; i++)
                m_cuda->ligand.rigid.orientation_m[0][i]
                    = m_ligand.node.get_orientation_m().data[i];
            m_cuda->ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
            m_cuda->ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
            m_cuda->ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
            m_cuda->ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
            for (int i = 0; i < 3; i++) {
                m_cuda->ligand.rigid.axis[0][i] = 0;
                m_cuda->ligand.rigid.relative_axis[0][i] = 0;
                m_cuda->ligand.rigid.relative_origin[0][i] = 0;
            }

            // Store children nodes (in depth-first order)
            DEBUG_PRINTF("store children nodes\n");

            tmp_struct ts;
            for (int i = 0; i < m_ligand.children.size(); i++) {
                ts.parent_index = 0;  // Start a new branch, whose parent is 0
                ts.store_node(m_ligand.children[i], m_cuda->ligand.rigid);
            }
            m_cuda->ligand.rigid.num_children = ts.start_index;

            // set children map
            DEBUG_PRINTF("set children map\n");
            for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
                for (int j = 0; j < MAX_NUM_OF_RIGID; j++) {
                    m_cuda->ligand.rigid.children_map[i][j] = false;
                    m_cuda->ligand.rigid.descendant_map[i][j] = false;
                }

            for (int i = MAX_NUM_OF_RIGID - 1; i >= 0; i--) {
                if (i > 0) {
                    m_cuda->ligand.rigid.children_map[m_cuda->ligand.rigid.parent[i]][i] = true;
                    m_cuda->ligand.rigid.descendant_map[m_cuda->ligand.rigid.parent[i]][i] = true;
                }
                for (int j = i + 1; j < MAX_NUM_OF_RIGID; j++) {
                    if (m_cuda->ligand.rigid.descendant_map[i][j])
                        m_cuda->ligand.rigid.descendant_map[m_cuda->ligand.rigid.parent[i]][j]
                            = true;
                }
            }
            m_cuda->m_num_movable_atoms = m.num_movable_atoms();

            DEBUG_PRINTF("copy m_cuda to gpu, size=%lu\n", sizeof(m_cuda_t));
            checkCUDA(cudaMemcpyAsync(m_cuda_gpu + l, m_cuda, sizeof(m_cuda_t), cudaMemcpyHostToDevice, curr_stream));

            /* Prepare rand_molec_struc data */
            int lig_torsion_size = tmp.c.ligands[0].torsions.size();
            DEBUG_PRINTF("lig_torsion_size=%d\n", lig_torsion_size);
            int flex_torsion_size;
            if (tmp.c.flex.size() != 0)
                flex_torsion_size = tmp.c.flex[0].torsions.size();
            else
                flex_torsion_size = 0;
            // std::vector<vec> uniform_data;
            // uniform_data.resize(thread);

            for (int i = 0; i < threads_per_ligand; ++i) {
                if (!local_only) {
                    tmp.c.randomize(corner1, corner2,
                                    generator);  // generate a random structure,
                                                 // can move to GPU if necessary
                }
                for (int j = 0; j < 3; j++)
                    rand_molec_struc_tmp->position[j] = tmp.c.ligands[0].rigid.position[j];
                assert(lig_torsion_size <= MAX_NUM_OF_LIG_TORSION);
                for (int j = 0; j < lig_torsion_size; j++)
                    rand_molec_struc_tmp->lig_torsion[j]
                        = tmp.c.ligands[0].torsions[j];  // Only support one ligand
                assert(flex_torsion_size <= MAX_NUM_OF_FLEX_TORSION);
                for (int j = 0; j < flex_torsion_size; j++)
                    rand_molec_struc_tmp->flex_torsion[j]
                        = tmp.c.flex[0].torsions[j];  // Only support one flex

                rand_molec_struc_tmp->orientation[0]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_1();
                rand_molec_struc_tmp->orientation[1]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_2();
                rand_molec_struc_tmp->orientation[2]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_3();
                rand_molec_struc_tmp->orientation[3]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_4();

                rand_molec_struc_tmp->lig_torsion_size = lig_torsion_size;

                float *rand_molec_struc_gpu_tmp
                    = rand_molec_struc_gpu
                      + (l * threads_per_ligand + i) * SIZE_OF_MOLEC_STRUC / sizeof(float);
                checkCUDA(cudaMemcpyAsync(rand_molec_struc_gpu_tmp, rand_molec_struc_tmp,
                                     SIZE_OF_MOLEC_STRUC, cudaMemcpyHostToDevice, curr_stream));
            }

            /* Preparing p related data */
            DEBUG_PRINTF("Preaparing p related data\n");  // debug

            // copy pointer instead of data
            p_cuda->m_cutoff_sqr = p.m_cutoff_sqr;
            p_cuda->factor = p.m_factor;
            p_cuda->n = p.m_n;
            p_cuda->m_data_size = p.m_data.m_data.size();
            checkCUDA(cudaMemcpyAsync(p_cuda_gpu + l, p_cuda, sizeof(p_cuda_t), cudaMemcpyHostToDevice, curr_stream));
            checkCUDA(cudaMemcpyAsync(&(p_cuda_gpu[l].m_data), &(m_data_list_gpu[l].p_data),
                                 sizeof(p_m_data_cuda_t *),
                                 cudaMemcpyHostToDevice, curr_stream));  // check if fl == float
        }
    }

    /* Prepare data only concerns rigid receptor */

    // Preparing igrid related data
    DEBUG_PRINTF("Preparing ig related data\n");  // debug

    bool multi_bias = (bias_batch_list.size() == num_of_ligands);
    if (multi_bias) {
        // multi bias mode
        std::cout << "with multi bias ";

        checkCUDA(cudaMalloc(&ig_cuda_gpu, ig_cuda_size * num_of_ligands));
        checkCUDA(cudaMemsetAsync(ig_cuda_gpu, 0, ig_cuda_size * num_of_ligands, curr_stream));
        for (int l = 0; l < num_of_ligands; ++l) {
            if (ig.get_atu() == atom_type::XS) {
                cache ig_tmp(ig.get_gd(), ig.get_slope());
                ig_tmp.m_grids = ig.get_grids();
                // // debug
                // if (l == 1){
                // 	std::cout << "writing original grid map\n";
                // 	ig_tmp.write(std::string("./ori"), szv(1,0));
                // }
                ig_tmp.compute_bias(m_gpu[l], bias_batch_list[l]);
                // // debug
                // std::cout << "writing bias\n";
                // ig_tmp.write(std::string("./")+std::to_string(l), szv(1,0));
                ig_cuda_ptr->atu = ig.get_atu();  // atu
                DEBUG_PRINTF("ig_cuda_ptr->atu=%d\n", ig_cuda_ptr->atu);
                ig_cuda_ptr->slope = ig.get_slope();  // slope
                std::vector<grid> tmp_grids = ig.get_grids();
                int grid_size = tmp_grids.size();
                DEBUG_PRINTF("ig.size()=%d, GRIDS_SIZE=%d, should be 33\n", grid_size, GRIDS_SIZE);

                for (int i = 0; i < grid_size; i++) {
                    // DEBUG_PRINTF("i=%d\n",i); //debug
                    for (int j = 0; j < 3; j++) {
                        ig_cuda_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
                        ig_cuda_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
                        ig_cuda_ptr->grids[i].m_dim_fl_minus_1[j]
                            = tmp_grids[i].m_dim_fl_minus_1[j];
                        ig_cuda_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
                    }
                    if (tmp_grids[i].m_data.dim0() != 0) {
                        ig_cuda_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0();
                        assert(MAX_NUM_OF_GRID_MI >= ig_cuda_ptr->grids[i].m_i);
                        ig_cuda_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1();
                        assert(MAX_NUM_OF_GRID_MJ >= ig_cuda_ptr->grids[i].m_j);
                        ig_cuda_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2();
                        assert(MAX_NUM_OF_GRID_MK >= ig_cuda_ptr->grids[i].m_k);

                        assert(tmp_grids[i].m_data.m_data.size()
                               == ig_cuda_ptr->grids[i].m_i * ig_cuda_ptr->grids[i].m_j
                                      * ig_cuda_ptr->grids[i].m_k);
                        assert(tmp_grids[i].m_data.m_data.size() <= MAX_NUM_OF_GRID_POINT);
                        memcpy(ig_cuda_ptr->grids[i].m_data, tmp_grids[i].m_data.m_data.data(),
                               tmp_grids[i].m_data.m_data.size() * sizeof(fl));
                    } else {
                        ig_cuda_ptr->grids[i].m_i = 0;
                        ig_cuda_ptr->grids[i].m_j = 0;
                        ig_cuda_ptr->grids[i].m_k = 0;
                    }
                }
            } else {
                ad4cache ig_tmp(ig.get_slope());
                ig_tmp.m_grids = ig.get_grids();
                // // debug
                // if (l == 1){
                // 	std::cout << "writing original grid map\n";
                // 	ig_tmp.write(std::string("./ori"), szv(1,0));
                // }
                ig_tmp.set_bias(bias_batch_list[l]);
                // // debug
                // std::cout << "writing bias\n";
                // ig_tmp.write(std::string("./")+std::to_string(l), szv(1,0));
                ig_cuda_ptr->atu = ig.get_atu();  // atu
                DEBUG_PRINTF("ig_cuda_ptr->atu=%d\n", ig_cuda_ptr->atu);
                ig_cuda_ptr->slope = ig.get_slope();  // slope
                std::vector<grid> tmp_grids = ig.get_grids();
                int grid_size = tmp_grids.size();
                DEBUG_PRINTF("ig.size()=%d, GRIDS_SIZE=%d, should be 33\n", grid_size, GRIDS_SIZE);

                for (int i = 0; i < grid_size; i++) {
                    // DEBUG_PRINTF("i=%d\n",i); //debug
                    for (int j = 0; j < 3; j++) {
                        ig_cuda_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
                        ig_cuda_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
                        ig_cuda_ptr->grids[i].m_dim_fl_minus_1[j]
                            = tmp_grids[i].m_dim_fl_minus_1[j];
                        ig_cuda_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
                    }
                    if (tmp_grids[i].m_data.dim0() != 0) {
                        ig_cuda_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0();
                        assert(MAX_NUM_OF_GRID_MI >= ig_cuda_ptr->grids[i].m_i);
                        ig_cuda_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1();
                        assert(MAX_NUM_OF_GRID_MJ >= ig_cuda_ptr->grids[i].m_j);
                        ig_cuda_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2();
                        assert(MAX_NUM_OF_GRID_MK >= ig_cuda_ptr->grids[i].m_k);

                        assert(tmp_grids[i].m_data.m_data.size()
                               == ig_cuda_ptr->grids[i].m_i * ig_cuda_ptr->grids[i].m_j
                                      * ig_cuda_ptr->grids[i].m_k);
                        memcpy(ig_cuda_ptr->grids[i].m_data, tmp_grids[i].m_data.m_data.data(),
                               tmp_grids[i].m_data.m_data.size() * sizeof(fl));
                    } else {
                        ig_cuda_ptr->grids[i].m_i = 0;
                        ig_cuda_ptr->grids[i].m_j = 0;
                        ig_cuda_ptr->grids[i].m_k = 0;
                    }
                }
            }

            checkCUDA(
                cudaMemcpyAsync(ig_cuda_gpu + l, ig_cuda_ptr, ig_cuda_size, cudaMemcpyHostToDevice, curr_stream));
        }
        std::cout << "set\n";
    } else {
        ig_cuda_ptr->atu = ig.get_atu();  // atu
        DEBUG_PRINTF("ig_cuda_ptr->atu=%d\n", ig_cuda_ptr->atu);
        ig_cuda_ptr->slope = ig.get_slope();  // slope
        std::vector<grid> tmp_grids = ig.get_grids();
        int grid_size = tmp_grids.size();
        DEBUG_PRINTF("ig.size()=%d, GRIDS_SIZE=%d, should be 33\n", grid_size, GRIDS_SIZE);

        for (int i = 0; i < grid_size; i++) {
            // DEBUG_PRINTF("i=%d\n",i); //debug
            for (int j = 0; j < 3; j++) {
                ig_cuda_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
                ig_cuda_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
                ig_cuda_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grids[i].m_dim_fl_minus_1[j];
                ig_cuda_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
            }
            if (tmp_grids[i].m_data.dim0() != 0) {
                ig_cuda_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0();
                assert(MAX_NUM_OF_GRID_MI >= ig_cuda_ptr->grids[i].m_i);
                ig_cuda_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1();
                assert(MAX_NUM_OF_GRID_MJ >= ig_cuda_ptr->grids[i].m_j);
                ig_cuda_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2();
                assert(MAX_NUM_OF_GRID_MK >= ig_cuda_ptr->grids[i].m_k);

                assert(tmp_grids[i].m_data.m_data.size()
                       == ig_cuda_ptr->grids[i].m_i * ig_cuda_ptr->grids[i].m_j
                              * ig_cuda_ptr->grids[i].m_k);
                memcpy(ig_cuda_ptr->grids[i].m_data, tmp_grids[i].m_data.m_data.data(),
                       tmp_grids[i].m_data.m_data.size() * sizeof(fl));
            } else {
                ig_cuda_ptr->grids[i].m_i = 0;
                ig_cuda_ptr->grids[i].m_j = 0;
                ig_cuda_ptr->grids[i].m_k = 0;
            }
        }
        DEBUG_PRINTF("memcpy ig_cuda, ig_cuda_size=%lu\n", ig_cuda_size);
        checkCUDA(cudaMalloc(&ig_cuda_gpu, ig_cuda_size));
        checkCUDA(cudaMemcpyAsync(ig_cuda_gpu, ig_cuda_ptr, ig_cuda_size, cudaMemcpyHostToDevice, curr_stream));
    }

    float mutation_amplitude_float = static_cast<float>(mutation_amplitude);

    checkCUDA(cudaMemcpyAsync(hunt_cap_gpu, hunt_cap_float, 3 * sizeof(float), cudaMemcpyHostToDevice, curr_stream));

    checkCUDA(cudaMemcpyAsync(authentic_v_gpu, authentic_v_float, sizeof(authentic_v_float),
                         cudaMemcpyHostToDevice, curr_stream));

    /* Add timing */
    cudaEvent_t start, stop;
    checkCUDA(cudaEventCreate(&start));
    checkCUDA(cudaEventCreate(&stop));
    checkCUDA(cudaEventRecord(start, curr_stream));

    /* Launch kernel */
    DEBUG_PRINTF("launch kernel, global_steps=%d, thread=%d, num_of_ligands=%d\n", global_steps,
                 thread, num_of_ligands);

    output_type_cuda_t *results_aux;
    checkCUDA(cudaMalloc(&results_aux, 5 * thread * sizeof(output_type_cuda_t)));
    checkCUDA(cudaMemsetAsync(results_aux, 0, 5 * thread * sizeof(output_type_cuda_t), curr_stream));
    change_cuda_t *change_aux;
    checkCUDA(cudaMalloc(&change_aux, 6 * thread * sizeof(change_cuda_t)));
    checkCUDA(cudaMemsetAsync(change_aux, 0, 6 * thread * sizeof(change_cuda_t), curr_stream));
    pot_cuda_t *pot_aux;
    checkCUDA(cudaMalloc(&pot_aux, thread * sizeof(pot_cuda_t)));
    checkCUDA(cudaMemsetAsync(pot_aux, 0, thread * sizeof(pot_cuda_t), curr_stream));

    kernel<32><<<thread, 32, 0, curr_stream>>>(m_cuda_gpu, ig_cuda_gpu, p_cuda_gpu, rand_molec_struc_gpu,
                               quasi_newton_par_max_steps, mutation_amplitude_float, states, seed,
                               epsilon_fl_float, hunt_cap_gpu, authentic_v_gpu, results_gpu,
                               results_aux, change_aux, pot_aux, h_cuda_global, m_cuda_global,
                               global_steps, num_of_ligands, threads_per_ligand, multi_bias);


    // Wait for stream operations to complete
    checkCUDA(cudaStreamSynchronize(curr_stream));

    // Device to Host memcpy of precalculated_byatom, copy back data to p_gpu
    p_m_data_cuda_t *p_data;
    checkCUDA(cudaMallocHost(&p_data, sizeof(p_m_data_cuda_t) * MAX_P_DATA_M_DATA_SIZE));
    memset(p_data, 0, sizeof(p_m_data_cuda_t) * MAX_P_DATA_M_DATA_SIZE);
    output_type_cuda_t *results;
    checkCUDA(cudaMallocHost(&results, thread * sizeof(output_type_cuda_t)));
    memset(results, 0, thread * sizeof(output_type_cuda_t));

    for (int l = 0; l < num_of_ligands; ++l) {
        // copy data to m_data on CPU, then to p_gpu[l]
        int pnum = p_gpu[l].m_data.m_data.size();
        checkCUDA(cudaMemcpy(p_data, m_data_list_gpu[l].p_data, sizeof(p_m_data_cuda_t) * pnum,
                             cudaMemcpyDeviceToHost));
        checkCUDA(cudaFree(m_data_list_gpu[l].p_data));  // free m_cuda pointers in p_cuda
        for (int i = 0; i < pnum; ++i) {
            memcpy(&p_gpu[l].m_data.m_data[i].fast[0], p_data[i].fast, sizeof(p_data[i].fast));
            memcpy(&p_gpu[l].m_data.m_data[i].smooth[0], p_data[i].smooth,
                   sizeof(p_data[i].smooth));
        }
    }
    // DEBUG_PRINTF("energies about the first ligand on GPU:\n");
    // for (int i = 0;i < 20; ++i){
    //     DEBUG_PRINTF("precalculated_byatom.m_data.m_data[%d]: (smooth.first,
    //     smooth.second, fast) ", i); for (int j = 0;j < FAST_SIZE; ++j){
    //         DEBUG_PRINTF("(%f, %f, %f) ",
    //         p_gpu[0].m_data.m_data[i].smooth[j].first,
    //         p_gpu[0].m_data.m_data[i].smooth[j].second,
    //         p_gpu[0].m_data.m_data[i].fast[j]);
    //     }
    //     DEBUG_PRINTF("\n");
    // }

    /* Timing output */

    checkCUDA(cudaEventRecord(stop, curr_stream));
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    DEBUG_PRINTF("Time spend on GPU is %f ms\n", msecTotal);

    /* Convert result data. Can be improved by mapping memory
     */
    DEBUG_PRINTF("cuda to vina\n");

    checkCUDA(cudaMemcpy(results, results_gpu, thread * sizeof(output_type_cuda_t),
                         cudaMemcpyDeviceToHost));

    std::vector<output_type> result_vina = cuda_to_vina(results, thread);

    DEBUG_PRINTF("result size=%lu\n", result_vina.size());

    for (int i = 0; i < thread; ++i) {
        add_to_output_container(out_gpu[i / threads_per_ligand], result_vina[i], min_rmsd,
                                num_saved_mins);
    }
    for (int i = 0; i < num_of_ligands; ++i) {
        DEBUG_PRINTF("output poses size = %lu\n", out_gpu[i].size());
        if (out_gpu[i].size() == 0) continue;
        DEBUG_PRINTF("output poses energy from gpu =");
        for (int j = 0; j < out_gpu[i].size(); ++j) DEBUG_PRINTF("%f ", out_gpu[i][j].e);
        DEBUG_PRINTF("\n");
    }

    /* Free memory */
    checkCUDA(cudaFree(m_cuda_gpu));
    checkCUDA(cudaFree(ig_cuda_gpu));
    checkCUDA(cudaFree(p_cuda_gpu));
    checkCUDA(cudaFree(rand_molec_struc_gpu));
    checkCUDA(cudaFree(hunt_cap_gpu));
    checkCUDA(cudaFree(authentic_v_gpu));
    checkCUDA(cudaFree(results_gpu));
    checkCUDA(cudaFree(change_aux));
    checkCUDA(cudaFree(results_aux));
    checkCUDA(cudaFree(pot_aux));
    checkCUDA(cudaFree(states));
    checkCUDA(cudaFree(h_cuda_global));
    checkCUDA(cudaFree(m_cuda_global));
    checkCUDA(cudaFreeHost(m_cuda));
    checkCUDA(cudaFreeHost(rand_molec_struc_tmp));
    checkCUDA(cudaFreeHost(ig_cuda_ptr));
    checkCUDA(cudaFreeHost(p_cuda));
    checkCUDA(cudaFreeHost(p_data));
    checkCUDA(cudaFreeHost(results));

    checkCUDA(cudaEventDestroy(start));
    checkCUDA(cudaEventDestroy(stop));
    checkCUDA(cudaStreamDestroy(curr_stream));
    curr_stream = 0;

    DEBUG_PRINTF("exit monte_carlo\n");       

}

/* Below based on monte-carlo.cpp */

// #ifdef ENABLE_CUDA

std::vector<output_type> monte_carlo::cuda_to_vina(output_type_cuda_t results_ptr[],
                                                   int thread) const {
    // DEBUG_PRINTF("entering cuda_to_vina\n");
    std::vector<output_type> results_vina;
    for (int i = 0; i < thread; ++i) {
        output_type_cuda_t results = results_ptr[i];
        conf tmp_c;
        tmp_c.ligands.resize(1);
        // Position
        for (int j = 0; j < 3; j++) tmp_c.ligands[0].rigid.position[j] = results.position[j];
        // Orientation
        qt q(results.orientation[0], results.orientation[1], results.orientation[2],
             results.orientation[3]);
        tmp_c.ligands[0].rigid.orientation = q;
        output_type tmp_vina(tmp_c, results.e);
        // torsion
        for (int j = 0; j < results.lig_torsion_size; j++)
            tmp_vina.c.ligands[0].torsions.push_back(results.lig_torsion[j]);
        // coords
        for (int j = 0; j < MAX_NUM_OF_ATOMS; j++) {
            vec v_tmp(results.coords[j][0], results.coords[j][1], results.coords[j][2]);
            if (v_tmp[0] * v_tmp[1] * v_tmp[2] != 0) tmp_vina.coords.push_back(v_tmp);
        }
        results_vina.push_back(tmp_vina);
    }
    return results_vina;
}

__host__ void monte_carlo::operator()(
    std::vector<model> &m_gpu, std::vector<output_container> &out_gpu,
    std::vector<precalculate_byatom> &p_gpu, triangular_matrix_cuda_t *m_data_list_gpu,
    const igrid &ig, const vec &corner1, const vec &corner2, rng &generator, int verbosity,
    unsigned long long seed, std::vector<std::vector<bias_element>> &bias_batch_list) const {
    /* Definitions from vina1.2 */
    DEBUG_PRINTF("entering CUDA monte_carlo search\n");  // debug

    vec authentic_v(1000, 1000,
                    1000);  // FIXME? this is here to avoid max_fl/max_fl

    quasi_newton quasi_newton_par;
    const int quasi_newton_par_max_steps = local_steps;  // no need to decrease step

    /* Allocate CPU memory and define new data structure */
    DEBUG_PRINTF("Allocating CPU memory\n");  // debug
    m_cuda_t *m_cuda;
    checkCUDA(cudaMallocHost(&m_cuda, sizeof(m_cuda_t)));
    memset(m_cuda, 0, sizeof(m_cuda_t));

    output_type_cuda_t *rand_molec_struc_tmp;
    checkCUDA(cudaMallocHost(&rand_molec_struc_tmp, sizeof(output_type_cuda_t)));
    memset(rand_molec_struc_tmp, 0, sizeof(output_type_cuda_t));

    ig_cuda_t *ig_cuda_ptr;
    checkCUDA(cudaMallocHost(&ig_cuda_ptr, sizeof(ig_cuda_t)));
    memset(ig_cuda_ptr, 0, sizeof(ig_cuda_t));

    p_cuda_t_cpu *p_cuda;
    checkCUDA(cudaMallocHost(&p_cuda, sizeof(p_cuda_t_cpu)));
    memset(p_cuda, 0, sizeof(p_cuda_t_cpu));

    /* End CPU allocation */

    /* Allocate GPU memory */
    DEBUG_PRINTF("Allocating GPU memory\n");
    size_t m_cuda_size = sizeof(m_cuda_t);
    DEBUG_PRINTF("m_cuda_size=%lu\n", m_cuda_size);
    size_t ig_cuda_size = sizeof(ig_cuda_t);
    DEBUG_PRINTF("ig_cuda_size=%lu\n", ig_cuda_size);
    DEBUG_PRINTF("p_cuda_size_cpu=%lu\n", sizeof(p_cuda_t_cpu));

    size_t p_cuda_size_gpu = sizeof(p_cuda_t);
    DEBUG_PRINTF("p_cuda_size_gpu=%lu\n", p_cuda_size_gpu);

    // rand_molec_struc_gpu
    float *rand_molec_struc_gpu;
    checkCUDA(cudaMalloc(&rand_molec_struc_gpu, thread * SIZE_OF_MOLEC_STRUC));
    checkCUDA(cudaMemset(rand_molec_struc_gpu, 0, thread * SIZE_OF_MOLEC_STRUC));

    float epsilon_fl_float = static_cast<float>(epsilon_fl);

    // use cuRand to generate random values on GPU
    curandStatePhilox4_32_10_t *states;
    DEBUG_PRINTF("random states size=%lu\n", sizeof(curandStatePhilox4_32_10_t) * thread);
    checkCUDA(cudaMalloc(&states, sizeof(curandStatePhilox4_32_10_t) * thread));
    checkCUDA(cudaMemset(states, 0, sizeof(curandStatePhilox4_32_10_t) * thread));

    // hunt_cap_gpu
    float *hunt_cap_gpu;
    float hunt_cap_float[3] = {static_cast<float>(hunt_cap[0]), static_cast<float>(hunt_cap[1]),
                               static_cast<float>(hunt_cap[2])};

    checkCUDA(cudaMalloc(&hunt_cap_gpu, 3 * sizeof(float)));
    checkCUDA(cudaMemset(hunt_cap_gpu, 0, 3 * sizeof(float)));
    // Preparing m related data
    m_cuda_t *m_cuda_gpu;
    DEBUG_PRINTF("m_cuda_size=%lu", m_cuda_size);
    checkCUDA(cudaMalloc(&m_cuda_gpu, num_of_ligands * m_cuda_size));
    checkCUDA(cudaMemset(m_cuda_gpu, 0, num_of_ligands * m_cuda_size));
    // Preparing p related data

    p_cuda_t *p_cuda_gpu;
    checkCUDA(cudaMalloc(&p_cuda_gpu, num_of_ligands * p_cuda_size_gpu));
    checkCUDA(cudaMemset(p_cuda_gpu, 0, num_of_ligands * p_cuda_size_gpu));
    DEBUG_PRINTF("p_cuda_gpu=%p\n", p_cuda_gpu);
    // Preparing ig related data (cache related data)
    ig_cuda_t *ig_cuda_gpu;

    float *authentic_v_gpu;
    float authentic_v_float[3]
        = {static_cast<float>(authentic_v[0]), static_cast<float>(authentic_v[1]),
           static_cast<float>(authentic_v[2])};

    checkCUDA(cudaMalloc(&authentic_v_gpu, sizeof(authentic_v_float)));
    checkCUDA(cudaMemset(authentic_v_gpu, 0, sizeof(authentic_v_float)));
    // Preparing result data
    output_type_cuda_t *results_gpu;
    checkCUDA(cudaMalloc(&results_gpu, thread * sizeof(output_type_cuda_t)));
    checkCUDA(cudaMemset(results_gpu, 0, thread * sizeof(output_type_cuda_t)));

    m_cuda_t *m_cuda_global;
    checkCUDA(cudaMalloc(&m_cuda_global, thread * sizeof(m_cuda_t)));
    checkCUDA(cudaMemset(m_cuda_global, 0, thread * sizeof(m_cuda_t)));

    matrix_d *h_cuda_global;
    checkCUDA(cudaMalloc(&h_cuda_global, thread * sizeof(matrix_d)));
    checkCUDA(cudaMemset(h_cuda_global, 0, thread * sizeof(matrix_d)));

    /* End Allocating GPU Memory */

    assert(num_of_ligands <= MAX_LIGAND_NUM);
    assert(thread <= MAX_THREAD);

    struct tmp_struct {
        int start_index = 0;
        int parent_index = 0;
        void store_node(tree<segment> &child_ptr, rigid_cuda_t &rigid) {
            start_index++;  // start with index 1, index 0 is root node
            rigid.parent[start_index] = parent_index;
            rigid.atom_range[start_index][0] = child_ptr.node.begin;
            rigid.atom_range[start_index][1] = child_ptr.node.end;
            for (int i = 0; i < 9; i++)
                rigid.orientation_m[start_index][i] = child_ptr.node.get_orientation_m().data[i];
            rigid.orientation_q[start_index][0] = child_ptr.node.orientation().R_component_1();
            rigid.orientation_q[start_index][1] = child_ptr.node.orientation().R_component_2();
            rigid.orientation_q[start_index][2] = child_ptr.node.orientation().R_component_3();
            rigid.orientation_q[start_index][3] = child_ptr.node.orientation().R_component_4();
            for (int i = 0; i < 3; i++) {
                rigid.origin[start_index][i] = child_ptr.node.get_origin()[i];
                rigid.axis[start_index][i] = child_ptr.node.get_axis()[i];
                rigid.relative_axis[start_index][i] = child_ptr.node.relative_axis[i];
                rigid.relative_origin[start_index][i] = child_ptr.node.relative_origin[i];
            }
            if (child_ptr.children.size() == 0)
                return;
            else {
                assert(start_index < MAX_NUM_OF_RIGID);
                int parent_index_tmp = start_index;
                for (int i = 0; i < child_ptr.children.size(); i++) {
                    this->parent_index = parent_index_tmp;  // Update parent index
                    this->store_node(child_ptr.children[i], rigid);
                }
            }
        }
    };

    for (int l = 0; l < num_of_ligands; ++l) {
        model &m = m_gpu[l];
        const precalculate_byatom &p = p_gpu[l];

        /* Prepare m related data */
        conf_size s = m.get_size();
        change g(s);
        output_type tmp(s, 0);
        tmp.c = m.get_initial_conf();

        assert(m.atoms.size() < MAX_NUM_OF_ATOMS);

        // Preparing ligand data
        DEBUG_PRINTF("prepare ligand data\n");
        assert(m.num_other_pairs() == 0);  // m.other_pairs is not supported!
        assert(m.ligands.size() <= 1);     // Only one ligand supported!

        if (m.ligands.size() == 0) {  // ligand parsing error
            m_cuda->m_num_movable_atoms = -1;
            DEBUG_PRINTF("copy m_cuda to gpu, size=%lu\n", sizeof(m_cuda_t));
            checkCUDA(cudaMemcpy(m_cuda_gpu + l, m_cuda, sizeof(m_cuda_t), cudaMemcpyHostToDevice));
        } else {
            for (int i = 0; i < m.atoms.size(); i++) {
                m_cuda->atoms[i].types[0]
                    = m.atoms[i].el;  // To store 4 atoms types (el, ad, xs, sy)
                m_cuda->atoms[i].types[1] = m.atoms[i].ad;
                m_cuda->atoms[i].types[2] = m.atoms[i].xs;
                m_cuda->atoms[i].types[3] = m.atoms[i].sy;
                for (int j = 0; j < 3; j++) {
                    m_cuda->atoms[i].coords[j] = m.atoms[i].coords[j];  // To store atom coords
                }
            }

            // To store atoms coords
            for (int i = 0; i < m.coords.size(); i++) {
                for (int j = 0; j < 3; j++) {
                    m_cuda->m_coords.coords[i][j] = m.coords[i].data[j];
                }
            }

            // To store minus forces
            for (int i = 0; i < m.coords.size(); i++) {
                for (int j = 0; j < 3; j++) {
                    m_cuda->minus_forces.coords[i][j] = m.minus_forces[i].data[j];
                }
            }

            m_cuda->ligand.pairs.num_pairs = m.ligands[0].pairs.size();
            for (int i = 0; i < m_cuda->ligand.pairs.num_pairs; i++) {
                m_cuda->ligand.pairs.type_pair_index[i] = m.ligands[0].pairs[i].type_pair_index;
                m_cuda->ligand.pairs.a[i] = m.ligands[0].pairs[i].a;
                m_cuda->ligand.pairs.b[i] = m.ligands[0].pairs[i].b;
            }
            m_cuda->ligand.begin = m.ligands[0].begin;  // 0
            m_cuda->ligand.end = m.ligands[0].end;      // 29
            ligand &m_ligand = m.ligands[0];            // Only support one ligand
            DEBUG_PRINTF("m_ligand.end=%lu, MAX_NUM_OF_ATOMS=%d\n", m_ligand.end, MAX_NUM_OF_ATOMS);
            assert(m_ligand.end < MAX_NUM_OF_ATOMS);

            // Store root node
            m_cuda->ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
            m_cuda->ligand.rigid.atom_range[0][1] = m_ligand.node.end;
            for (int i = 0; i < 3; i++)
                m_cuda->ligand.rigid.origin[0][i] = m_ligand.node.get_origin()[i];
            for (int i = 0; i < 9; i++)
                m_cuda->ligand.rigid.orientation_m[0][i]
                    = m_ligand.node.get_orientation_m().data[i];
            m_cuda->ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
            m_cuda->ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
            m_cuda->ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
            m_cuda->ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
            for (int i = 0; i < 3; i++) {
                m_cuda->ligand.rigid.axis[0][i] = 0;
                m_cuda->ligand.rigid.relative_axis[0][i] = 0;
                m_cuda->ligand.rigid.relative_origin[0][i] = 0;
            }

            // Store children nodes (in depth-first order)
            DEBUG_PRINTF("store children nodes\n");

            tmp_struct ts;
            for (int i = 0; i < m_ligand.children.size(); i++) {
                ts.parent_index = 0;  // Start a new branch, whose parent is 0
                ts.store_node(m_ligand.children[i], m_cuda->ligand.rigid);
            }
            m_cuda->ligand.rigid.num_children = ts.start_index;

            // set children map
            DEBUG_PRINTF("set children map\n");
            for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
                for (int j = 0; j < MAX_NUM_OF_RIGID; j++) {
                    m_cuda->ligand.rigid.children_map[i][j] = false;
                    m_cuda->ligand.rigid.descendant_map[i][j] = false;
                }

            for (int i = MAX_NUM_OF_RIGID - 1; i >= 0; i--) {
                if (i > 0) {
                    m_cuda->ligand.rigid.children_map[m_cuda->ligand.rigid.parent[i]][i] = true;
                    m_cuda->ligand.rigid.descendant_map[m_cuda->ligand.rigid.parent[i]][i] = true;
                }
                for (int j = i + 1; j < MAX_NUM_OF_RIGID; j++) {
                    if (m_cuda->ligand.rigid.descendant_map[i][j])
                        m_cuda->ligand.rigid.descendant_map[m_cuda->ligand.rigid.parent[i]][j]
                            = true;
                }
            }
            m_cuda->m_num_movable_atoms = m.num_movable_atoms();

            DEBUG_PRINTF("copy m_cuda to gpu, size=%lu\n", sizeof(m_cuda_t));
            checkCUDA(cudaMemcpy(m_cuda_gpu + l, m_cuda, sizeof(m_cuda_t), cudaMemcpyHostToDevice));

            /* Prepare rand_molec_struc data */
            int lig_torsion_size = tmp.c.ligands[0].torsions.size();
            DEBUG_PRINTF("lig_torsion_size=%d\n", lig_torsion_size);
            int flex_torsion_size;
            if (tmp.c.flex.size() != 0)
                flex_torsion_size = tmp.c.flex[0].torsions.size();
            else
                flex_torsion_size = 0;
            // std::vector<vec> uniform_data;
            // uniform_data.resize(thread);

            for (int i = 0; i < threads_per_ligand; ++i) {
                if (!local_only) {
                    tmp.c.randomize(corner1, corner2,
                                    generator);  // generate a random structure,
                                                 // can move to GPU if necessary
                }
                for (int j = 0; j < 3; j++)
                    rand_molec_struc_tmp->position[j] = tmp.c.ligands[0].rigid.position[j];
                assert(lig_torsion_size <= MAX_NUM_OF_LIG_TORSION);
                for (int j = 0; j < lig_torsion_size; j++)
                    rand_molec_struc_tmp->lig_torsion[j]
                        = tmp.c.ligands[0].torsions[j];  // Only support one ligand
                assert(flex_torsion_size <= MAX_NUM_OF_FLEX_TORSION);
                for (int j = 0; j < flex_torsion_size; j++)
                    rand_molec_struc_tmp->flex_torsion[j]
                        = tmp.c.flex[0].torsions[j];  // Only support one flex

                rand_molec_struc_tmp->orientation[0]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_1();
                rand_molec_struc_tmp->orientation[1]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_2();
                rand_molec_struc_tmp->orientation[2]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_3();
                rand_molec_struc_tmp->orientation[3]
                    = (float)tmp.c.ligands[0].rigid.orientation.R_component_4();

                rand_molec_struc_tmp->lig_torsion_size = lig_torsion_size;

                float *rand_molec_struc_gpu_tmp
                    = rand_molec_struc_gpu
                      + (l * threads_per_ligand + i) * SIZE_OF_MOLEC_STRUC / sizeof(float);
                checkCUDA(cudaMemcpy(rand_molec_struc_gpu_tmp, rand_molec_struc_tmp,
                                     SIZE_OF_MOLEC_STRUC, cudaMemcpyHostToDevice));
            }

            /* Preparing p related data */
            DEBUG_PRINTF("Preaparing p related data\n");  // debug

            // copy pointer instead of data
            p_cuda->m_cutoff_sqr = p.m_cutoff_sqr;
            p_cuda->factor = p.m_factor;
            p_cuda->n = p.m_n;
            p_cuda->m_data_size = p.m_data.m_data.size();
            checkCUDA(cudaMemcpy(p_cuda_gpu + l, p_cuda, sizeof(p_cuda_t), cudaMemcpyHostToDevice));
            checkCUDA(cudaMemcpy(&(p_cuda_gpu[l].m_data), &(m_data_list_gpu[l].p_data),
                                 sizeof(p_m_data_cuda_t *),
                                 cudaMemcpyHostToDevice));  // check if fl == float
        }
    }

    /* Prepare data only concerns rigid receptor */

    // Preparing igrid related data
    DEBUG_PRINTF("Preparing ig related data\n");  // debug

    bool multi_bias = (bias_batch_list.size() == num_of_ligands);
    if (multi_bias) {
        // multi bias mode
        std::cout << "with multi bias ";

        checkCUDA(cudaMalloc(&ig_cuda_gpu, ig_cuda_size * num_of_ligands));
        checkCUDA(cudaMemset(&ig_cuda_gpu, 0, ig_cuda_size * num_of_ligands));
        for (int l = 0; l < num_of_ligands; ++l) {
            if (ig.get_atu() == atom_type::XS) {
                cache ig_tmp(ig.get_gd(), ig.get_slope());
                ig_tmp.m_grids = ig.get_grids();
                // // debug
                // if (l == 1){
                // 	std::cout << "writing original grid map\n";
                // 	ig_tmp.write(std::string("./ori"), szv(1,0));
                // }
                ig_tmp.compute_bias(m_gpu[l], bias_batch_list[l]);
                // // debug
                // std::cout << "writing bias\n";
                // ig_tmp.write(std::string("./")+std::to_string(l), szv(1,0));
                ig_cuda_ptr->atu = ig.get_atu();  // atu
                DEBUG_PRINTF("ig_cuda_ptr->atu=%d\n", ig_cuda_ptr->atu);
                ig_cuda_ptr->slope = ig.get_slope();  // slope
                std::vector<grid> tmp_grids = ig.get_grids();
                int grid_size = tmp_grids.size();
                DEBUG_PRINTF("ig.size()=%d, GRIDS_SIZE=%d, should be 33\n", grid_size, GRIDS_SIZE);

                for (int i = 0; i < grid_size; i++) {
                    // DEBUG_PRINTF("i=%d\n",i); //debug
                    for (int j = 0; j < 3; j++) {
                        ig_cuda_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
                        ig_cuda_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
                        ig_cuda_ptr->grids[i].m_dim_fl_minus_1[j]
                            = tmp_grids[i].m_dim_fl_minus_1[j];
                        ig_cuda_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
                    }
                    if (tmp_grids[i].m_data.dim0() != 0) {
                        ig_cuda_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0();
                        assert(MAX_NUM_OF_GRID_MI >= ig_cuda_ptr->grids[i].m_i);
                        ig_cuda_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1();
                        assert(MAX_NUM_OF_GRID_MJ >= ig_cuda_ptr->grids[i].m_j);
                        ig_cuda_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2();
                        assert(MAX_NUM_OF_GRID_MK >= ig_cuda_ptr->grids[i].m_k);

                        assert(tmp_grids[i].m_data.m_data.size()
                               == ig_cuda_ptr->grids[i].m_i * ig_cuda_ptr->grids[i].m_j
                                      * ig_cuda_ptr->grids[i].m_k);
                        assert(tmp_grids[i].m_data.m_data.size() <= MAX_NUM_OF_GRID_POINT);
                        memcpy(ig_cuda_ptr->grids[i].m_data, tmp_grids[i].m_data.m_data.data(),
                               tmp_grids[i].m_data.m_data.size() * sizeof(fl));
                    } else {
                        ig_cuda_ptr->grids[i].m_i = 0;
                        ig_cuda_ptr->grids[i].m_j = 0;
                        ig_cuda_ptr->grids[i].m_k = 0;
                    }
                }
            } else {
                ad4cache ig_tmp(ig.get_slope());
                ig_tmp.m_grids = ig.get_grids();
                // // debug
                // if (l == 1){
                // 	std::cout << "writing original grid map\n";
                // 	ig_tmp.write(std::string("./ori"), szv(1,0));
                // }
                ig_tmp.set_bias(bias_batch_list[l]);
                // // debug
                // std::cout << "writing bias\n";
                // ig_tmp.write(std::string("./")+std::to_string(l), szv(1,0));
                ig_cuda_ptr->atu = ig.get_atu();  // atu
                DEBUG_PRINTF("ig_cuda_ptr->atu=%d\n", ig_cuda_ptr->atu);
                ig_cuda_ptr->slope = ig.get_slope();  // slope
                std::vector<grid> tmp_grids = ig.get_grids();
                int grid_size = tmp_grids.size();
                DEBUG_PRINTF("ig.size()=%d, GRIDS_SIZE=%d, should be 33\n", grid_size, GRIDS_SIZE);

                for (int i = 0; i < grid_size; i++) {
                    // DEBUG_PRINTF("i=%d\n",i); //debug
                    for (int j = 0; j < 3; j++) {
                        ig_cuda_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
                        ig_cuda_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
                        ig_cuda_ptr->grids[i].m_dim_fl_minus_1[j]
                            = tmp_grids[i].m_dim_fl_minus_1[j];
                        ig_cuda_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
                    }
                    if (tmp_grids[i].m_data.dim0() != 0) {
                        ig_cuda_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0();
                        assert(MAX_NUM_OF_GRID_MI >= ig_cuda_ptr->grids[i].m_i);
                        ig_cuda_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1();
                        assert(MAX_NUM_OF_GRID_MJ >= ig_cuda_ptr->grids[i].m_j);
                        ig_cuda_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2();
                        assert(MAX_NUM_OF_GRID_MK >= ig_cuda_ptr->grids[i].m_k);

                        assert(tmp_grids[i].m_data.m_data.size()
                               == ig_cuda_ptr->grids[i].m_i * ig_cuda_ptr->grids[i].m_j
                                      * ig_cuda_ptr->grids[i].m_k);
                        memcpy(ig_cuda_ptr->grids[i].m_data, tmp_grids[i].m_data.m_data.data(),
                               tmp_grids[i].m_data.m_data.size() * sizeof(fl));
                    } else {
                        ig_cuda_ptr->grids[i].m_i = 0;
                        ig_cuda_ptr->grids[i].m_j = 0;
                        ig_cuda_ptr->grids[i].m_k = 0;
                    }
                }
            }

            checkCUDA(
                cudaMemcpy(ig_cuda_gpu + l, ig_cuda_ptr, ig_cuda_size, cudaMemcpyHostToDevice));
        }
        std::cout << "set\n";
    } else {
        ig_cuda_ptr->atu = ig.get_atu();  // atu
        DEBUG_PRINTF("ig_cuda_ptr->atu=%d\n", ig_cuda_ptr->atu);
        ig_cuda_ptr->slope = ig.get_slope();  // slope
        std::vector<grid> tmp_grids = ig.get_grids();
        int grid_size = tmp_grids.size();
        DEBUG_PRINTF("ig.size()=%d, GRIDS_SIZE=%d, should be 33\n", grid_size, GRIDS_SIZE);

        for (int i = 0; i < grid_size; i++) {
            // DEBUG_PRINTF("i=%d\n",i); //debug
            for (int j = 0; j < 3; j++) {
                ig_cuda_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
                ig_cuda_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
                ig_cuda_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grids[i].m_dim_fl_minus_1[j];
                ig_cuda_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
            }
            if (tmp_grids[i].m_data.dim0() != 0) {
                ig_cuda_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0();
                assert(MAX_NUM_OF_GRID_MI >= ig_cuda_ptr->grids[i].m_i);
                ig_cuda_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1();
                assert(MAX_NUM_OF_GRID_MJ >= ig_cuda_ptr->grids[i].m_j);
                ig_cuda_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2();
                assert(MAX_NUM_OF_GRID_MK >= ig_cuda_ptr->grids[i].m_k);

                assert(tmp_grids[i].m_data.m_data.size()
                       == ig_cuda_ptr->grids[i].m_i * ig_cuda_ptr->grids[i].m_j
                              * ig_cuda_ptr->grids[i].m_k);
                memcpy(ig_cuda_ptr->grids[i].m_data, tmp_grids[i].m_data.m_data.data(),
                       tmp_grids[i].m_data.m_data.size() * sizeof(fl));
            } else {
                ig_cuda_ptr->grids[i].m_i = 0;
                ig_cuda_ptr->grids[i].m_j = 0;
                ig_cuda_ptr->grids[i].m_k = 0;
            }
        }
        DEBUG_PRINTF("memcpy ig_cuda, ig_cuda_size=%lu\n", ig_cuda_size);
        checkCUDA(cudaMalloc(&ig_cuda_gpu, ig_cuda_size));
        checkCUDA(cudaMemcpy(ig_cuda_gpu, ig_cuda_ptr, ig_cuda_size, cudaMemcpyHostToDevice));
    }

    float mutation_amplitude_float = static_cast<float>(mutation_amplitude);

    checkCUDA(cudaMemcpy(hunt_cap_gpu, hunt_cap_float, 3 * sizeof(float), cudaMemcpyHostToDevice));
    float hunt_test[3];
    checkCUDA(cudaMemcpy(hunt_test, hunt_cap_gpu, 3 * sizeof(float), cudaMemcpyDeviceToHost));
    DEBUG_PRINTF("hunt_test[1]=%f, hunt_cap_float[1]=%f\n", hunt_test[1], hunt_cap_float[1]);
    checkCUDA(cudaMemcpy(authentic_v_gpu, authentic_v_float, sizeof(authentic_v_float),
                         cudaMemcpyHostToDevice));

    /* Add timing */
    cudaEvent_t start, stop;
    checkCUDA(cudaEventCreate(&start));
    checkCUDA(cudaEventCreate(&stop));
    checkCUDA(cudaEventRecord(start, NULL));

    /* Launch kernel */
    DEBUG_PRINTF("launch kernel, global_steps=%d, thread=%d, num_of_ligands=%d\n", global_steps,
                 thread, num_of_ligands);

    output_type_cuda_t *results_aux;
    checkCUDA(cudaMalloc(&results_aux, 5 * thread * sizeof(output_type_cuda_t)));
    checkCUDA(cudaMemset(results_aux, 0, 5 * thread * sizeof(output_type_cuda_t)));
    change_cuda_t *change_aux;
    checkCUDA(cudaMalloc(&change_aux, 6 * thread * sizeof(change_cuda_t)));
    checkCUDA(cudaMemset(change_aux, 0, 6 * thread * sizeof(change_cuda_t)));
    pot_cuda_t *pot_aux;
    checkCUDA(cudaMalloc(&pot_aux, thread * sizeof(pot_cuda_t)));
    checkCUDA(cudaMemset(pot_aux, 0, thread * sizeof(pot_cuda_t)));

    kernel<32><<<thread, 32>>>(m_cuda_gpu, ig_cuda_gpu, p_cuda_gpu, rand_molec_struc_gpu,
                               quasi_newton_par_max_steps, mutation_amplitude_float, states, seed,
                               epsilon_fl_float, hunt_cap_gpu, authentic_v_gpu, results_gpu,
                               results_aux, change_aux, pot_aux, h_cuda_global, m_cuda_global,
                               global_steps, num_of_ligands, threads_per_ligand, multi_bias);

    // Device to Host memcpy of precalculated_byatom, copy back data to p_gpu
    p_m_data_cuda_t *p_data;
    checkCUDA(cudaMallocHost(&p_data, sizeof(p_m_data_cuda_t) * MAX_P_DATA_M_DATA_SIZE));
    memset(p_data, 0, sizeof(p_m_data_cuda_t) * MAX_P_DATA_M_DATA_SIZE);
    output_type_cuda_t *results;
    checkCUDA(cudaMallocHost(&results, thread * sizeof(output_type_cuda_t)));
    memset(results, 0, thread * sizeof(output_type_cuda_t));

    for (int l = 0; l < num_of_ligands; ++l) {
        // copy data to m_data on CPU, then to p_gpu[l]
        int pnum = p_gpu[l].m_data.m_data.size();
        checkCUDA(cudaMemcpy(p_data, m_data_list_gpu[l].p_data, sizeof(p_m_data_cuda_t) * pnum,
                             cudaMemcpyDeviceToHost));
        checkCUDA(cudaFree(m_data_list_gpu[l].p_data));  // free m_cuda pointers in p_cuda
        for (int i = 0; i < pnum; ++i) {
            memcpy(&p_gpu[l].m_data.m_data[i].fast[0], p_data[i].fast, sizeof(p_data[i].fast));
            memcpy(&p_gpu[l].m_data.m_data[i].smooth[0], p_data[i].smooth,
                   sizeof(p_data[i].smooth));
        }
    }
    // DEBUG_PRINTF("energies about the first ligand on GPU:\n");
    // for (int i = 0;i < 20; ++i){
    //     DEBUG_PRINTF("precalculated_byatom.m_data.m_data[%d]: (smooth.first,
    //     smooth.second, fast) ", i); for (int j = 0;j < FAST_SIZE; ++j){
    //         DEBUG_PRINTF("(%f, %f, %f) ",
    //         p_gpu[0].m_data.m_data[i].smooth[j].first,
    //         p_gpu[0].m_data.m_data[i].smooth[j].second,
    //         p_gpu[0].m_data.m_data[i].fast[j]);
    //     }
    //     DEBUG_PRINTF("\n");
    // }

    checkCUDA(cudaDeviceSynchronize());
    /* Timing output */

    checkCUDA(cudaEventRecord(stop, NULL));
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    DEBUG_PRINTF("Time spend on GPU is %f ms\n", msecTotal);

    /* Convert result data. Can be improved by mapping memory
     */
    DEBUG_PRINTF("cuda to vina\n");

    checkCUDA(cudaMemcpy(results, results_gpu, thread * sizeof(output_type_cuda_t),
                         cudaMemcpyDeviceToHost));

    std::vector<output_type> result_vina = cuda_to_vina(results, thread);

    DEBUG_PRINTF("result size=%lu\n", result_vina.size());

    for (int i = 0; i < thread; ++i) {
        add_to_output_container(out_gpu[i / threads_per_ligand], result_vina[i], min_rmsd,
                                num_saved_mins);
    }
    for (int i = 0; i < num_of_ligands; ++i) {
        DEBUG_PRINTF("output poses size = %lu\n", out_gpu[i].size());
        if (out_gpu[i].size() == 0) continue;
        DEBUG_PRINTF("output poses energy from gpu =");
        for (int j = 0; j < out_gpu[i].size(); ++j) DEBUG_PRINTF("%f ", out_gpu[i][j].e);
        DEBUG_PRINTF("\n");
    }

    /* Free memory */
    checkCUDA(cudaFree(m_cuda_gpu));
    checkCUDA(cudaFree(ig_cuda_gpu));
    checkCUDA(cudaFree(p_cuda_gpu));
    checkCUDA(cudaFree(rand_molec_struc_gpu));
    checkCUDA(cudaFree(hunt_cap_gpu));
    checkCUDA(cudaFree(authentic_v_gpu));
    checkCUDA(cudaFree(results_gpu));
    checkCUDA(cudaFree(change_aux));
    checkCUDA(cudaFree(results_aux));
    checkCUDA(cudaFree(pot_aux));
    checkCUDA(cudaFree(states));
    checkCUDA(cudaFree(h_cuda_global));
    checkCUDA(cudaFree(m_cuda_global));
    checkCUDA(cudaFreeHost(m_cuda));
    checkCUDA(cudaFreeHost(rand_molec_struc_tmp));
    checkCUDA(cudaFreeHost(ig_cuda_ptr));
    checkCUDA(cudaFreeHost(p_cuda));
    checkCUDA(cudaFreeHost(p_data));
    checkCUDA(cudaFreeHost(results));

    DEBUG_PRINTF("exit monte_carlo\n");
}

bool metropolis_accept(fl old_f, fl new_f, fl temperature, rng &generator) {
    if (new_f < old_f) return true;
    const fl acceptance_probability = std::exp((old_f - new_f) / temperature);
    return random_fl(0, 1, generator) < acceptance_probability;
}

__host__ void monte_carlo::operator()(model &m, output_container &out, const precalculate_byatom &p,
                                      const igrid &ig, const vec &corner1, const vec &corner2,
                                      rng &generator) const {
    int evalcount = 0;
    vec authentic_v(1000, 1000,
                    1000);  // FIXME? this is here to avoid max_fl/max_fl
    conf_size s = m.get_size();
    change g(s);
    output_type tmp(s, 0);
    tmp.c.randomize(corner1, corner2, generator);
    fl best_e = max_fl;
    quasi_newton quasi_newton_par;
    quasi_newton_par.max_steps = local_steps;
    VINA_U_FOR(step, global_steps) {
        // if(increment_me)
        // 	++(*increment_me);
        if ((max_evals > 0) & (evalcount > max_evals)) break;
        output_type candidate = tmp;
        mutate_conf(candidate.c, m, mutation_amplitude, generator);
        quasi_newton_par(m, p, ig, candidate, g, hunt_cap, evalcount);
        if (step == 0 || metropolis_accept(tmp.e, candidate.e, temperature, generator)) {
            tmp = candidate;

            m.set(tmp.c);  // FIXME? useless?

            // FIXME only for very promising ones
            if (tmp.e < best_e || out.size() < num_saved_mins) {
                quasi_newton_par(m, p, ig, tmp, g, authentic_v, evalcount);
                m.set(tmp.c);  // FIXME? useless?
                tmp.coords = m.get_heavy_atom_movable_coords();
                add_to_output_container(out, tmp, min_rmsd,
                                        num_saved_mins);  // 20 - max size
                if (tmp.e < best_e) best_e = tmp.e;
            }
        }
    }
    VINA_CHECK(!out.empty());
    VINA_CHECK(out.front().e <= out.back().e);  // make sure the sorting worked in the correct order
}

/* Above based on monte-carlo.cpp */

// #endif
