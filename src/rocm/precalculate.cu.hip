#include "hip/hip_runtime.h"

#include "kernel.h"
#include "math.h"
#include "model.h"
#include <vector>

#include "precalculate.h"
#include "precalculate_gpu.h"

// TODO: define kernel here
__global__
void precalculate_gpu(triangular_matrix_cuda_t *m_data_gpu_list, scoring_function_cuda_t *sf_gpu,
    sz *atom_xs_gpu, sz *atom_ad_gpu, fl *atom_charge_gpu, int *atom_num_gpu, fl factor, fl *common_rs_gpu, fl max_fl, int thread, int max_atom_num){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= thread){
        return;
    }
    // DEBUG_PRINTF("idx=%d\n", idx);
    // move to correct atom offset
    atom_xs_gpu += idx * max_atom_num;
    atom_ad_gpu += idx * max_atom_num;
    atom_charge_gpu += idx * max_atom_num;
    int atom_num = atom_num_gpu[idx];
    // DEBUG_PRINTF("atom_num=%d\n", atom_num);
    precalculate_element_cuda_t *p_data_gpu = m_data_gpu_list[idx].p_data;

    // // debug
    // for (int i = 0;i < atom_num;++i){
    //     DEBUG_PRINTF("atom[%d] on gpu: xs=%lu\n", i, atom_xs_gpu[i]);
    // }

    for (int i = 0;i < atom_num; ++i){
        for (int j = i;j < atom_num; ++j){
            int offset = i + j*(j+1)/2; // copied from "triangular_matrix_index.h"
            int n = SMOOTH_SIZE;
            p_data_gpu[offset].factor = 32.0;
            switch (sf_gpu->m_sf_choice){
                case SF_VINA:
                {
                    for (int k = 0;k < n; ++k){
                        fl sum = 0;
                        // calculate smooth_e
                        sum += sf_gpu->m_weights[0] * vina_gaussian_cuda_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vina_gaussian_cutoff_1, sf_gpu->vina_gaussian_offset_1, sf_gpu->vina_gaussian_width_1);
                        sum += sf_gpu->m_weights[1] * vina_gaussian_cuda_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vina_gaussian_cutoff_2, sf_gpu->vina_gaussian_offset_2, sf_gpu->vina_gaussian_width_2);
                        sum += sf_gpu->m_weights[2] * vina_repulsion_cuda_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vina_repulsion_cutoff, sf_gpu->vina_repulsion_offset);
                        sum += sf_gpu->m_weights[3] * vina_hydrophobic_cuda_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vina_hydrophobic_good, sf_gpu->vina_hydrophobic_bad, sf_gpu->vina_hydrophobic_cutoff);
                        sum += sf_gpu->m_weights[4] * vina_non_dir_h_bond_cuda_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vina_non_dir_h_bond_good, sf_gpu->vina_non_dir_h_bond_bad, sf_gpu->vina_non_dir_h_bond_cutoff);
                        sum += sf_gpu->m_weights[5] * linearattraction_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->linearattraction_cutoff);
                        p_data_gpu[offset].smooth[k][0] = sum;
                        // DEBUG_PRINTF("i=%d, j=%d, k=%d, sum=%f\n", i, j, k, sum);
                    }
                    break;
                }
                case SF_VINARDO:
                {
                    for (int k = 0;k < n; ++k){
                        fl sum = 0;
                        // calculate smooth_e
                        sum += sf_gpu->m_weights[0] * vinardo_gaussian_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vinardo_gaussian_offset, sf_gpu->vinardo_gaussian_width, sf_gpu->vinardo_gaussian_cutoff);
                        sum += sf_gpu->m_weights[1] * vinardo_repulsion_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vinardo_repulsion_cutoff, sf_gpu->vinardo_repulsion_offset);
                        sum += sf_gpu->m_weights[2] * vinardo_hydrophobic_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vinardo_hydrophobic_good, sf_gpu->vinardo_hydrophobic_bad, sf_gpu->vinardo_hydrophobic_cutoff);
                        sum += sf_gpu->m_weights[3] * vinardo_non_dir_h_bond_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->vinardo_non_dir_h_bond_good, sf_gpu->vinardo_non_dir_h_bond_bad, sf_gpu->vinardo_non_dir_h_bond_cutoff);
                        sum += sf_gpu->m_weights[4] * linearattraction_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->linearattraction_cutoff);
                        p_data_gpu[offset].smooth[k][0] = sum;
                        // DEBUG_PRINTF("i=%d, j=%d, k=%d, sum=%f\n", i, j, k, sum);
                    }
                    break;
                }
                case SF_AD42:
                {
                    for (int k = 0;k < n; ++k){
                        fl sum = 0;
                        // calculate smooth_e
                        sum += sf_gpu->m_weights[0] * ad4_vdw_eval(atom_ad_gpu[i], atom_ad_gpu[j], common_rs_gpu[k], sf_gpu->ad4_vdw_smoothing, sf_gpu->ad4_vdw_cap, sf_gpu->ad4_vdw_cutoff);
                        sum += sf_gpu->m_weights[1] * ad4_hb_eval(atom_ad_gpu[i], atom_ad_gpu[j], common_rs_gpu[k], sf_gpu->ad4_hb_smoothing, sf_gpu->ad4_hb_cap, sf_gpu->ad4_hb_cutoff);
                        sum += sf_gpu->m_weights[2] * ad4_electrostatic_eval(atom_charge_gpu[i], atom_charge_gpu[j], common_rs_gpu[k], sf_gpu->ad4_electrostatic_cap, sf_gpu->ad4_electrostatic_cutoff);
                        sum += sf_gpu->m_weights[3] * ad4_solvation_eval_gpu(atom_ad_gpu[i], atom_xs_gpu[i], atom_charge_gpu[i], atom_ad_gpu[j],
                                 atom_xs_gpu[j], atom_charge_gpu[j], sf_gpu->ad4_solvation_desolvation_sigma, sf_gpu->ad4_solvation_solvation_q, sf_gpu->ad4_solvation_charge_dependent, sf_gpu->ad4_solvation_cutoff, common_rs_gpu[k]);
                        sum += sf_gpu->m_weights[4] * linearattraction_eval(atom_xs_gpu[i], atom_xs_gpu[j], common_rs_gpu[k], sf_gpu->linearattraction_cutoff);
                        p_data_gpu[offset].smooth[k][0] = sum;
                        // DEBUG_PRINTF("i=%d, j=%d, k=%d, sum=%f\n", i, j, k, sum);
                    }
                    break;
                }
                default:
                    break;
            }
            // init smooth_dor and fast
            for (int k = 0;k < n; ++k){
                fl dor;
                if (k == 0 || k == n - 1){
                    dor = 0;
                }
                else{
                    fl delta = common_rs_gpu[k+1] - common_rs_gpu[k-1];
                    fl r = common_rs_gpu[k];
                    dor = (p_data_gpu[offset].smooth[k+1][0] - p_data_gpu[offset].smooth[k-1][0]) / (delta * r);
                }
                // DEBUG_PRINTF("i=%d, j=%d, k=%d, dor=%f", i, j, k, dor);

                p_data_gpu[offset].smooth[k][1] = dor;
                fl f1 = p_data_gpu[offset].smooth[k][0];
                fl f2 = (k+1 >= n) ? 0 : p_data_gpu[offset].smooth[k+1][0];
                p_data_gpu[offset].fast[k] = (f2 + f1) / 2;
                // DEBUG_PRINTF("fast=%f\n", p_data_gpu[offset].fast[k]);
            }
        }
    }


}

void precalculate_parallel(triangular_matrix_cuda_t *m_data_list_cpu, std::vector<precalculate_byatom> & m_precalculated_byatom_gpu,
    const ScoringFunction &m_scoring_function, std::vector<model> &m_model_gpu,
    const flv & common_rs, int thread){

    // TODO: copy and transfer data to gpu array
    int max_atom_num = 0;
    for (int i = 0;i < thread; ++i){
        if ((int)m_model_gpu[i].num_atoms() > max_atom_num)
            max_atom_num = (int)m_model_gpu[i].num_atoms();
    }
    DEBUG_PRINTF("max_atom_num = %d\n", max_atom_num);

    // copy atomv from m_model_gpu array and put into atom array

    // using hipHostMalloc may be better
    assert(MAX_LIGAND_NUM >= thread);
    sz atom_xs[thread * max_atom_num]; // maybe size_t is too large
    sz atom_ad[thread * max_atom_num];
    fl atom_charge[thread * max_atom_num];
    int atom_num[thread];
    int precalculate_matrix_size[thread];

    sz *atom_xs_gpu;
    sz *atom_ad_gpu;
    fl *atom_charge_gpu;
    int *atom_num_gpu;
    checkCUDA(hipMalloc(&atom_xs_gpu, thread * max_atom_num * sizeof(sz)));
    checkCUDA(hipMalloc(&atom_ad_gpu, thread * max_atom_num * sizeof(sz)));
    checkCUDA(hipMalloc(&atom_charge_gpu, thread * max_atom_num * sizeof(fl)));
    checkCUDA(hipMalloc(&atom_num_gpu, thread * sizeof(int)));

    for (int i = 0; i < thread; ++i){
        atomv atoms = m_model_gpu[i].get_atoms();
        atom_num[i] = atoms.size();
        precalculate_matrix_size[i] = atom_num[i] * (atom_num[i] + 1) / 2;
        for (int j = 0; j < atoms.size(); ++j){
            atom_xs[i * max_atom_num + j] = atoms[j].xs;
            // DEBUG_PRINTF("atom[%d] on CPU: xs=%lu %lu\n", j, atoms[j].xs, atom_xs[i * max_atom_num + j]);
            atom_ad[i * max_atom_num + j] = atoms[j].ad;
            atom_charge[i * max_atom_num + j] = atoms[j].charge;
        }
    }

    checkCUDA(hipMemcpy(atom_xs_gpu, atom_xs, thread * max_atom_num * sizeof(sz), hipMemcpyHostToDevice));
    // // debug
    // sz atom_xs_check[max_atom_num];
    // checkCUDA(hipMemcpy(atom_xs_check, atom_xs_gpu, max_atom_num * sizeof(sz), hipMemcpyDeviceToHost));
    // for (int i = 0;i < max_atom_num;++i){
    //     DEBUG_PRINTF("atom[%d] on gpu check: xs=%lu\n", i, atom_xs_check[i]);
    // }
    checkCUDA(hipMemcpy(atom_ad_gpu, atom_ad, thread * max_atom_num * sizeof(sz), hipMemcpyHostToDevice));
    checkCUDA(hipMemcpy(atom_charge_gpu, atom_charge, thread * max_atom_num * sizeof(fl), hipMemcpyHostToDevice));
    checkCUDA(hipMemcpy(atom_num_gpu, atom_num, thread * sizeof(int), hipMemcpyHostToDevice));

    // copy scoring function parameters
    scoring_function_cuda_t scoring_cuda;
    scoring_cuda.m_num_potentials = m_scoring_function.m_potentials.size();
    for (int w = 0; w < scoring_cuda.m_num_potentials; ++w){
        scoring_cuda.m_weights[w] = m_scoring_function.m_weights[w];
    }
    scoring_cuda.m_sf_choice = m_scoring_function.m_sf_choice;
    switch (scoring_cuda.m_sf_choice){
        case SF_VINA: // vina
        {
            scoring_cuda.vina_gaussian_offset_1 = 0;
            scoring_cuda.vina_gaussian_width_1 = 0.5;
            scoring_cuda.vina_gaussian_cutoff_1 = 8;
            scoring_cuda.vina_gaussian_offset_2 = 3;
            scoring_cuda.vina_gaussian_width_2 = 2.0;
            scoring_cuda.vina_gaussian_cutoff_2 = 8;
            scoring_cuda.vina_repulsion_offset = 0.0;
            scoring_cuda.vina_repulsion_cutoff = 8.0;
            scoring_cuda.vina_hydrophobic_good = 0.5;
            scoring_cuda.vina_hydrophobic_bad = 1.5;
            scoring_cuda.vina_hydrophobic_cutoff = 8;
            scoring_cuda.vina_non_dir_h_bond_good = -0.7;
            scoring_cuda.vina_non_dir_h_bond_bad = 0;
            scoring_cuda.vina_non_dir_h_bond_cutoff = 8.0;
            scoring_cuda.linearattraction_cutoff = 20.0;
            break;
        }
        case SF_VINARDO: // vinardo
        {
            scoring_cuda.vinardo_gaussian_offset = 0;
            scoring_cuda.vinardo_gaussian_width = 0.8;
            scoring_cuda.vinardo_gaussian_cutoff = 8.0;
            scoring_cuda.vinardo_repulsion_offset = 0;
            scoring_cuda.vinardo_repulsion_cutoff = 8.0;
            scoring_cuda.vinardo_hydrophobic_good = 0;
            scoring_cuda.vinardo_hydrophobic_bad = 2.5;
            scoring_cuda.vinardo_hydrophobic_cutoff = 8.0;
            scoring_cuda.vinardo_non_dir_h_bond_good = -0.6;
            scoring_cuda.vinardo_non_dir_h_bond_bad = 0;
            scoring_cuda.vinardo_non_dir_h_bond_cutoff = 8.0;
            scoring_cuda.linearattraction_cutoff = 20.0;
            break;
        }
        case SF_AD42: // ad4
        {
            scoring_cuda.ad4_vdw_smoothing = 0.5;
            scoring_cuda.ad4_vdw_cap = 100000;
            scoring_cuda.ad4_vdw_cutoff = 8.0;
            scoring_cuda.ad4_hb_smoothing = 0.5;
            scoring_cuda.ad4_hb_cap = 100000;
            scoring_cuda.ad4_hb_cutoff = 8.0;
            scoring_cuda.ad4_electrostatic_cap = 100;
            scoring_cuda.ad4_electrostatic_cutoff = 20.48;
            scoring_cuda.ad4_solvation_desolvation_sigma = 3.6;
            scoring_cuda.ad4_solvation_solvation_q = 0.01097;
            scoring_cuda.ad4_solvation_charge_dependent = true;
            scoring_cuda.ad4_solvation_cutoff = 20.48;
            scoring_cuda.linearattraction_cutoff = 20.0;
            break;
        }
    }
    scoring_function_cuda_t *scoring_cuda_gpu;
    checkCUDA(hipMalloc(&scoring_cuda_gpu, sizeof(scoring_function_cuda_t)));
    checkCUDA(hipMemcpy(scoring_cuda_gpu, &scoring_cuda, sizeof(scoring_function_cuda_t), hipMemcpyHostToDevice));

    // transfer common_rs to gpu
    fl *common_rs_gpu;
    checkCUDA(hipMalloc(&common_rs_gpu, FAST_SIZE * sizeof(fl)));
    checkCUDA(hipMemcpy(common_rs_gpu, common_rs.data(), FAST_SIZE * sizeof(fl), hipMemcpyHostToDevice));

    // malloc output buffer for m_data, array of precalculate_element
    triangular_matrix_cuda_t *m_data_gpu_list;
    checkCUDA(hipMalloc(&m_data_gpu_list, sizeof(triangular_matrix_cuda_t) * thread));
    triangular_matrix_cuda_t m_data_cpu_list[thread];
    triangular_matrix_cuda_t m_data_gpu;
    precalculate_element_cuda_t *precalculate_element_list_ptr_gpu;
    for (int i = 0;i < thread; ++i){
        checkCUDA(hipMalloc(&precalculate_element_list_ptr_gpu, sizeof(precalculate_element_cuda_t) * precalculate_matrix_size[i]));
        m_data_gpu.p_data = precalculate_element_list_ptr_gpu;
        m_data_cpu_list[i].p_data = precalculate_element_list_ptr_gpu;
        checkCUDA(hipMemcpy(m_data_gpu_list + i, &m_data_gpu, sizeof(triangular_matrix_cuda_t), hipMemcpyHostToDevice));
    }


    // TODO: launch kernel
    DEBUG_PRINTF("launch kernel precalculate_gpu, thread=%d\n", thread);
    hipLaunchKernelGGL(precalculate_gpu, dim3(thread / 4 + 1), dim3(4), 0, 0, m_data_gpu_list, scoring_cuda_gpu, atom_xs_gpu, atom_ad_gpu, atom_charge_gpu, atom_num_gpu,
         32, common_rs_gpu, max_fl, thread, max_atom_num);

	checkCUDA(hipDeviceSynchronize());

    DEBUG_PRINTF("kernel exited\n");

    memcpy(m_data_list_cpu, m_data_cpu_list, sizeof(m_data_cpu_list));

    // // debug printing, only check the first ligand
    // DEBUG_PRINTF("energies about the first ligand on GPU:\n");
    // for (int i = 0;i < precalculate_matrix_size[0]; ++i){
    //     DEBUG_PRINTF("precalculated_byatom.m_data.m_data[%d]: (smooth.first, smooth.second, fast) ", i);
    //     for (int j = 0;j < FAST_SIZE; ++j){
    //         DEBUG_PRINTF("(%f, %f, %f) ", m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].first,
    //         m_precalculated_byatom_gpu[0].m_data.m_data[i].smooth[j].second, m_precalculated_byatom_gpu[0].m_data.m_data[i].fast[j]);
    //     }
    //     DEBUG_PRINTF("\n");
    // }


    // TODO: free memory
    checkCUDA(hipFree(atom_xs_gpu));
    checkCUDA(hipFree(atom_ad_gpu));
    checkCUDA(hipFree(atom_charge_gpu));
    checkCUDA(hipFree(atom_num_gpu));
    checkCUDA(hipFree(scoring_cuda_gpu));
    checkCUDA(hipFree(common_rs_gpu));
    checkCUDA(hipFree(m_data_gpu_list));

}
