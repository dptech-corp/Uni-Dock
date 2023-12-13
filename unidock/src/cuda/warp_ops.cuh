#pragma once
#include "common.cuh"
#include "kernel.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

template <unsigned int TileSize>
__device__ __forceinline__ void matrix_d_init_warp(cg::thread_block_tile<TileSize> &tile,
                                                   matrix_d *m, int dim, float fill_data) {
    if (tile.thread_rank() == 0) m->dim = dim;
    if ((dim * (dim + 1) / 2) > MAX_HESSIAN_MATRIX_D_SIZE)
        DEBUG_PRINTF("\nnmatrix_d: matrix_d_init() ERROR!");
    // ((dim * (dim + 1) / 2)*sizeof(float)); // symmetric matrix_d
    for (int i = tile.thread_rank(); i < (dim * (dim + 1) / 2); i += tile.num_threads())
        m->data[i] = fill_data;
    for (int i = (dim * (dim + 1) / 2) + tile.thread_rank(); i < MAX_HESSIAN_MATRIX_D_SIZE;
         i += tile.num_threads())
        m->data[i] = 0;  // Others will be 0
    tile.sync();
}

template <unsigned int TileSize>
__device__ __forceinline__ void matrix_d_set_diagonal_warp(cg::thread_block_tile<TileSize> &tile,
                                                           matrix_d *m, float fill_data) {
    for (int i = tile.thread_rank(); i < m->dim; i += tile.num_threads()) {
        m->data[i + i * (i + 1) / 2] = fill_data;
    }
    tile.sync();
}

template <unsigned int TileSize>
__device__ __forceinline__ float scalar_product_warp(cg::thread_block_tile<TileSize> &tile,
                                                     const change_cuda_t *a, const change_cuda_t *b,
                                                     int n) {
    float tmp = 0;
    for (int i = tile.thread_rank(); i < n; i += tile.num_threads()) {
        tmp += find_change_index_read(a, i) * find_change_index_read(b, i);
    }
    tile.sync();

    return cg::reduce(tile, tmp, cg::plus<float>());
}

template <unsigned int TileSize>
__device__ __forceinline__ void minus_mat_vec_product_warp(cg::thread_block_tile<TileSize> &tile,
                                                           const matrix_d *h,
                                                           const change_cuda_t *in,
                                                           change_cuda_t *out) {
    int n = h->dim;
    for (int i = tile.thread_rank(); i < n; i += tile.num_threads()) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            sum += h->data[index_permissive(h, i, j)] * find_change_index_read(in, j);
        }
        find_change_index_write(out, i, -sum);
    }
    tile.sync();
}

template <unsigned int TileSize>
__device__ __forceinline__ void output_type_cuda_init_warp(cg::thread_block_tile<TileSize> &tile,
                                                           output_type_cuda_t *out,
                                                           const float *ptr) {
    for (int i = tile.thread_rank(); i < 3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION;
         i += tile.num_threads()) {
        if (i < 3)
            out->position[i] = ptr[i];
        else if (i < 7)
            out->orientation[i - 3] = ptr[i];
        else if (i < 7 + MAX_NUM_OF_LIG_TORSION)
            out->lig_torsion[i - 7] = ptr[i];
        else
            out->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION] = ptr[i];
    }

    if (tile.thread_rank() == 0)
        out->lig_torsion_size = ptr[3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION];
    // did not assign coords and e

    tile.sync();
}

template <unsigned int TileSize>
__device__ __forceinline__ void output_type_cuda_init_with_output_warp(
    cg::thread_block_tile<TileSize> &tile, output_type_cuda_t *out_new,
    const output_type_cuda_t *out_old) {
    for (int i = tile.thread_rank(); i < 3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION;
         i += tile.num_threads()) {
        if (i < 3)
            out_new->position[i] = out_old->position[i];
        else if (i < 7)
            out_new->orientation[i - 3] = out_old->orientation[i - 3];
        else if (i < 7 + MAX_NUM_OF_LIG_TORSION)
            out_new->lig_torsion[i - 7] = out_old->lig_torsion[i - 7];
        else
            out_new->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION]
                = out_old->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION];
    }

    if (tile.thread_rank() == 0) {
        out_new->lig_torsion_size = out_old->lig_torsion_size;
        // assign e but not coords
        out_new->e = out_old->e;
    }

    tile.sync();
}

template <unsigned int TileSize> __device__ __forceinline__ void output_type_cuda_increment_warp(
    cg::thread_block_tile<TileSize> &tile, output_type_cuda_t *x, const change_cuda_t *c,
    float factor, float epsilon_fl) {
    // position increment
    if (tile.thread_rank() == 0) {
        for (int k = 0; k < 3; k++) x->position[k] += factor * c->position[k];
        // orientation increment
        float rotation[3];
        for (int k = 0; k < 3; k++) rotation[k] = factor * c->orientation[k];
        quaternion_increment(x->orientation, rotation, epsilon_fl);
    }

    // torsion increment
    for (int k = tile.thread_rank(); k < x->lig_torsion_size; k += tile.num_threads()) {
        float tmp = factor * c->lig_torsion[k];
        normalize_angle(&tmp);
        x->lig_torsion[k] += tmp;
        normalize_angle(&(x->lig_torsion[k]));
    }

    tile.sync();
}

template <unsigned int TileSize> __device__ __forceinline__ void change_cuda_init_with_change_warp(
    cg::thread_block_tile<TileSize> &tile, change_cuda_t *g_new, const change_cuda_t *g_old) {
    for (int i = tile.thread_rank(); i < 3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION;
         i += tile.num_threads()) {
        if (i < 3)
            g_new->position[i] = g_old->position[i];
        else if (i < 7)
            g_new->orientation[i - 3] = g_old->orientation[i - 3];
        else if (i < 7 + MAX_NUM_OF_LIG_TORSION)
            g_new->lig_torsion[i - 7] = g_old->lig_torsion[i - 7];
        else
            g_new->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION]
                = g_old->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION];
    }

    if (tile.thread_rank() == 0) g_new->lig_torsion_size = g_old->lig_torsion_size;
    // did not assign coords and e

    tile.sync();
}

template <unsigned int TileSize>
__device__ __forceinline__ void ligand_init_with_ligand_warp(cg::thread_block_tile<TileSize> &tile,
                                                             const ligand_cuda_t *ligand_cuda_old,
                                                             ligand_cuda_t *ligand_cuda_new) {
    for (int i = tile.thread_rank(); i < MAX_NUM_OF_LIG_PAIRS; i += tile.num_threads()) {
        ligand_cuda_new->pairs.type_pair_index[i] = ligand_cuda_old->pairs.type_pair_index[i];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_LIG_PAIRS; i += tile.num_threads()) {
        ligand_cuda_new->pairs.a[i] = ligand_cuda_old->pairs.a[i];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_LIG_PAIRS; i += tile.num_threads()) {
        ligand_cuda_new->pairs.b[i] = ligand_cuda_old->pairs.b[i];
    }

    if (tile.thread_rank() == 0)
        ligand_cuda_new->pairs.num_pairs = ligand_cuda_old->pairs.num_pairs;

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        for (int j = 0; j < 2; ++j)
            ligand_cuda_new->rigid.atom_range[i][j] = ligand_cuda_old->rigid.atom_range[i][j];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        for (int j = 0; j < 3; ++j)
            ligand_cuda_new->rigid.origin[i][j] = ligand_cuda_old->rigid.origin[i][j];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        for (int j = 0; j < 9; ++j)
            ligand_cuda_new->rigid.orientation_m[i][j] = ligand_cuda_old->rigid.orientation_m[i][j];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        for (int j = 0; j < 4; ++j)
            ligand_cuda_new->rigid.orientation_q[i][j] = ligand_cuda_old->rigid.orientation_q[i][j];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        for (int j = 0; j < 3; ++j)
            ligand_cuda_new->rigid.axis[i][j] = ligand_cuda_old->rigid.axis[i][j];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        for (int j = 0; j < 3; ++j)
            ligand_cuda_new->rigid.relative_axis[i][j] = ligand_cuda_old->rigid.relative_axis[i][j];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        for (int j = 0; j < 3; ++j)
            ligand_cuda_new->rigid.relative_origin[i][j]
                = ligand_cuda_old->rigid.relative_origin[i][j];
    }

    for (int i = tile.thread_rank(); i < MAX_NUM_OF_RIGID; i += tile.num_threads()) {
        ligand_cuda_new->rigid.parent[i] = ligand_cuda_old->rigid.parent[i];
    }

    for (int i = 0; i < MAX_NUM_OF_RIGID; i++) {
        for (int j = tile.thread_rank(); j < MAX_NUM_OF_RIGID; j += tile.num_threads())
            ligand_cuda_new->rigid.children_map[i][j] = ligand_cuda_old->rigid.children_map[i][j];

        for (int j = tile.thread_rank(); j < MAX_NUM_OF_RIGID; j += tile.num_threads())
            ligand_cuda_new->rigid.descendant_map[i][j]
                = ligand_cuda_old->rigid.descendant_map[i][j];
    }

    if (tile.thread_rank() == 0) {
        ligand_cuda_new->rigid.num_children = ligand_cuda_old->rigid.num_children;
        ligand_cuda_new->begin = ligand_cuda_old->begin;
        ligand_cuda_new->end = ligand_cuda_old->end;
    }

    tile.sync();
}

template <unsigned int TileSize>
__device__ __forceinline__ void m_cuda_init_with_m_cuda_warp(cg::thread_block_tile<TileSize> &tile,
                                                             const m_cuda_t *m_cuda_old,
                                                             m_cuda_t *m_cuda_new) {
    for (int i = tile.thread_rank(); i < MAX_NUM_OF_ATOMS; i += tile.num_threads()) {
        m_cuda_new->atoms[i] = m_cuda_old->atoms[i];
    }
    for (int i = tile.thread_rank(); i < MAX_NUM_OF_ATOMS; i += tile.num_threads()) {
        for (int j = 0; j < 3; ++j)
            m_cuda_new->m_coords.coords[i][j] = m_cuda_old->m_coords.coords[i][j];
    }
    for (int i = tile.thread_rank(); i < MAX_NUM_OF_ATOMS; i += tile.num_threads()) {
        for (int j = 0; j < 3; ++j)
            m_cuda_new->minus_forces.coords[i][j] = m_cuda_old->minus_forces.coords[i][j];
    }

    ligand_init_with_ligand_warp(tile, &m_cuda_old->ligand, &m_cuda_new->ligand);

    if (tile.thread_rank() == 0) m_cuda_new->m_num_movable_atoms = m_cuda_old->m_num_movable_atoms;

    tile.sync();
}

template <unsigned int TileSize>
__device__ __forceinline__ float ig_eval_deriv_warp(cg::thread_block_tile<TileSize> &tile,
                                                    output_type_cuda_t *x, const float v,
                                                    ig_cuda_t *ig_cuda_gpu, m_cuda_t *m_cuda_gpu,
                                                    const float epsilon_fl) {
    float e = 0;
    float deriv[3];
    int nat = num_atom_types(ig_cuda_gpu->atu);
    for (int i = tile.thread_rank(); i < m_cuda_gpu->m_num_movable_atoms; i += tile.num_threads()) {
        int t = m_cuda_gpu->atoms[i].types[ig_cuda_gpu->atu];
        if (t >= nat) {
            m_cuda_gpu->minus_forces.coords[i][0] = 0.0f;
            m_cuda_gpu->minus_forces.coords[i][1] = 0.0f;
            m_cuda_gpu->minus_forces.coords[i][2] = 0.0f;
            continue;
        }

        e += g_evaluate(&ig_cuda_gpu->grids[t], m_cuda_gpu->m_coords.coords[i], ig_cuda_gpu->slope,
                        v, deriv, epsilon_fl);

        m_cuda_gpu->minus_forces.coords[i][0] = deriv[0];
        m_cuda_gpu->minus_forces.coords[i][1] = deriv[1];
        m_cuda_gpu->minus_forces.coords[i][2] = deriv[2];
    }
    tile.sync();
    return e;
}

template <unsigned int TileSize> __device__ __forceinline__ float eval_interacting_pairs_deriv_warp(
    cg::thread_block_tile<TileSize> &tile, p_cuda_t *p_cuda_gpu, const float v,
    const lig_pairs_cuda_t *pairs, const m_coords_cuda_t *m_coords, m_minus_forces_t *minus_forces,
    const float epsilon_fl) {
    float e = 0.0f;

    for (int i = tile.thread_rank(); i < pairs->num_pairs; i += tile.num_threads()) {
        int ai = pairs->a[i], bi = pairs->b[i];
        int index = pairs->a[i] + pairs->b[i] * (pairs->b[i] + 1) / 2;
        float r[3] = {m_coords->coords[bi][0] - m_coords->coords[ai][0],
                      m_coords->coords[bi][1] - m_coords->coords[ai][1],
                      m_coords->coords[bi][2] - m_coords->coords[ai][2]};
        float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

        if (r2 < p_cuda_gpu->m_cutoff_sqr) {
            float tmp[2];
            p_eval_deriv(tmp, index, r2, p_cuda_gpu, epsilon_fl);
            float force[3] = {r[0] * tmp[1], r[1] * tmp[1], r[2] * tmp[1]};
            curl(&tmp[0], force, v, epsilon_fl);
            e += tmp[0];
            atomicAdd(&minus_forces->coords[ai][0], -force[0]);
            atomicAdd(&minus_forces->coords[ai][1], -force[1]);
            atomicAdd(&minus_forces->coords[ai][2], -force[2]);
            atomicAdd(&minus_forces->coords[bi][0], force[0]);
            atomicAdd(&minus_forces->coords[bi][1], force[1]);
            atomicAdd(&minus_forces->coords[bi][2], force[2]);
        }
    }
    tile.sync();
    return e;
}

template <unsigned int TileSize>
__device__ __forceinline__ void POT_deriv_warp(cg::thread_block_tile<TileSize> &tile,
                                               const m_minus_forces_t *minus_forces,
                                               const rigid_cuda_t *lig_rigid_gpu,
                                               const m_coords_cuda_t *m_coords, change_cuda_t *g,
                                               pot_cuda_t *p) {
    int num_torsion = lig_rigid_gpu->num_children;
    int num_rigid = num_torsion + 1;

    float pos_tmp[3], ori_tmp[3], tmp1[3], tmp2[3], tmp3[3];
    for (int i = tile.thread_rank(); i < num_rigid; i += tile.num_threads()) {
        int begin = lig_rigid_gpu->atom_range[i][0];
        int end = lig_rigid_gpu->atom_range[i][1];
        pos_tmp[0] = pos_tmp[1] = pos_tmp[2] = 0.0f;
        ori_tmp[0] = ori_tmp[1] = ori_tmp[2] = 0.0f;
        for (int j = begin; j < end; j++) {
            pos_tmp[0] += minus_forces->coords[j][0];
            pos_tmp[1] += minus_forces->coords[j][1];
            pos_tmp[2] += minus_forces->coords[j][2];
            tmp1[0] = m_coords->coords[j][0] - lig_rigid_gpu->origin[i][0];
            tmp1[1] = m_coords->coords[j][1] - lig_rigid_gpu->origin[i][1];
            tmp1[2] = m_coords->coords[j][2] - lig_rigid_gpu->origin[i][2];
            tmp2[0] = minus_forces->coords[j][0];
            tmp2[1] = minus_forces->coords[j][1];
            tmp2[2] = minus_forces->coords[j][2];
            product(tmp3, tmp1, tmp2);
            ori_tmp[0] += tmp3[0];
            ori_tmp[1] += tmp3[1];
            ori_tmp[2] += tmp3[2];
        }
        p->ptmp[i][0] = pos_tmp[0];
        p->ptmp[i][1] = pos_tmp[1];
        p->ptmp[i][2] = pos_tmp[2];
        p->o[i][0] = ori_tmp[0];
        p->o[i][1] = ori_tmp[1];
        p->o[i][2] = ori_tmp[2];
    }
    tile.sync();

    /* position_derivative  */
    for (int i = tile.thread_rank(); i < num_rigid; i += tile.num_threads()) {
        p->p[i][0] = p->ptmp[i][0];
        p->p[i][1] = p->ptmp[i][1];
        p->p[i][2] = p->ptmp[i][2];
        for (int j = i + 1; j < num_rigid; j++) {
            if (lig_rigid_gpu->descendant_map[i][j]) {
                p->p[i][0] += p->ptmp[j][0];
                p->p[i][1] += p->ptmp[j][1];
                p->p[i][2] += p->ptmp[j][2];
            }
        }
    }
    tile.sync();

    /* orientation derivative */
    if (tile.thread_rank() == 0) {  // NOTE: Single thread is better here
        float origin_temp[3], product_out[3];
        for (int i = num_rigid - 1; i >= 0; i--) { /* from bottom to top */
            ori_tmp[0] = p->o[i][0];
            ori_tmp[1] = p->o[i][1];
            ori_tmp[2] = p->o[i][2];
            for (int j = i + 1; j < num_rigid; j++) {
                if (lig_rigid_gpu->children_map[i][j]) {
                    ori_tmp[0] += p->o[j][0];
                    ori_tmp[1] += p->o[j][1];
                    ori_tmp[2] += p->o[j][2]; /* self+children node
                                               */

                    origin_temp[0] = lig_rigid_gpu->origin[j][0] - lig_rigid_gpu->origin[i][0];
                    origin_temp[1] = lig_rigid_gpu->origin[j][1] - lig_rigid_gpu->origin[i][1];
                    origin_temp[2] = lig_rigid_gpu->origin[j][2] - lig_rigid_gpu->origin[i][2];

                    product(product_out, origin_temp, p->p[j]);
                    ori_tmp[0] += product_out[0];
                    ori_tmp[1] += product_out[1];
                    ori_tmp[2] += product_out[2];
                }
            }
            p->o[i][0] = ori_tmp[0];
            p->o[i][1] = ori_tmp[1];
            p->o[i][2] = ori_tmp[2];
        }
    }
    tile.sync();

    /* torsion_derivative */
    for (int i = tile.thread_rank(); i < num_torsion; i += tile.num_threads()) {
        g->lig_torsion[i - 1] = p->o[i][0] * lig_rigid_gpu->axis[i][0]
                                + p->o[i][1] * lig_rigid_gpu->axis[i][1]
                                + p->o[i][2] * lig_rigid_gpu->axis[i][2];
    }
    tile.sync();

    for (int i = tile.thread_rank(); i < 3; i += tile.num_threads()) {
        g->position[i] = p->p[0][i];
        g->orientation[i] = p->o[0][i];
    }

    tile.sync();
}

template <unsigned int TileSize>
__device__ float m_eval_deriv_warp(cg::thread_block_tile<TileSize> &tile, output_type_cuda_t *c,
                                   change_cuda_t *g, m_cuda_t *m_cuda_gpu, p_cuda_t *p_cuda_gpu,
                                   ig_cuda_t *ig_cuda_gpu, pot_cuda_t *pot_aux, const float *v,
                                   const float epsilon_fl) {
    // check set args
    if (tile.thread_rank() == 0) {
        set(c, &m_cuda_gpu->ligand.rigid, &m_cuda_gpu->m_coords, m_cuda_gpu->atoms,
            m_cuda_gpu->m_num_movable_atoms, epsilon_fl);
    }
    tile.sync();

    float e = ig_eval_deriv_warp(tile, c, v[1], ig_cuda_gpu, m_cuda_gpu, epsilon_fl);
    e += eval_interacting_pairs_deriv_warp(tile, p_cuda_gpu, v[0], &m_cuda_gpu->ligand.pairs,
                                           &m_cuda_gpu->m_coords, &m_cuda_gpu->minus_forces,
                                           epsilon_fl);
    tile.sync();
    e = cg::reduce(tile, e, cg::plus<float>());

    // should add derivs for glue, other and inter pairs
    POT_deriv_warp(tile, &m_cuda_gpu->minus_forces, &m_cuda_gpu->ligand.rigid,
                   &m_cuda_gpu->m_coords, g, pot_aux);

    return e;
}

template <unsigned int TileSize> __device__ __forceinline__ void line_search_warp(
    cg::thread_block_tile<TileSize> &tile, m_cuda_t *m_cuda_gpu, p_cuda_t *p_cuda_gpu,
    ig_cuda_t *ig_cuda_gpu, int n, const output_type_cuda_t *x, const change_cuda_t *g,
    const float f0, const change_cuda_t *p, output_type_cuda_t *x_new, change_cuda_t *g_new,
    pot_cuda_t *pot_aux, float *f, float *alpha, const float epsilon_fl, const float *hunt_cap) {
    const float c0 = 0.0001;
    const int max_trials = 10;
    const float multiplier = 0.5;
    float alpha_ = 1.0, f_;

    const float pg = scalar_product_warp(tile, p, g, n);
    for (int trial = 0; trial < max_trials; trial++) {
        output_type_cuda_init_with_output_warp(tile, x_new, x);
        output_type_cuda_increment_warp(tile, x_new, p, alpha_, epsilon_fl);
        f_ = m_eval_deriv_warp(tile, x_new, g_new, m_cuda_gpu, p_cuda_gpu, ig_cuda_gpu, pot_aux,
                               hunt_cap, epsilon_fl);
        if (f_ - f0 < c0 * alpha_ * pg) break;
        alpha_ *= multiplier;
    }

    *f = f_;
    *alpha = alpha_;
}

template <unsigned int TileSize>
__device__ __forceinline__ bool bfgs_update_warp(cg::thread_block_tile<TileSize> &tile, matrix_d *h,
                                                 const change_cuda_t *p, const change_cuda_t *y,
                                                 change_cuda_t *minus_hy, const float alpha,
                                                 const float epsilon_fl) {
    float yp, yhy;
    yp = scalar_product_warp(tile, y, p, h->dim);
    if (alpha * yp < epsilon_fl) return false;

    change_cuda_init_with_change_warp(tile, minus_hy, y);
    minus_mat_vec_product_warp(tile, h, y, minus_hy);
    yhy = -scalar_product_warp(tile, y, minus_hy, h->dim);
    float r = 1 / (alpha * yp);
    int n = 6 + p->lig_torsion_size;

    __shared__ float minus_hy_[(6 + MAX_NUM_OF_LIG_TORSION) * 32 / TileSize],
        p_[(6 + MAX_NUM_OF_LIG_TORSION) * 32 / TileSize];

    // Calculate offset
    auto minus_hy_ptr_ = &minus_hy_[tile.meta_group_rank() * (6 + MAX_NUM_OF_LIG_TORSION)];
    auto p_ptr_ = &p_[tile.meta_group_rank() * (6 + MAX_NUM_OF_LIG_TORSION)];
    for (int i = tile.thread_rank(); i < n; i += tile.num_threads()) {
        minus_hy_ptr_[i] = find_change_index_read(minus_hy, i);
        p_ptr_[i] = find_change_index_read(p, i);
    }
    tile.sync();

    for (int i = tile.thread_rank(); i < n; i += tile.num_threads()) {
        for (int j = i; j < n; j++) {
            float tmp = alpha * r * (minus_hy_ptr_[i] * p_ptr_[j] + minus_hy_ptr_[j] * p_ptr_[i])
                        + alpha * alpha * (r * r * yhy + r) * p_ptr_[i] * p_ptr_[j];
            h->data[i + j * (j + 1) / 2] += tmp;
        }
    }
    tile.sync();
    return true;
}

template <unsigned int TileSize>
__device__ void bfgs_warp(cg::thread_block_tile<TileSize> &tile, output_type_cuda_t *x,
                          output_type_cuda_t *x_new, output_type_cuda_t *x_orig, change_cuda_t *g,
                          change_cuda_t *g_new, change_cuda_t *g_orig, change_cuda_t *p,
                          change_cuda_t *y, change_cuda_t *minus_hy, matrix_d *h,
                          m_cuda_t *m_cuda_gpu, p_cuda_t *p_cuda_gpu, ig_cuda_t *ig_cuda_gpu,
                          pot_cuda_t *pot_aux, const float *hunt_cap, const float epsilon_fl,
                          const int max_steps) {
    // Profiling: perform timing within kernel
    int n = 3 + 3 + x->lig_torsion_size; /* the dimensions of matirx */

    float f0, f1, f_orig, alpha;

    matrix_d_init_warp(tile, h, n, 0);
    matrix_d_set_diagonal_warp(tile, h, 1);
    change_cuda_init_with_change_warp(tile, g_new, g);
    output_type_cuda_init_with_output_warp(tile, x_new, x);
    f_orig = m_eval_deriv_warp(tile, x, g, m_cuda_gpu, p_cuda_gpu, ig_cuda_gpu, pot_aux, hunt_cap,
                               epsilon_fl);

    /* Init g_orig, x_orig */
    change_cuda_init_with_change_warp(tile, g_orig, g);
    output_type_cuda_init_with_output_warp(tile, x_orig, x);

    /* Init p */
    change_cuda_init_with_change_warp(tile, p, g);

    for (int step = 0; step < max_steps; step++) {
        minus_mat_vec_product_warp(tile, h, g, p);
        line_search_warp(tile, m_cuda_gpu, p_cuda_gpu, ig_cuda_gpu, n, x, g, f0, p, x_new, g_new,
                         pot_aux, &f1, &alpha, epsilon_fl, hunt_cap);
        change_cuda_init_with_change_warp(tile, y, g_new);

        /* subtract_change */
        for (int i = tile.thread_rank(); i < n; i += tile.num_threads()) {
            float tmp = find_change_index_read(y, i) - find_change_index_read(g, i);
            find_change_index_write(y, i, tmp);
        }
        tile.sync();
        f0 = f1;

        output_type_cuda_init_with_output_warp(tile, x, x_new);

        float gg = scalar_product_warp(tile, g, g, n);
        if (!(sqrtf(gg) >= 1e-5f)) break;

        change_cuda_init_with_change_warp(tile, g, g_new);

        if (step == 0) {
            float yy = scalar_product_warp(tile, y, y, n);
            if (fabs(yy) > epsilon_fl) {
                float yp = scalar_product_warp(tile, y, p, n);
                matrix_d_set_diagonal_warp(tile, h, alpha * yp / yy);
            }
        }
        tile.sync();

        bfgs_update_warp(tile, h, p, y, minus_hy, alpha, epsilon_fl);
    }

    if (!(f0 <= f_orig)) {
        f0 = f_orig;
        output_type_cuda_init_with_output_warp(tile, x, x_orig);
        change_cuda_init_with_change_warp(tile, g, g_orig);
    }

    // write output_type_cuda energy
    x->e = f0;
}

__device__ __forceinline__ void set_warp(const output_type_cuda_t *x, rigid_cuda_t *lig_rigid_gpu,
                                         m_coords_cuda_t *m_coords_gpu, const atom_cuda_t *atoms,
                                         const int m_num_movable_atoms, const float epsilon_fl) {
    for (int i = 0; i < 3; i++) lig_rigid_gpu->origin[0][i] = x->position[i];
    for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[0][i] = x->orientation[i];
    quaternion_to_r3(lig_rigid_gpu->orientation_q[0],
                     lig_rigid_gpu->orientation_m[0]); /* set orientation_m */

    int begin = lig_rigid_gpu->atom_range[0][0];
    int end = lig_rigid_gpu->atom_range[0][1];
    for (int i = begin; i < end; i++) {
        local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[0], atoms[i].coords,
                     lig_rigid_gpu->orientation_m[0]);
    }
    /* ************* end node.set_conf ************* */

    /* ************* branches_set_conf ************* */
    /* update nodes in depth-first order */
    for (int current = 1; current < lig_rigid_gpu->num_children + 1;
         current++) { /* current starts from 1 (namely starts from first
                         child node) */
        int parent = lig_rigid_gpu->parent[current];
        float torsion = x->lig_torsion[current - 1]; /* torsions are all related to child nodes */
        local_to_lab(lig_rigid_gpu->origin[current], lig_rigid_gpu->origin[parent],
                     lig_rigid_gpu->relative_origin[current], lig_rigid_gpu->orientation_m[parent]);
        local_to_lab_direction(lig_rigid_gpu->axis[current], lig_rigid_gpu->relative_axis[current],
                               lig_rigid_gpu->orientation_m[parent]);
        float tmp[4];
        float parent_q[4]
            = {lig_rigid_gpu->orientation_q[parent][0], lig_rigid_gpu->orientation_q[parent][1],
               lig_rigid_gpu->orientation_q[parent][2], lig_rigid_gpu->orientation_q[parent][3]};
        float current_axis[3] = {lig_rigid_gpu->axis[current][0], lig_rigid_gpu->axis[current][1],
                                 lig_rigid_gpu->axis[current][2]};

        angle_to_quaternion2(tmp, current_axis, torsion);
        angle_to_quaternion_multi(tmp, parent_q);
        quaternion_normalize_approx(tmp, epsilon_fl);

        for (int i = 0; i < 4; i++)
            lig_rigid_gpu->orientation_q[current][i] = tmp[i]; /* set orientation_q */
        quaternion_to_r3(lig_rigid_gpu->orientation_q[current],
                         lig_rigid_gpu->orientation_m[current]); /* set orientation_m */

        /* set coords */
        begin = lig_rigid_gpu->atom_range[current][0];
        end = lig_rigid_gpu->atom_range[current][1];
        for (int i = begin; i < end; i++) {
            local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[current], atoms[i].coords,
                         lig_rigid_gpu->orientation_m[current]);
        }
    }
    /* ************* end branches_set_conf ************* */
}

template <unsigned int TileSize>
__device__ __forceinline__ void write_back_warp(cg::thread_block_tile<TileSize> &tile,
                                                output_type_cuda_t *results,
                                                const output_type_cuda_t *best_out) {
    for (int i = tile.thread_rank();
         i < 3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION + MAX_NUM_OF_ATOMS;
         i += tile.num_threads()) {
        if (i < 3) {
            results->position[i] = best_out->position[i];
        } else if (i < 7) {
            results->orientation[i - 3] = best_out->orientation[i - 3];
        } else if (i < 7 + MAX_NUM_OF_LIG_TORSION) {
            results->lig_torsion[i - 7] = best_out->lig_torsion[i - 7];
        } else if (i < 7 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION) {
            results->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION]
                = best_out->flex_torsion[i - 7 - MAX_NUM_OF_LIG_TORSION];
        } else {
#pragma unroll
            for (int j = 0; j < 3; j++) {
                results->coords[i - 7 - MAX_NUM_OF_LIG_TORSION - MAX_NUM_OF_FLEX_TORSION][j]
                    = best_out->coords[i - 7 - MAX_NUM_OF_LIG_TORSION - MAX_NUM_OF_FLEX_TORSION][j];
            }
        }
    }

    if (tile.thread_rank() == 0) {
        results->lig_torsion_size = best_out->lig_torsion_size;
        results->e = best_out->e;
    }

    tile.sync();
}
