#pragma once
#include "curand_kernel.h"
#include "kernel.h"
#include "math.h"
#include <cmath>
#include "precalculate.h"

#define M_PI_F 3.1415927f

/* Below based on mutate_conf.cpp */

__device__ __forceinline__ void quaternion_increment(float *q, const float *rotation,
                                                     float epsilon_fl);

__device__ __forceinline__ void normalize_angle(float *x);

__device__ __forceinline__ float norm3(const float *a) {
    return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

__device__ __forceinline__ void random_inside_sphere_gpu(float *random_inside_sphere,
                                                         curandStatePhilox4_32_10_t *state) {
    float4 random_inside_sphere_fl;
    while (true) {  // on average, this will have to be run about twice
        random_inside_sphere_fl = curand_uniform4(state);  // ~ U[0,1]
        random_inside_sphere[0] = (random_inside_sphere_fl.x - 0.5) * 2.0;
        random_inside_sphere[1] = (random_inside_sphere_fl.y - 0.5) * 2.0;
        random_inside_sphere[2] = (random_inside_sphere_fl.z - 0.5) * 2.0;
        random_inside_sphere[3] = random_inside_sphere_fl.w;
        float r = norm3(random_inside_sphere);
        if (r < 1) {
            return;
        }
    }
}

__device__ __forceinline__ void normalize_angle(float *x) {
    while (1) {
        if (*x >= -(M_PI_F) && *x <= (M_PI_F)) {
            break;
        } else if (*x > 3 * (M_PI_F)) {
            float n = (*x - (M_PI_F)) / (2 * (M_PI_F));
            *x -= 2 * (M_PI_F)*ceil(n);
        } else if (*x < 3 * -(M_PI_F)) {
            float n = (-*x - (M_PI_F)) / (2 * (M_PI_F));
            *x += 2 * (M_PI_F)*ceil(n);
        } else if (*x > (M_PI_F)) {
            *x -= 2 * (M_PI_F);
        } else if (*x < -(M_PI_F)) {
            *x += 2 * (M_PI_F);
        } else {
            break;
        }
    }
}

__device__ __forceinline__ bool quaternion_is_normalized(float *q) {
    float q_pow = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    float sqrt_q_pow = sqrtf(q_pow);
    return (q_pow - 1 < 0.001) && (sqrt_q_pow - 1 < 0.001);
}

__device__ __forceinline__ void angle_to_quaternion(float *q, const float *rotation,
                                                    float epsilon_fl) {
    float angle = norm3(rotation);
    if (angle > epsilon_fl) {
        float axis[3] = {rotation[0] / angle, rotation[1] / angle, rotation[2] / angle};
        normalize_angle(&angle);
        float c = cos(angle / 2);
        float s = sin(angle / 2);
        q[0] = c;
        q[1] = s * axis[0];
        q[2] = s * axis[1];
        q[3] = s * axis[2];
        return;
    }
    q[0] = 1;
    q[1] = 0;
    q[2] = 0;
    q[3] = 0;
    return;
}

// quaternion multiplication
__device__ __forceinline__ void angle_to_quaternion_multi(float *qa, const float *qb) {
    float tmp[4] = {qa[0], qa[1], qa[2], qa[3]};
    qa[0] = tmp[0] * qb[0] - tmp[1] * qb[1] - tmp[2] * qb[2] - tmp[3] * qb[3];
    qa[1] = tmp[0] * qb[1] + tmp[1] * qb[0] + tmp[2] * qb[3] - tmp[3] * qb[2];
    qa[2] = tmp[0] * qb[2] - tmp[1] * qb[3] + tmp[2] * qb[0] + tmp[3] * qb[1];
    qa[3] = tmp[0] * qb[3] + tmp[1] * qb[2] - tmp[2] * qb[1] + tmp[3] * qb[0];
}

__device__ __forceinline__ void quaternion_normalize_approx(float *q, float epsilon_fl) {
    const float s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    // Omit one assert()
    if (fabs(s - 1) < TOLERANCE)
        ;
    else {
        const float a = sqrtf(s);
        for (int i = 0; i < 4; i++) q[i] /= a;
    }
}

__device__ __forceinline__ void quaternion_increment(float *q, const float *rotation,
                                                     float epsilon_fl) {
    float q_old[4] = {q[0], q[1], q[2], q[3]};
    angle_to_quaternion(q, rotation, epsilon_fl);
    angle_to_quaternion_multi(q, q_old);
    quaternion_normalize_approx(q, epsilon_fl);
    // assert(quaternion_is_normalized(q)); // unnecessary
}

__device__ __forceinline__ float vec_distance_sqr(float *a, float *b) {
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
           + (a[2] - b[2]) * (a[2] - b[2]);
}

__device__ __forceinline__ float gyration_radius(int m_lig_begin, int m_lig_end,
                                                 const atom_cuda_t *atoms,
                                                 const m_coords_cuda_t *m_coords_gpu,
                                                 const float *m_lig_node_origin) {
    float acc = 0;
    int counter = 0;
    float origin[3] = {m_lig_node_origin[0], m_lig_node_origin[1], m_lig_node_origin[2]};
    for (int i = m_lig_begin; i < m_lig_end; i++) {
        float current_coords[3]
            = {m_coords_gpu->coords[i][0], m_coords_gpu->coords[i][1], m_coords_gpu->coords[i][2]};
        if (atoms[i].types[0]
            != EL_TYPE_H) {  // for el, we use the first element (atoms[i].types[0])
            acc += vec_distance_sqr(current_coords, origin);
            ++counter;
        }
    }
    return (counter > 0) ? sqrtf(acc / counter) : 0;
}

__device__ __forceinline__ void mutate_conf_cuda(const int num_steps, output_type_cuda_t *c,
                                                 curandStatePhilox4_32_10_t *state,
                                                 const int m_lig_begin, const int m_lig_end,
                                                 const atom_cuda_t *atoms,
                                                 const m_coords_cuda_t *m_coords_gpu,
                                                 const float *m_lig_node_origin_gpu,
                                                 const float epsilon_fl, const float amplitude) {
    int flex_torsion_size = 0;  // FIX? 20210727
    int count_mutable_entities = 2 + c->lig_torsion_size + flex_torsion_size;
    int which = curand(state) % count_mutable_entities;
    float random_inside_sphere[4];
    random_inside_sphere_gpu(random_inside_sphere, state);
    if (which == 0) {
        DEBUG_PRINTF("random sphere r=%f\n", norm3(random_inside_sphere));
    }

    float random_pi = (random_inside_sphere[3] - 0.5) * 2.0 * pi;  // ~ U[-pi, pi]
    if (which == 0) {
        DEBUG_PRINTF("random pi=%f\n", random_pi);
    }

    if (which == 0) {
        for (int i = 0; i < 3; i++) c->position[i] += amplitude * random_inside_sphere[i];
        return;
    }
    --which;
    if (which == 0) {
        float gr
            = gyration_radius(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu);
        if (gr > epsilon_fl) {
            float rotation[3];
            for (int i = 0; i < 3; i++) rotation[i] = amplitude / gr * random_inside_sphere[i];
            quaternion_increment(c->orientation, rotation, epsilon_fl);
        }
        return;
    }
    --which;
    if (which < c->lig_torsion_size) {
        c->lig_torsion[which] = random_pi;
        return;
    }
    which -= c->lig_torsion_size;

    if (flex_torsion_size != 0) {
        if (which < flex_torsion_size) {
            c->flex_torsion[which] = random_pi;
            return;
        }
        which -= flex_torsion_size;
    }
}

/*  Above based on mutate_conf.cpp */

/* Below based on matrix.cpp */

// as rugular 3x3 matrix_d
__device__ __forceinline__ void mat_init(matrix_d *m, float fill_data) {
    m->dim = 3;  // fixed to 3x3 matrix_d
    if (9 > MAX_HESSIAN_MATRIX_D_SIZE) DEBUG_PRINTF("\nnmatrix_d: mat_init() ERROR!");
    for (int i = 0; i < 9; i++) m->data[i] = fill_data;
}

// as regular matrix_d
__device__ __forceinline__ void matrix_d_set_element(matrix_d *m, int dim, int x, int y,
                                                     float fill_data) {
    m->data[x + y * dim] = fill_data;
}

__device__ __forceinline__ void matrix_d_set_element_tri(matrix_d *m, int x, int y,
                                                         float fill_data) {
    m->data[x + y * (y + 1) / 2] = fill_data;
}
__device__ __forceinline__ int tri_index(int n, int i, int j) {
    if (j >= n || i > j) DEBUG_PRINTF("\nmatrix_d: tri_index ERROR!");
    return i + j * (j + 1) / 2;
}

__device__ __forceinline__ int index_permissive(const matrix_d *m, int i, int j) {
    return (i < j) ? tri_index(m->dim, i, j) : tri_index(m->dim, j, i);
}

/* Above based on matrix_d.cpp */

/* Below based on quasi_newton.cpp */

__device__ __forceinline__ void change_cuda_init(change_cuda_t *g, const float *ptr) {
    for (int i = 0; i < 3; i++) g->position[i] = ptr[i];
    for (int i = 0; i < 3; i++) g->orientation[i] = ptr[i + 3];
    for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++) g->lig_torsion[i] = ptr[i + 3 + 3];
    for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)
        g->flex_torsion[i] = ptr[i + 3 + 3 + MAX_NUM_OF_LIG_TORSION];
    g->lig_torsion_size = ptr[3 + 3 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION];
}

__device__ __forceinline__ void change_cuda_init_with_change(change_cuda_t *g_new,
                                                             const change_cuda_t *g_old) {
    for (int i = 0; i < 3; i++) g_new->position[i] = g_old->position[i];
    for (int i = 0; i < 3; i++) g_new->orientation[i] = g_old->orientation[i];
    for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++) g_new->lig_torsion[i] = g_old->lig_torsion[i];
    for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)
        g_new->flex_torsion[i] = g_old->flex_torsion[i];
    g_new->lig_torsion_size = g_old->lig_torsion_size;
}

void print_output_type(output_type_cuda_t *x, int torsion_size) {
    for (int i = 0; i < 3; i++) DEBUG_PRINTF("\nx.position[%d] = %0.16f", i, x->position[i]);
    for (int i = 0; i < 4; i++) DEBUG_PRINTF("\nx.orientation[%d] = %0.16f", i, x->orientation[i]);
    for (int i = 0; i < torsion_size; i++)
        DEBUG_PRINTF("\n x.torsion[%d] = %0.16f", i, x->lig_torsion[i]);
    DEBUG_PRINTF("\n x.torsion_size = %f", x->lig_torsion_size);
    DEBUG_PRINTF("\n !!! x.e = %f\n", x->e);
}

void print_change(change_cuda_t *g, int torsion_size) {
    for (int i = 0; i < 3; i++) DEBUG_PRINTF("\ng.position[%d] = %0.16f", i, g->position[i]);
    for (int i = 0; i < 3; i++) DEBUG_PRINTF("\ng.orientation[%d] = %0.16f", i, g->orientation[i]);
    for (int i = 0; i < torsion_size; i++)
        DEBUG_PRINTF("\ng.torsion[%d] = %0.16f", i, g->lig_torsion[i]);
    DEBUG_PRINTF("\ng.torsion_size = %f", g->lig_torsion_size);
}

__device__ __forceinline__ int num_atom_types(int atu) {
    switch (atu) {
        case 0:
            return EL_TYPE_SIZE;
        case 1:
            return AD_TYPE_SIZE;
        case 2:
            return XS_TYPE_SIZE;
        case 3:
            return SY_TYPE_SIZE;
        default:
            DEBUG_PRINTF("Kernel1:num_atom_types() ERROR!");
            return INFINITY;
    }
}

__device__ __forceinline__ void elementwise_product(float *out, const float *a, const float *b) {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    out[2] = a[2] * b[2];
}

__device__ __forceinline__ float elementwise_product_sum(const float *a, const float *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ float access_m_data(float *m_data, int m_i, int m_j, int i, int j,
                                               int k) {
    return m_data[i + m_i * (j + m_j * k)];
}

__device__ __forceinline__ bool not_max_gpu(float x) {
    return (x < 0.1 * INFINITY); /* Problem: replace max_fl with INFINITY? */
}

__device__ __forceinline__ void curl_with_deriv(float *e, float *deriv, float v,
                                                const float epsilon_fl) {
    if (*e > 0 && not_max_gpu(v)) {
        float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
        *e *= tmp;
        for (int i = 0; i < 3; i++) deriv[i] *= tmp * tmp;
    }
}

__device__ __forceinline__ void curl_without_deriv(float *e, float v, const float epsilon_fl) {
    if (*e > 0 && not_max_gpu(v)) {
        float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
        *e *= tmp;
    }
}

__device__ __forceinline__ float g_evaluate(grid_cuda_t *g, const float *m_coords, /* double[3] */
                                            const float slope,                     /* double */
                                            const float v,                         /* double */
                                            float *deriv,                          /* double[3] */
                                            const float epsilon_fl) {
    int m_i = g->m_i;
    int m_j = g->m_j;
    int m_k = g->m_k;
    if (m_i * m_j * m_k == 0) DEBUG_PRINTF("\nkernel2: g_evaluate ERROR!#1");
    float tmp_vec[3]
        = {m_coords[0] - g->m_init[0], m_coords[1] - g->m_init[1], m_coords[2] - g->m_init[2]};
    float tmp_vec2[3] = {g->m_factor[0], g->m_factor[1], g->m_factor[2]};
    float s[3];
    elementwise_product(s, tmp_vec, tmp_vec2);

    float miss[3] = {0, 0, 0};
    int region[3];
    int a[3];
    int m_data_dims[3] = {m_i, m_j, m_k};
    for (int i = 0; i < 3; i++) {
        if (s[i] < 0) {
            miss[i] = -s[i];
            region[i] = -1;
            a[i] = 0;
            s[i] = 0;
        } else if (s[i] >= g->m_dim_fl_minus_1[i]) {
            miss[i] = s[i] - g->m_dim_fl_minus_1[i];
            region[i] = 1;
            if (m_data_dims[i] < 2) DEBUG_PRINTF("\nKernel2: g_evaluate ERROR!#2");
            a[i] = m_data_dims[i] - 2;
            s[i] = 1;
        } else {
            region[i] = 0;
            a[i] = (int)s[i];
            s[i] -= a[i];
        }
        if (s[i] < 0) DEBUG_PRINTF("\nKernel2: g_evaluate ERROR!#3");
        if (s[i] > 1) DEBUG_PRINTF("\nKernel2: g_evaluate ERROR!#4");
        if (a[i] < 0) DEBUG_PRINTF("\nKernel2: g_evaluate ERROR!#5");
        if (a[i] + 1 >= m_data_dims[i]) DEBUG_PRINTF("\nKernel2: g_evaluate ERROR!#5");
    }

    float tmp_m_factor_inv[3] = {g->m_factor_inv[0], g->m_factor_inv[1], g->m_factor_inv[2]};
    const float penalty = slope * elementwise_product_sum(miss, tmp_m_factor_inv);
    if (penalty <= -epsilon_fl) DEBUG_PRINTF("\nKernel2: g_evaluate ERROR!#6");

    const int x0 = a[0];
    const int y0 = a[1];
    const int z0 = a[2];

    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    const float f000 = access_m_data(g->m_data, m_i, m_j, x0, y0, z0);
    const float f100 = access_m_data(g->m_data, m_i, m_j, x1, y0, z0);
    const float f010 = access_m_data(g->m_data, m_i, m_j, x0, y1, z0);
    const float f110 = access_m_data(g->m_data, m_i, m_j, x1, y1, z0);
    const float f001 = access_m_data(g->m_data, m_i, m_j, x0, y0, z1);
    const float f101 = access_m_data(g->m_data, m_i, m_j, x1, y0, z1);
    const float f011 = access_m_data(g->m_data, m_i, m_j, x0, y1, z1);
    const float f111 = access_m_data(g->m_data, m_i, m_j, x1, y1, z1);

    const float x = s[0];
    const float y = s[1];
    const float z = s[2];

    const float mx = 1 - x;
    const float my = 1 - y;
    const float mz = 1 - z;

    float f = f000 * mx * my * mz + f100 * x * my * mz + f010 * mx * y * mz + f110 * x * y * mz
              + f001 * mx * my * z + f101 * x * my * z + f011 * mx * y * z + f111 * x * y * z;

    if (deriv) {
        const float x_g = f000 * (-1) * my * mz + f100 * 1 * my * mz + f010 * (-1) * y * mz
                          + f110 * 1 * y * mz + f001 * (-1) * my * z + f101 * 1 * my * z
                          + f011 * (-1) * y * z + f111 * 1 * y * z;

        const float y_g = f000 * mx * (-1) * mz + f100 * x * (-1) * mz + f010 * mx * 1 * mz
                          + f110 * x * 1 * mz + f001 * mx * (-1) * z + f101 * x * (-1) * z
                          + f011 * mx * 1 * z + f111 * x * 1 * z;

        const float z_g = f000 * mx * my * (-1) + f100 * x * my * (-1) + f010 * mx * y * (-1)
                          + f110 * x * y * (-1) + f001 * mx * my * 1 + f101 * x * my * 1
                          + f011 * mx * y * 1 + f111 * x * y * 1;

        float gradient[3] = {x_g, y_g, z_g};

        curl_with_deriv(&f, gradient, v, epsilon_fl);

        float gradient_everywhere[3];

        for (int i = 0; i < 3; i++) {
            gradient_everywhere[i] = ((region[i] == 0) ? gradient[i] : 0);
            deriv[i] = g->m_factor[i] * gradient_everywhere[i] + slope * region[i];
        }
        return f + penalty;
    } else { /* none valid pointer */
        DEBUG_PRINTF("\nKernel2: g_evaluate ERROR!#7");
        curl_without_deriv(&f, v, epsilon_fl);
        return f + penalty;
    }
}

__device__ __forceinline__ float ig_eval_deriv(output_type_cuda_t *x, change_cuda_t *g,
                                               const float v, ig_cuda_t *ig_cuda_gpu,
                                               m_cuda_t *m_cuda_gpu, const float epsilon_fl) {
    float e = 0;
    int nat = num_atom_types(ig_cuda_gpu->atu);
    for (int i = 0; i < m_cuda_gpu->m_num_movable_atoms; i++) {
        int t = m_cuda_gpu->atoms[i].types[ig_cuda_gpu->atu];
        if (t >= nat) {
            for (int j = 0; j < 3; j++) m_cuda_gpu->minus_forces.coords[i][j] = 0;
            continue;
        }
        float deriv[3];

        e = e
            + g_evaluate(&ig_cuda_gpu->grids[t], m_cuda_gpu->m_coords.coords[i], ig_cuda_gpu->slope,
                         v, deriv, epsilon_fl);

        for (int j = 0; j < 3; j++) m_cuda_gpu->minus_forces.coords[i][j] = deriv[j];
    }
    return e;
}

__device__ __forceinline__ void quaternion_to_r3(const float *q, float *orientation_m) {
    /* Omit assert(quaternion_is_normalized(q)); */
    const float a = q[0];
    const float b = q[1];
    const float c = q[2];
    const float d = q[3];

    const float aa = a * a;
    const float ab = a * b;
    const float ac = a * c;
    const float ad = a * d;
    const float bb = b * b;
    const float bc = b * c;
    const float bd = b * d;
    const float cc = c * c;
    const float cd = c * d;
    const float dd = d * d;

    /* Omit assert(eq(aa + bb + cc + dd, 1)); */
    matrix_d tmp;
    mat_init(&tmp, 0); /* matrix_d with fixed dimension 3(here we treate this as
                          a regular matrix_d(not triangular matrix_d!)) */

    matrix_d_set_element(&tmp, 3, 0, 0, (aa + bb - cc - dd));
    matrix_d_set_element(&tmp, 3, 0, 1, 2 * (-ad + bc));
    matrix_d_set_element(&tmp, 3, 0, 2, 2 * (ac + bd));

    matrix_d_set_element(&tmp, 3, 1, 0, 2 * (ad + bc));
    matrix_d_set_element(&tmp, 3, 1, 1, (aa - bb + cc - dd));
    matrix_d_set_element(&tmp, 3, 1, 2, 2 * (-ab + cd));

    matrix_d_set_element(&tmp, 3, 2, 0, 2 * (-ac + bd));
    matrix_d_set_element(&tmp, 3, 2, 1, 2 * (ab + cd));
    matrix_d_set_element(&tmp, 3, 2, 2, (aa - bb - cc + dd));

    for (int i = 0; i < 9; i++) orientation_m[i] = tmp.data[i];
}

__device__ __forceinline__ void local_to_lab_direction(float *out, const float *local_direction,
                                                       const float *orientation_m) {
    out[0] = orientation_m[0] * local_direction[0] + orientation_m[3] * local_direction[1]
             + orientation_m[6] * local_direction[2];
    out[1] = orientation_m[1] * local_direction[0] + orientation_m[4] * local_direction[1]
             + orientation_m[7] * local_direction[2];
    out[2] = orientation_m[2] * local_direction[0] + orientation_m[5] * local_direction[1]
             + orientation_m[8] * local_direction[2];
}

__device__ __forceinline__ void local_to_lab(float *out, const float *origin,
                                             const float *local_coords,
                                             const float *orientation_m) {
    out[0] = origin[0]
             + (orientation_m[0] * local_coords[0] + orientation_m[3] * local_coords[1]
                + orientation_m[6] * local_coords[2]);
    out[1] = origin[1]
             + (orientation_m[1] * local_coords[0] + orientation_m[4] * local_coords[1]
                + orientation_m[7] * local_coords[2]);
    out[2] = origin[2]
             + (orientation_m[2] * local_coords[0] + orientation_m[5] * local_coords[1]
                + orientation_m[8] * local_coords[2]);
}

__device__ __forceinline__ void angle_to_quaternion2(float *out, const float *axis, float angle) {
    normalize_angle(&angle);
    float c = cos(angle / 2);
    float s = sin(angle / 2);
    out[0] = c;
    out[1] = s * axis[0];
    out[2] = s * axis[1];
    out[3] = s * axis[2];
}

__device__ __forceinline__ void set(const output_type_cuda_t *x, rigid_cuda_t *lig_rigid_gpu,
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
         current++) { /* current starts from 1 (namely starts from first child
                         node) */
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

__device__ __forceinline__ void p_eval_deriv(float *out, int type_pair_index, float r2,
                                             p_cuda_t *p_cuda_gpu, const float epsilon_fl) {
    const float cutoff_sqr = p_cuda_gpu->m_cutoff_sqr;
    if (r2 > cutoff_sqr)
        DEBUG_PRINTF(
            "\nkernel2: p_eval_deriv() ERROR!, r2 > Cutoff_sqr, r2=%f, "
            "cutoff_sqr=%f",
            r2, cutoff_sqr);

    p_m_data_cuda_t *tmp = &p_cuda_gpu->m_data[type_pair_index];
    float r2_factored = tmp->factor * r2;
    int i1 = (int)(r2_factored);
    int i2 = i1 + 1;
    float rem = r2_factored - i1;
    if (rem < -epsilon_fl) DEBUG_PRINTF("\nkernel2: p_eval_deriv() ERROR!");
    if (rem >= 1 + epsilon_fl) DEBUG_PRINTF("\nkernel2: p_eval_deriv() ERROR!");
    float p1[2] = {tmp->smooth[i1][0], tmp->smooth[i1][1]};
    if (i1 >= SMOOTH_SIZE) p1[0] = p1[1] = 0;
    float p2[2] = {tmp->smooth[i2][0], tmp->smooth[i2][1]};
    if (i2 >= SMOOTH_SIZE) p2[0] = p2[1] = 0;
    float e = p1[0] + rem * (p2[0] - p1[0]);
    float dor = p1[1] + rem * (p2[1] - p1[1]);
    out[0] = e;
    out[1] = dor;
}

__device__ __forceinline__ void curl(float *e, float *deriv, float v, const float epsilon_fl) {
    if (*e > 0 && not_max_gpu(v)) {
        float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
        (*e) = tmp * (*e);
        for (int i = 0; i < 3; i++) deriv[i] = deriv[i] * (tmp * tmp);
    }
}

__device__ __forceinline__ float eval_interacting_pairs_deriv(p_cuda_t *p_cuda_gpu, const float v,
                                                              const lig_pairs_cuda_t *pairs,
                                                              const m_coords_cuda_t *m_coords,
                                                              m_minus_forces_t *minus_forces,
                                                              const float epsilon_fl) {
    float e = 0;
    for (int i = 0; i < pairs->num_pairs; i++) {
        const int ip[3] = {pairs->type_pair_index[i], pairs->a[i], pairs->b[i]};
        int index = pairs->a[i] + pairs->b[i] * (pairs->b[i] + 1) / 2;
        float coords_b[3]
            = {m_coords->coords[ip[2]][0], m_coords->coords[ip[2]][1], m_coords->coords[ip[2]][2]};
        float coords_a[3]
            = {m_coords->coords[ip[1]][0], m_coords->coords[ip[1]][1], m_coords->coords[ip[1]][2]};
        float r[3]
            = {coords_b[0] - coords_a[0], coords_b[1] - coords_a[1], coords_b[2] - coords_a[2]};
        float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

        if (r2 < p_cuda_gpu->m_cutoff_sqr) {
            float tmp[2];
            p_eval_deriv(tmp, index, r2, p_cuda_gpu, epsilon_fl);
            float force[3] = {r[0] * tmp[1], r[1] * tmp[1], r[2] * tmp[1]};
            curl(&tmp[0], force, v, epsilon_fl);
            e += tmp[0];
            for (int j = 0; j < 3; j++) minus_forces->coords[ip[1]][j] -= force[j];
            for (int j = 0; j < 3; j++) minus_forces->coords[ip[2]][j] += force[j];
        }
    }
    return e;
}

template <typename T1, typename T2, typename T3>
__device__ __forceinline__ void product(T1 *res, const T2 *a, const T3 *b) {
    res[0] = a[1] * b[2] - a[2] * b[1];
    res[1] = a[2] * b[0] - a[0] * b[2];
    res[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ float find_change_index_read(const change_cuda_t *g, int index) {
    if (index < 3) return g->position[index];
    index -= 3;
    if (index < 3) return g->orientation[index];
    index -= 3;
    if (index < g->lig_torsion_size) return g->lig_torsion[index];
    DEBUG_PRINTF("\nKernel2:find_change_index_read() ERROR!"); /* Shouldn't be here */
}

__device__ __forceinline__ void find_change_index_write(change_cuda_t *g, int index, float data) {
    if (index < 3) {
        g->position[index] = data;
        return;
    }
    index -= 3;
    if (index < 3) {
        g->orientation[index] = data;
        return;
    }
    index -= 3;
    if (index < g->lig_torsion_size) {
        g->lig_torsion[index] = data;
        return;
    }
    DEBUG_PRINTF("\nKernel2:find_change_index_write() ERROR!"); /* Shouldn't be here */
}
