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


#include "kernel.h"
#include "math.h"
#include <vector>
/* Original Include files */
#include "monte_carlo.h"
#include "coords.h"
#include "mutate.h"
#include "quasi_newton.h"
#include "model.h"
#include "precalculate.h"
#include "cache.h"

// #define checkCUDA(ret) assert((ret) == cudaSuccess)

__host__
void checkCUDA(cudaError_t ret){
	if (ret != cudaSuccess){
		printf("Cuda error\n");
	}
}


//  __device__ __forceinline__
// float pown(const float base, const int n)
// {
// 	// if (n == 0)
// 		return 1;
//  	if (n == 1)
// 	 	return base;
//  	if (n % 2 == 0)
// 	 	return pown(base*base, n/2);
//  	else
// 	 	return pown(base*base, n/2) * base;
// }

/* Below based on mutate_cont.cpp */

 __device__ __forceinline__ void quaternion_increment(float* q, const float* rotation, float epsilon_fl);

 __device__ __forceinline__ void normalize_angle(float* x);

 __device__ __forceinline__
void output_type_cuda_init(output_type_cuda_t* out, const float* ptr) {
	for (int i = 0; i < 3; i++)out->position[i] = ptr[i];
	for (int i = 0; i < 4; i++)out->orientation[i] = ptr[i + 3];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)out->lig_torsion[i] = ptr[i + 3 + 4];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)out->flex_torsion[i] = ptr[i + 3 + 4 + MAX_NUM_OF_LIG_TORSION];
	out->lig_torsion_size = ptr[3 + 4 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION];
	//did not assign coords and e
}

 __device__ __forceinline__
void output_type_cuda_init_with_output(output_type_cuda_t* out_new, const output_type_cuda_t* out_old) {
	for (int i = 0; i < 3; i++)out_new->position[i] = out_old->position[i];
	for (int i = 0; i < 4; i++)out_new->orientation[i] = out_old->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)out_new->lig_torsion[i] = out_old->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)out_new->flex_torsion[i] = out_old->flex_torsion[i];
	out_new->lig_torsion_size = out_old->lig_torsion_size;
	//assign e but not coords
	out_new->e = out_old->e;
}

 __device__ __forceinline__
void output_type_cuda_increment(output_type_cuda_t* x, const change_cuda_t* c, float factor, float epsilon_fl) {
	// position increment
	for (int k = 0; k < 3; k++) x->position[k] += factor * c->position[k];
	// orientation increment
	float rotation[3];
	for (int k = 0; k < 3; k++) rotation[k] = factor * c->orientation[k];
	quaternion_increment(x->orientation, rotation, epsilon_fl);

	// torsion increment
	for (int k = 0; k < x->lig_torsion_size; k++) {
		float tmp = factor * c->lig_torsion[k];
		normalize_angle(&tmp);
		x->lig_torsion[k] += tmp;
		normalize_angle(&(x->lig_torsion[k]));
	}
}

 __device__ __forceinline__
float norm3(const float* a) {
	return sqrt(pow(a[0], 2) + pow(a[1], 2) + pow(a[2], 2));
}

 __device__ __forceinline__
void normalize_angle(float* x) {
	while (1) {
		if (*x >= -(M_PI) && *x <= (M_PI)) {
			break;
		}
		else if (*x > 3 * (M_PI)) {
			float n = (*x - (M_PI)) / (2 * (M_PI));
			*x -= 2 * (M_PI) * ceil(n);
		}
		else if (*x < 3 * -(M_PI)) {
			float n = (-*x - (M_PI)) / (2 * (M_PI));
			*x += 2 * (M_PI) * ceil(n);
		}
		else if (*x > (M_PI)) {
			*x -= 2 * (M_PI);
		}
		else if (*x < -(M_PI)) {
			*x += 2 * (M_PI);
		}
		else {
			break;
		}
	}
}

 __device__ __forceinline__
bool quaternion_is_normalized(float* q) {
	float q_pow = pow(q[0], 2) + pow(q[1], 2) + pow(q[2], 2) + pow(q[3], 2);
	float sqrt_q_pow = sqrt(q_pow);
	return (q_pow - 1 < 0.001) && (sqrt_q_pow - 1 < 0.001);
}

 __device__ __forceinline__
void angle_to_quaternion(float* q, const float* rotation, float epsilon_fl) {
	float angle = norm3(rotation);
	if (angle > epsilon_fl) {
		float axis[3] = { rotation[0] / angle, rotation[1] / angle ,rotation[2] / angle };
		// if (norm3(axis) - 1 >= 0.001)printf("\nmutate: angle_to_quaternion() ERROR!"); // Replace assert(eq(axis.norm(), 1));
		normalize_angle(&angle);
		float c = cos(angle / 2);
		float s = sin(angle / 2);
		q[0] = c; q[1] = s * axis[0]; q[2] = s * axis[1]; q[3] = s * axis[2];
		return;
	}
	q[0] = 1; q[1] = 0; q[2] = 0; q[3] = 0;
	return;
}

// quaternion multiplication
 __device__ __forceinline__
void angle_to_quaternion_multi(float* qa, const float* qb) {
	float tmp[4] = { qa[0],qa[1],qa[2],qa[3] };
	qa[0] = tmp[0] * qb[0] - tmp[1] * qb[1] - tmp[2] * qb[2] - tmp[3] * qb[3];
	qa[1] = tmp[0] * qb[1] + tmp[1] * qb[0] + tmp[2] * qb[3] - tmp[3] * qb[2];
	qa[2] = tmp[0] * qb[2] - tmp[1] * qb[3] + tmp[2] * qb[0] + tmp[3] * qb[1];
	qa[3] = tmp[0] * qb[3] + tmp[1] * qb[2] - tmp[2] * qb[1] + tmp[3] * qb[0];
}

 __device__ __forceinline__
void quaternion_normalize_approx(float* q, float epsilon_fl) {
	const float s = pow(q[0], 2) + pow(q[1], 2) + pow(q[2], 2) + pow(q[3], 2);
	// Omit one assert()
	if (fabs(s - 1) < TOLERANCE)
		;
	else {
		const float a = sqrt(s);
		//if (a <= epsilon_fl) printf("\nmutate: quaternion_normalize_approx ERROR!"); // Replace assert(a > epsilon_fl);
		for (int i = 0; i < 4; i++)q[i] *= (1 / a);
		//if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_normalize_approx() ERROR!");// Replace assert(quaternion_is_normalized(q));
	}
}

 __device__ __forceinline__
void quaternion_increment(float* q, const float* rotation, float epsilon_fl) {
	//if (quaternion_is_normalized(q) != true)printf("\nmutate: quaternion_increment() ERROR!"); // Replace assert(quaternion_is_normalized(q))
	float q_old[4] = { q[0],q[1],q[2],q[3] };
	angle_to_quaternion(q, rotation, epsilon_fl);
	angle_to_quaternion_multi(q, q_old);
	quaternion_normalize_approx(q, epsilon_fl);
}


 __device__ __forceinline__
float vec_distance_sqr(float* a, float* b) {
	return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2);
}

 __device__ __forceinline__
float gyration_radius(				int				m_lig_begin,
									int				m_lig_end,
						const		atom_cuda_t*		atoms,
						const		m_coords_cuda_t*	m_coords_gpu,
						const		float*			m_lig_node_origin
) {
	float acc = 0;
	int counter = 0;
	float origin[3] = { m_lig_node_origin[0], m_lig_node_origin[1], m_lig_node_origin[2] };
	for (int i = m_lig_begin; i < m_lig_end; i++) {
		float current_coords[3] = { m_coords_gpu->coords[i][0], m_coords_gpu->coords[i][1], m_coords_gpu->coords[i][2] };
		if (atoms[i].types[0] != EL_TYPE_H) { // for el, we use the first element (atoms[i].types[0])
			acc += vec_distance_sqr(current_coords, origin);
			++counter;
		}
	}
	return (counter > 0) ? sqrt(acc / counter) : 0;
}

 __device__ __forceinline__
void mutate_conf_cuda(const	int	step, const	int	num_steps, output_type_cuda_t *c, int *random_int_map_gpu,
			float random_inside_sphere_map_gpu[][3], float*	random_fl_pi_map_gpu,
			const int	m_lig_begin,
			const int	m_lig_end,
			const atom_cuda_t* atoms,
			const m_coords_cuda_t* m_coords_gpu,
			const float*		m_lig_node_origin_gpu,
			const float			epsilon_fl,
			const float			amplitude
) {

	int index = step; // global index (among all exhaus)
	int which = random_int_map_gpu[index];
	int lig_torsion_size = c->lig_torsion_size;
	int flex_torsion_size = 0; // FIX? 20210727
		if (which == 0) {
			for (int i = 0; i < 3; i++)
				c->position[i] += amplitude * random_inside_sphere_map_gpu[index][i];
			return;
		}
		--which;
		if (which == 0) {
			float gr = gyration_radius(m_lig_begin, m_lig_end, atoms, m_coords_gpu, m_lig_node_origin_gpu);
			if (gr > epsilon_fl) {
				float rotation[3];
				for (int i = 0; i < 3; i++)rotation[i] = amplitude / gr * random_inside_sphere_map_gpu[index][i];
				quaternion_increment(c->orientation, rotation, epsilon_fl);
			}
			return;
		}
		--which;
		if (which < lig_torsion_size) { c->lig_torsion[which] = random_fl_pi_map_gpu[index]; return; }
		which -= lig_torsion_size;

	if (flex_torsion_size != 0) {
		if (which < flex_torsion_size) { c->flex_torsion[which] = random_fl_pi_map_gpu[index]; return; }
		which -= flex_torsion_size;
	}
}

/*  Above based on mutate_conf.cpp */

/* Below based on matrix.cpp */

// symmetric matrix_d (only half of it are stored)
typedef struct {
	float data[MAX_HESSIAN_MATRIX_D_SIZE];
	int dim;
}matrix_d;

 __device__ __forceinline__
void matrix_d_init(matrix_d* m, int dim, float fill_data) {
	m->dim = dim;
	if ((dim * (dim + 1) / 2) > MAX_HESSIAN_MATRIX_D_SIZE)printf("\nnmatrix_d: matrix_d_init() ERROR!");
	// ((dim * (dim + 1) / 2)*sizeof(float)); // symmetric matrix_d
	for (int i = 0; i < (dim * (dim + 1) / 2); i++)m->data[i] = fill_data;
	for (int i = (dim * (dim + 1) / 2); i < MAX_HESSIAN_MATRIX_D_SIZE; i++)m->data[i] = 0;// Others will be 0
}

// as rugular 3x3 matrix_d
 __device__ __forceinline__
void mat_init(matrix_d* m, float fill_data) {
	m->dim = 3; // fixed to 3x3 matrix_d
	if (9 > MAX_HESSIAN_MATRIX_D_SIZE)printf("\nnmatrix_d: mat_init() ERROR!");
	for (int i = 0; i < 9; i++)m->data[i] = fill_data;
}

 __device__ __forceinline__
void matrix_d_set_diagonal(matrix_d* m, float fill_data) {
	for (int i = 0; i < m->dim; i++) {
		m->data[i + i * (i + 1) / 2] = fill_data;
	}
}

// as regular matrix_d
 __device__ __forceinline__
void matrix_d_set_element(matrix_d* m, int dim, int x, int y, float fill_data) {
	m->data[x + y * dim] = fill_data;
}

 __device__ __forceinline__
void matrix_d_set_element_tri(matrix_d* m, int x, int y, float fill_data) {
	m->data[x + y*(y+1)/2] = fill_data;
}
 __device__ __forceinline__
int tri_index(int n, int i, int j) {
	if (j >= n || i > j)printf("\nmatrix_d: tri_index ERROR!");
	return i + j * (j + 1) / 2;
}

 __device__ __forceinline__
int index_permissive(const matrix_d* m, int i, int j) {
	return (i < j) ? tri_index(m->dim, i, j) : tri_index(m->dim, j, i);
}

/* Above based on matrix_d.cpp */

/* Below based on quasi_newton.cpp */

#define EL_TYPE_SIZE 11
#define AD_TYPE_SIZE 20
#define XS_TYPE_SIZE 17
#define SY_TYPE_SIZE 18



 __device__ __forceinline__
void change_cuda_init(change_cuda_t* g, const float* ptr) {
	for (int i = 0; i < 3; i++)g->position[i] = ptr[i];
	for (int i = 0; i < 3; i++)g->orientation[i] = ptr[i + 3];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)g->lig_torsion[i] = ptr[i + 3 + 3];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)g->flex_torsion[i] = ptr[i + 3 + 3 + MAX_NUM_OF_LIG_TORSION];
	g->lig_torsion_size = ptr[3 + 3 + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION];
}

 __device__ __forceinline__
void change_cuda_init_with_change(change_cuda_t* g_new, const change_cuda_t* g_old) {
	for (int i = 0; i < 3; i++)g_new->position[i] = g_old->position[i];
	for (int i = 0; i < 3; i++)g_new->orientation[i] = g_old->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)g_new->lig_torsion[i] = g_old->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)g_new->flex_torsion[i] = g_old->flex_torsion[i];
	g_new->lig_torsion_size = g_old->lig_torsion_size;
}

void print_ouput_type(output_type_cuda_t* x, int torsion_size) {
	for (int i = 0; i < 3; i++)printf("\nx.position[%d] = %0.16f", i, x->position[i]);
	for (int i = 0; i < 4; i++)printf("\nx.orientation[%d] = %0.16f", i, x->orientation[i]);
	for (int i = 0; i < torsion_size; i++)printf("\n x.torsion[%d] = %0.16f", i, x->lig_torsion[i]);
	printf("\n x.torsion_size = %f", x->lig_torsion_size);
}

void print_change(change_cuda_t* g, int torsion_size) {
	for (int i = 0; i < 3; i++)printf("\ng.position[%d] = %0.16f", i, g->position[i]);
	for (int i = 0; i < 3; i++)printf("\ng.orientation[%d] = %0.16f", i, g->orientation[i]);
	for (int i = 0; i < torsion_size; i++)printf("\ng.torsion[%d] = %0.16f", i, g->lig_torsion[i]);
	printf("\ng.torsion_size = %f", g->lig_torsion_size);
}

 __device__ __forceinline__
int num_atom_types(int atu) {
	switch (atu) {
	case 0: return EL_TYPE_SIZE;
	case 1: return AD_TYPE_SIZE;
	case 2: return XS_TYPE_SIZE;
	case 3: return SY_TYPE_SIZE;
	default: printf("Kernel1:num_atom_types() ERROR!"); return INFINITY;
	}
}

 __device__ __forceinline__
void elementwise_product(float* out, const float* a, const float* b) {
	out[0] = a[0] * b[0];
	out[1] = a[1] * b[1];
	out[2] = a[2] * b[2];
}

 __device__ __forceinline__
float elementwise_product_sum(const float* a, const float* b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

 __device__ __forceinline__
float access_m_data(float* m_data, int m_i, int m_j, int i, int j, int k) {
	return m_data[i + m_i * (j + m_j * k)];
}

 __device__ __forceinline__
bool not_max(float x) {
	return (x < 0.1 * INFINITY); /* Problem: replace max_fl with INFINITY? */
}

 __device__ __forceinline__
void curl_with_deriv(float* e, float* deriv, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
		for (int i = 0; i < 3; i++)deriv[i] *= pow(tmp, 2);
	}
}

 __device__ __forceinline__
void curl_without_deriv(float* e, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		*e *= tmp;
	}
}

 __device__ __forceinline__
float g_evaluate(	grid_cuda_t*	g,
					const				float*		m_coords,				/* double[3] */
					const				float		slope,				/* double */
					const				float		v,					/* double */
										float*		deriv,				/* double[3] */
					const				float		epsilon_fl
) {
	int m_i = g->m_i;
	int m_j = g->m_j;
	int m_k = g->m_k;
	if(m_i * m_j * m_k == 0)printf("\nkernel2: g_evaluate ERROR!#1");
	float tmp_vec[3] = { m_coords[0] - g->m_init[0],m_coords[1] - g->m_init[1] ,m_coords[2] - g->m_init[2] };
	float tmp_vec2[3] = { g->m_factor[0],g->m_factor[1] ,g->m_factor[2] };
	float s[3];
	elementwise_product(s, tmp_vec, tmp_vec2);

	float miss[3] = { 0,0,0 };
	int region[3];
	int a[3];
	int m_data_dims[3] = { m_i,m_j,m_k };
	for (int i = 0; i < 3; i++){
		if (s[i] < 0) {
			miss[i] = -s[i];
			region[i] = -1;
			a[i] = 0;
			s[i] = 0;
		}
		else if (s[i] >= g->m_dim_fl_minus_1[i]) {
			miss[i] = s[i] - g->m_dim_fl_minus_1[i];
			region[i] = 1;
			if (m_data_dims[i] < 2)printf("\nKernel2: g_evaluate ERROR!#2");
			a[i] = m_data_dims[i] - 2;
			s[i] = 1;
		}
		else {
			region[i] = 0;
			a[i] = (int)s[i];
			s[i] -= a[i];
		}
		if (s[i] < 0)
            printf("\nKernel2: g_evaluate ERROR!#3");
		if (s[i] > 1)
            printf("\nKernel2: g_evaluate ERROR!#4");
		if (a[i] < 0)
            printf("\nKernel2: g_evaluate ERROR!#5");
		if (a[i] + 1 >= m_data_dims[i])printf("\nKernel2: g_evaluate ERROR!#5");
	}

	float tmp_m_factor_inv[3] = { g->m_factor_inv[0],g->m_factor_inv[1],g->m_factor_inv[2] };
	const float penalty = slope * elementwise_product_sum(miss, tmp_m_factor_inv);
	if (penalty <= -epsilon_fl)printf("\nKernel2: g_evaluate ERROR!#6");

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

	float f =
		f000 * mx * my * mz +
		f100 * x  * my * mz +
		f010 * mx * y  * mz +
		f110 * x  * y  * mz +
		f001 * mx * my * z	+
		f101 * x  * my * z	+
		f011 * mx * y  * z	+
		f111 * x  * y  * z  ;

	if (deriv) {
		const float x_g =
			f000 * (-1) * my * mz +
			f100 *   1  * my * mz +
			f010 * (-1) * y  * mz +
			f110 *	 1  * y  * mz +
			f001 * (-1) * my * z  +
			f101 *   1  * my * z  +
			f011 * (-1) * y  * z  +
			f111 *   1  * y  * z  ;


		const float y_g =
			f000 * mx * (-1) * mz +
			f100 * x  * (-1) * mz +
			f010 * mx *   1  * mz +
			f110 * x  *   1  * mz +
			f001 * mx * (-1) * z  +
			f101 * x  * (-1) * z  +
			f011 * mx *   1  * z  +
			f111 * x  *   1  * z  ;


		const float z_g =
			f000 * mx * my * (-1) +
			f100 * x  * my * (-1) +
			f010 * mx * y  * (-1) +
			f110 * x  * y  * (-1) +
			f001 * mx * my *   1  +
			f101 * x  * my *   1  +
			f011 * mx * y  *   1  +
			f111 * x  * y  *   1  ;

		float gradient[3] = { x_g, y_g, z_g };

		curl_with_deriv(&f, gradient, v, epsilon_fl);

		float gradient_everywhere[3];

		for (int i = 0; i < 3; i++) {
			gradient_everywhere[i] = ((region[i] == 0) ? gradient[i] : 0);
			deriv[i] = g->m_factor[i] * gradient_everywhere[i] + slope * region[i];
		}
		return f + penalty;
	}
	else {  /* none valid pointer */
		printf("\nKernel2: g_evaluate ERROR!#7");
		curl_without_deriv(&f, v, epsilon_fl);
		return f + penalty;
	}
}

 __device__ __forceinline__
float ig_eval_deriv(						output_type_cuda_t*		x,
											change_cuda_t*			g,
						const				float				v,
									ig_cuda_t*				ig_cuda_gpu,
											m_cuda_t*				m_cuda_gpu,
						const				float				epsilon_fl
) {
	float e = 0;
	int nat = num_atom_types(ig_cuda_gpu->atu);
	for (int i = 0; i < m_cuda_gpu->m_num_movable_atoms; i++) {
		int t = m_cuda_gpu->atoms[i].types[ig_cuda_gpu->atu];
		if (t >= nat) {
			for (int j = 0; j < 3; j++)m_cuda_gpu->minus_forces.coords[i][j] = 0;
			continue;
		}
		float deriv[3];

		e = e + g_evaluate(&ig_cuda_gpu->grids[t], m_cuda_gpu->m_coords.coords[i], ig_cuda_gpu->slope, v, deriv, epsilon_fl);

		for (int j = 0; j < 3; j++) m_cuda_gpu->minus_forces.coords[i][j] = deriv[j];
	}
	return e;
}

 __device__ __forceinline__
void quaternion_to_r3(const float* q, float* orientation_m) {
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
	mat_init(&tmp, 0); /* matrix_d with fixed dimension 3(here we treate this as a regular matrix_d(not triangular matrix_d!)) */

	matrix_d_set_element(&tmp, 3, 0, 0,		(aa + bb - cc - dd)	);
	matrix_d_set_element(&tmp, 3, 0, 1, 2 *	(-ad + bc)			);
	matrix_d_set_element(&tmp, 3, 0, 2, 2 *	(ac + bd)			);

	matrix_d_set_element(&tmp, 3, 1, 0, 2 *	(ad + bc)			);
	matrix_d_set_element(&tmp, 3, 1, 1,		(aa - bb + cc - dd)	);
	matrix_d_set_element(&tmp, 3, 1, 2, 2 *	(-ab + cd)			);

	matrix_d_set_element(&tmp, 3, 2, 0, 2 *	(-ac + bd)			);
	matrix_d_set_element(&tmp, 3, 2, 1, 2 *	(ab + cd)			);
	matrix_d_set_element(&tmp, 3, 2, 2,		(aa - bb - cc + dd)	);

	for (int i = 0; i < 9; i++) orientation_m[i] = tmp.data[i];
}

 __device__ __forceinline__
void local_to_lab_direction(			float* out,
									const	float* local_direction,
									const	float* orientation_m
) {
	out[0] =	orientation_m[0] * local_direction[0] +
				orientation_m[3] * local_direction[1] +
				orientation_m[6] * local_direction[2];
	out[1] =	orientation_m[1] * local_direction[0] +
				orientation_m[4] * local_direction[1] +
				orientation_m[7] * local_direction[2];
	out[2] =	orientation_m[2] * local_direction[0] +
				orientation_m[5] * local_direction[1] +
				orientation_m[8] * local_direction[2];
}

 __device__ __forceinline__
void local_to_lab(						float*		out,
							const				float*		origin,
							const				float*		local_coords,
							const				float*		orientation_m
) {
	out[0] = origin[0] + (	orientation_m[0] * local_coords[0] +
							orientation_m[3] * local_coords[1] +
							orientation_m[6] * local_coords[2]
							);
	out[1] = origin[1] + (	orientation_m[1] * local_coords[0] +
							orientation_m[4] * local_coords[1] +
							orientation_m[7] * local_coords[2]
							);
	out[2] = origin[2] + (	orientation_m[2] * local_coords[0] +
							orientation_m[5] * local_coords[1] +
							orientation_m[8] * local_coords[2]
							);
}

 __device__ __forceinline__
void angle_to_quaternion2(				float*		out,
									const		float*		axis,
												float		angle
) {
	// if (norm3(axis) - 1 >= 0.001)
	// 	printf("\nkernel2: angle_to_quaternion() ERROR!"); /* Replace assert(eq(axis.norm(), 1)); */
	normalize_angle(&angle);
	float c = cos(angle / 2);
	float s = sin(angle / 2);
	out[0] = c;
	out[1] = s * axis[0];
	out[2] = s * axis[1];
	out[3] = s * axis[2];
}

 __device__ __forceinline__
void set(	const				output_type_cuda_t* x,
								rigid_cuda_t*		lig_rigid_gpu,
								m_coords_cuda_t*		m_coords_gpu,
			const				atom_cuda_t*		atoms,
			const				int				m_num_movable_atoms,
			const				float			epsilon_fl
) {

	for (int i = 0; i < 3; i++) lig_rigid_gpu->origin[0][i] = x->position[i];
	for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[0][i] = x->orientation[i];
	quaternion_to_r3(lig_rigid_gpu->orientation_q[0], lig_rigid_gpu->orientation_m[0]); /* set orientation_m */

	int begin = lig_rigid_gpu->atom_range[0][0];
	int end =	lig_rigid_gpu->atom_range[0][1];
	for (int i = begin; i < end; i++) {
		local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[0], atoms[i].coords, lig_rigid_gpu->orientation_m[0]);
	}
	/* ************* end node.set_conf ************* */

	/* ************* branches_set_conf ************* */
	/* update nodes in depth-first order */
	for (int current = 1; current < lig_rigid_gpu->num_children + 1; current++) { /* current starts from 1 (namely starts from first child node) */
		int parent = lig_rigid_gpu->parent[current];
		float torsion = x->lig_torsion[current - 1]; /* torsions are all related to child nodes */
		local_to_lab(	lig_rigid_gpu->origin[current],
						lig_rigid_gpu->origin[parent],
						lig_rigid_gpu->relative_origin[current],
						lig_rigid_gpu->orientation_m[parent]
						);
		local_to_lab_direction(	lig_rigid_gpu->axis[current],
								lig_rigid_gpu->relative_axis[current],
								lig_rigid_gpu->orientation_m[parent]
								);
		float tmp[4];
		float parent_q[4] = {	lig_rigid_gpu->orientation_q[parent][0],
								lig_rigid_gpu->orientation_q[parent][1] ,
								lig_rigid_gpu->orientation_q[parent][2] ,
								lig_rigid_gpu->orientation_q[parent][3] };
		float current_axis[3] = {	lig_rigid_gpu->axis[current][0],
									lig_rigid_gpu->axis[current][1],
									lig_rigid_gpu->axis[current][2] };

		angle_to_quaternion2(tmp, current_axis, torsion);
		angle_to_quaternion_multi(tmp, parent_q);
		quaternion_normalize_approx(tmp, epsilon_fl);

		for (int i = 0; i < 4; i++) lig_rigid_gpu->orientation_q[current][i] = tmp[i]; /* set orientation_q */
		quaternion_to_r3(lig_rigid_gpu->orientation_q[current], lig_rigid_gpu->orientation_m[current]); /* set orientation_m */

		/* set coords */
		begin = lig_rigid_gpu->atom_range[current][0];
		end =	lig_rigid_gpu->atom_range[current][1];
		for (int i = begin; i < end; i++) {
			local_to_lab(m_coords_gpu->coords[i], lig_rigid_gpu->origin[current], atoms[i].coords, lig_rigid_gpu->orientation_m[current]);
		}
	}
	/* ************* end branches_set_conf ************* */
}

 __device__ __forceinline__
void p_eval_deriv(						float*		out,
										int			type_pair_index,
										float		r2,
									p_cuda_t*		p_cuda_gpu,
					const				float		epsilon_fl
) {
	// const float cutoff_sqr = p_cuda_gpu->m_cutoff_sqr;
	// if(r2 > cutoff_sqr) printf("\nkernel2: p_eval_deriv() ERROR!");
	p_m_data_cuda_t* tmp = &p_cuda_gpu->m_data[type_pair_index];
	float r2_factored = tmp->factor * r2;
	// if (r2_factored + 1 >= SMOOTH_SIZE) printf("\nkernel2: p_eval_deriv() ERROR!");
	int i1 = (int)(r2_factored);
	int i2 = i1 + 1;
	// if (i1 >= SMOOTH_SIZE || i1 < 0)printf("\n kernel2: p_eval_deriv() ERROR!");
	// if (i2 >= SMOOTH_SIZE || i2 < 0)printf("\n : p_eval_deriv() ERROR!");
	float rem = r2_factored - i1;
	// if (rem < -epsilon_fl)printf("\nkernel2: p_eval_deriv() ERROR!");
	// if (rem >= 1 + epsilon_fl)printf("\nkernel2: p_eval_deriv() ERROR!");
	float p1[2] = { tmp->smooth[i1][0], tmp->smooth[i1][1] };
	float p2[2] = { tmp->smooth[i2][0], tmp->smooth[i2][1] };
	float e = p1[0] + rem * (p2[0] - p1[0]);
	float dor = p1[1] + rem * (p2[1] - p1[1]);
	out[0] = e;
	out[1] = dor;
}

 __device__ __forceinline__
void curl(float* e, float* deriv, float v, const float epsilon_fl) {
	if (*e > 0 && not_max(v)) {
		float tmp = (v < epsilon_fl) ? 0 : (v / (v + *e));
		(*e) = tmp * (*e);
		for (int i = 0; i < 3; i++)deriv[i] = deriv[i] * (tmp * tmp);
	}
}

 __device__ __forceinline__
float eval_interacting_pairs_deriv(			p_cuda_t*			p_cuda_gpu,
									const				float			v,
									const				lig_pairs_cuda_t*   pairs,
									const			 	m_coords_cuda_t*		m_coords,
														m_minus_forces_t* minus_forces,
									const				float			epsilon_fl
) {
	float e = 0;

	for (int i = 0; i < pairs->num_pairs; i++) {
		const int ip[3] = { pairs->type_pair_index[i], pairs->a[i] ,pairs->b[i] };
		float coords_b[3] = { m_coords->coords[ip[2]][0], m_coords->coords[ip[2]][1], m_coords->coords[ip[2]][2] };
		float coords_a[3] = { m_coords->coords[ip[1]][0], m_coords->coords[ip[1]][1], m_coords->coords[ip[1]][2] };
		float r[3] = { coords_b[0] - coords_a[0], coords_b[1] - coords_a[1] ,coords_b[2] - coords_a[2] };
		float r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

		if (r2 < p_cuda_gpu->m_cutoff_sqr) {
			float tmp[2];
			p_eval_deriv(tmp, ip[0], r2, p_cuda_gpu, epsilon_fl);
			float force[3] = { r[0] * tmp[1], r[1] * tmp[1] ,r[2] * tmp[1] };
			curl(&tmp[0], force, v, epsilon_fl);
			e += tmp[0];
			for (int j = 0; j < 3; j++)minus_forces->coords[ip[1]][j] -= force[j];
			for (int j = 0; j < 3; j++)minus_forces->coords[ip[2]][j] += force[j];
		}
	}
	return e;
}

 __device__ __forceinline__
void product(float* res, const float*a,const float*b) {
	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = a[2] * b[0] - a[0] * b[2];
	res[2] = a[0] * b[1] - a[1] * b[0];
}

 __device__ __forceinline__
void POT_deriv(	const					m_minus_forces_t* minus_forces,
				const					rigid_cuda_t*		lig_rigid_gpu,
				const					m_coords_cuda_t*		m_coords,
										change_cuda_t*		g
) {
	int num_torsion = lig_rigid_gpu->num_children;
	int num_rigid = num_torsion + 1;
	float position_derivative_tmp[MAX_NUM_OF_RIGID][3];
	float position_derivative[MAX_NUM_OF_RIGID][3];
	float orientation_derivative_tmp[MAX_NUM_OF_RIGID][3];
	float orientation_derivative[MAX_NUM_OF_RIGID][3];
	float torsion_derivative[MAX_NUM_OF_RIGID]; /* torsion_derivative[0] has no meaning(root node has no torsion) */

	for (int i = 0; i < num_rigid; i++) {
		int begin = lig_rigid_gpu->atom_range[i][0];
		int end = lig_rigid_gpu->atom_range[i][1];
		for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] = 0;
		for (int k = 0; k < 3; k++)orientation_derivative_tmp[i][k] = 0;
		for (int j = begin; j < end; j++) {
			for (int k = 0; k < 3; k++)position_derivative_tmp[i][k] += minus_forces->coords[j][k];

			float tmp1[3] = {	m_coords->coords[j][0] - lig_rigid_gpu->origin[i][0],
								m_coords->coords[j][1] - lig_rigid_gpu->origin[i][1],
								m_coords->coords[j][2] - lig_rigid_gpu->origin[i][2] };
			float tmp2[3] = {  minus_forces->coords[j][0],
								minus_forces->coords[j][1],
								minus_forces->coords[j][2] };
			float tmp3[3];
			product(tmp3, tmp1, tmp2);
			for (int k = 0; k < 3; k++)
                orientation_derivative_tmp[i][k] += tmp3[k];
		}
	}

	/* position_derivative  */
	for (int i = num_rigid - 1; i >= 0; i--) { /* from bottom to top */
		for (int k = 0; k < 3; k++)position_derivative[i][k] = position_derivative_tmp[i][k];
		/* looking for chidren node */
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i][j] == true) {
				for (int k = 0; k < 3; k++)position_derivative[i][k] += position_derivative[j][k]; /* self+children node */
			}
		}
	}

	/* orientation_derivetive */
	for (int i = num_rigid - 1; i >= 0; i--) { /* from bottom to top */
		for (int k = 0; k < 3; k++)orientation_derivative[i][k] = orientation_derivative_tmp[i][k];
		/* looking for chidren node */
		for (int j = 0; j < num_rigid; j++) {
			if (lig_rigid_gpu->children_map[i][j] == true) { /* self + children node + product */
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += orientation_derivative[j][k];
				float product_out[3];
				float origin_temp[3] = {	lig_rigid_gpu->origin[j][0] - lig_rigid_gpu->origin[i][0],
											lig_rigid_gpu->origin[j][1] - lig_rigid_gpu->origin[i][1],
											lig_rigid_gpu->origin[j][2] - lig_rigid_gpu->origin[i][2] };
				product(product_out, origin_temp, position_derivative[j]);
				for (int k = 0; k < 3; k++)orientation_derivative[i][k] += product_out[k];
			}
		}
	}

	/* torsion_derivative */
	for (int i = num_rigid - 1; i >= 0; i--) {
		float sum = 0;
		for (int j = 0; j < 3; j++) sum += orientation_derivative[i][j] * lig_rigid_gpu->axis[i][j];
		torsion_derivative[i] = sum;
	}

	for (int k = 0; k < 3; k++)	g->position[k] = position_derivative[0][k];
	for (int k = 0; k < 3; k++) g->orientation[k] = orientation_derivative[0][k];
	for (int k = 0; k < num_torsion; k++) g->lig_torsion[k] = torsion_derivative[k + 1];
}

 __device__ __forceinline__
float m_eval_deriv(					output_type_cuda_t*		c,
										change_cuda_t*			g,
										m_cuda_t*				m_cuda_gpu,
								p_cuda_t*				p_cuda_gpu,
								ig_cuda_t*				ig_cuda_gpu,
					const	float*				v,
					const				float				epsilon_fl
) {
	set(c, &m_cuda_gpu->ligand.rigid, &m_cuda_gpu->m_coords, m_cuda_gpu->atoms, m_cuda_gpu->m_num_movable_atoms, epsilon_fl);

	float e = ig_eval_deriv(	c,
								g,
								v[1],
								ig_cuda_gpu,
								m_cuda_gpu,
								epsilon_fl
							);

	e += eval_interacting_pairs_deriv(	p_cuda_gpu,
										v[0],
										&m_cuda_gpu->ligand.pairs,
										&m_cuda_gpu->m_coords,
										&m_cuda_gpu->minus_forces,
										epsilon_fl
									);

	POT_deriv(&m_cuda_gpu->minus_forces, &m_cuda_gpu->ligand.rigid, &m_cuda_gpu->m_coords, g);

	return e;
}


 __device__ __forceinline__
float find_change_index_read(const change_cuda_t* g, int index) {
	if (index < 3) return g->position[index];
	index -= 3;
	if (index < 3) return g->orientation[index];
	index -= 3;
	if (index < g->lig_torsion_size) return g->lig_torsion[index];
	printf("\nKernel2:find_change_index_read() ERROR!"); /* Shouldn't be here */
}

 __device__ __forceinline__
void find_change_index_write(change_cuda_t* g, int index, float data) {
	if (index < 3) { g->position[index] = data; return; }
	index -= 3;
	if (index < 3) { g->orientation[index] = data; return; }
	index -= 3;
	if (index < g->lig_torsion_size) { g->lig_torsion[index] = data; return; }
	printf("\nKernel2:find_change_index_write() ERROR!"); /* Shouldn't be here */
}

 __device__ __forceinline__
void minus_mat_vec_product(	const		matrix_d*		h,
							const		change_cuda_t*	in,
										change_cuda_t*  out
) {
	int n = h->dim;
	for (int i = 0; i < n; i++) {
		float sum = 0;
		for (int j = 0; j < n; j++) {
			sum += h->data[index_permissive(h, i, j)] * find_change_index_read(in, j);
		}
		find_change_index_write(out, i, -sum);
	}
}


 __device__ __forceinline__
float scalar_product(	const	change_cuda_t*			a,
								const	change_cuda_t*			b,
								int							n
) {
	float tmp = 0;
	for (int i = 0; i < n; i++) {
		tmp += find_change_index_read(a, i) * find_change_index_read(b, i);
	}
	return tmp;
}

 __device__ __forceinline__
float line_search(					 	m_cuda_t*				m_cuda_gpu,
								p_cuda_t*				p_cuda_gpu,
								ig_cuda_t*				ig_cuda_gpu,
										int					n,
					const				output_type_cuda_t*		x,
					const				change_cuda_t*			g,
					const				float				f0,
					const				change_cuda_t*			p,
										output_type_cuda_t*		x_new,
										change_cuda_t*			g_new,
										float*				f1,
					const				float				epsilon_fl,
					const	float*				hunt_cap
) {
	const float c0 = 0.0001;
	const int max_trials = 10;
	const float multiplier = 0.5;
	float alpha = 1;

	const float pg = scalar_product(p, g, n);

	for (int trial = 0; trial < max_trials; trial++) {

		output_type_cuda_init_with_output(x_new, x);

		output_type_cuda_increment(x_new, p, alpha, epsilon_fl);

		*f1 =  m_eval_deriv(x_new,
							g_new,
							m_cuda_gpu,
							p_cuda_gpu,
							ig_cuda_gpu,
							hunt_cap,
							epsilon_fl
							);

		if (*f1 - f0 < c0 * alpha * pg)
			break;
		alpha *= multiplier;
	}
	return alpha;
}

 __device__ __forceinline__
bool bfgs_update(			matrix_d*			h,
					const	change_cuda_t*		p,
					const	change_cuda_t*		y,
					const	float			alpha,
					const	float			epsilon_fl
) {

	const float yp = scalar_product(y, p, h->dim);

	if (alpha * yp < epsilon_fl) return false;
	change_cuda_t minus_hy;
	change_cuda_init_with_change(&minus_hy, y);
	minus_mat_vec_product(h, y, &minus_hy);
	const float yhy = -scalar_product(y, &minus_hy, h->dim);
	const float r = 1 / (alpha * yp);
	const int n = 6 + p->lig_torsion_size;

	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			float tmp = alpha * r * (find_change_index_read(&minus_hy, i) * find_change_index_read(p, j)
									+ find_change_index_read(&minus_hy, j) * find_change_index_read(p, i)) +
									+alpha * alpha * (r * r * yhy + r) * find_change_index_read(p, i) * find_change_index_read(p, j);

			h->data[i + j * (j + 1) / 2] += tmp;
		}
	}

	return true;
}


 __device__ __forceinline__
void bfgs(					output_type_cuda_t*			x,
								change_cuda_t*			g,
								m_cuda_t*				m_cuda_gpu,
						p_cuda_t*				p_cuda_gpu,
						ig_cuda_t*				ig_cuda_gpu,
			const	float*				hunt_cap,
			const				float				epsilon_fl,
			const				int					max_steps
)
{
	int n = 3 + 3 + x->lig_torsion_size; /* the dimensions of matirx */

	matrix_d h;
	matrix_d_init(&h, n, 0);
	matrix_d_set_diagonal(&h, 1);

	change_cuda_t g_new;
	change_cuda_init_with_change(&g_new, g);

	output_type_cuda_t x_new;
	output_type_cuda_init_with_output(&x_new, x);

	float f0 = m_eval_deriv(	x,
								g,
								m_cuda_gpu,
								p_cuda_gpu,
								ig_cuda_gpu,
								hunt_cap,
								epsilon_fl
							);

	float f_orig = f0;
	/* Init g_orig, x_orig */
	change_cuda_t g_orig;
	change_cuda_init_with_change(&g_orig, g);
	output_type_cuda_t x_orig;
	output_type_cuda_init_with_output(&x_orig, x);
	/* Init p */
	change_cuda_t p;
	change_cuda_init_with_change(&p, g);

	float f_values[MAX_NUM_OF_BFGS_STEPS + 1];
	f_values[0] = f0;

	for (int step = 0; step < max_steps; step++) {

		minus_mat_vec_product(&h, g, &p);
		float f1 = 0;

		const float alpha = line_search(	m_cuda_gpu,
											p_cuda_gpu,
											ig_cuda_gpu,
											n,
											x,
											g,
											f0,
											&p,
											&x_new,
											&g_new,
											&f1,
											epsilon_fl,
											hunt_cap
										);

		change_cuda_t y;
		change_cuda_init_with_change(&y, &g_new);
		/* subtract_change */
		for (int i = 0; i < n; i++) {
			float tmp = find_change_index_read(&y, i) - find_change_index_read(g, i);
			find_change_index_write(&y, i, tmp);
		}
		f_values[step + 1] = f1;
		f0 = f1;
		output_type_cuda_init_with_output(x, &x_new);
		if (!(sqrt(scalar_product(g, g, n)) >= 1e-5))break;
		change_cuda_init_with_change(g, &g_new);

		if (step == 0) {
			float yy = scalar_product(&y, &y, n);
			if (fabs(yy) > epsilon_fl) {
				matrix_d_set_diagonal(&h, alpha * scalar_product(&y, &p, n) / yy);
			}
		}

		bool h_updated = bfgs_update(&h, &p, &y, alpha, epsilon_fl);
	}

	if (!(f0 <= f_orig)) {
		f0 = f_orig;
		output_type_cuda_init_with_output(x, &x_orig);
		change_cuda_init_with_change(g, &g_orig);
	}

	// write output_type_cuda energy
	x->e = f0;
}


/* Above based on quasi_newton.cpp */

/* Below is monte-carlo kernel, based on kernel.cl*/

 __device__ __forceinline__
void m_cuda_init_with_m_cuda(const m_cuda_t* m_cuda_old, m_cuda_t* m_cuda_new) {
	for (int i = 0; i < MAX_NUM_OF_ATOMS; i++)m_cuda_new->atoms[i] = m_cuda_old->atoms[i];
	m_cuda_new->m_coords = m_cuda_old->m_coords;
	m_cuda_new->minus_forces = m_cuda_old->minus_forces;
	m_cuda_new->ligand = m_cuda_old->ligand;
	m_cuda_new->m_num_movable_atoms = m_cuda_old->m_num_movable_atoms;
}


 __device__ __forceinline__
void get_heavy_atom_movable_coords(output_type_cuda_t* tmp, const m_cuda_t* m_cuda_gpu) {
	int counter = 0;
	for (int i = 0; i < m_cuda_gpu->m_num_movable_atoms; i++) {
		if (m_cuda_gpu->atoms[i].types[0] != EL_TYPE_H) {
			for (int j = 0; j < 3; j++)tmp->coords[counter][j] = m_cuda_gpu->m_coords.coords[i][j];
			counter++;
		}
		else {
			// printf("\n kernel2: removed H atom coords in get_heavy_atom_movable_coords()!");
		}
	}
	/* assign 0 for others */
	for (int i = counter; i < MAX_NUM_OF_ATOMS; i++) {
		for (int j = 0; j < 3; j++)tmp->coords[i][j] = 0;
	}
}

/* Bubble Sort */
//  __device__ __forceinline__
// void container_sort(output_container_cuda_t* out) {
// 	output_type_cuda_t out_tmp;
// 	for (int i = 0; i < out->current_size - 1; i++) {
// 		for (int j = 0; j < out->current_size - 1 - i; j++) {
// 			if (out->container[j].e > out->container[j + 1].e) {
// 				output_type_cuda_init_with_output(&out_tmp, &out->container[j]);
// 				output_type_cuda_init_with_output(&out->container[j], &out->container[j+1]);
// 				output_type_cuda_init_with_output(&out->container[j + 1], &out_tmp);
// 			}
// 		}
// 	}
// }

//  __device__ __forceinline__
// void add_to_output_container_cuda(output_container_cuda_t* out, const output_type_cuda_t* tmp) {
// 	if (out->current_size <= MAX_CONTAINER_SIZE_EVERY_WI) {
// 		out->container[out->current_size - 1] = *tmp;
// 		out->current_size++;
// 		container_sort(out);
// 	}
// 	else {
// 		out->container[MAX_CONTAINER_SIZE_EVERY_WI - 1] = *tmp;
// 		container_sort(out);
// 	}
// }

 __device__ __forceinline__
float generate_n(const float* pi_map, const int step) {
	return fabs(pi_map[step]) / M_PI;
}

 __device__ __forceinline__
bool metropolis_accept(float old_f, float new_f, float temperature, float n) {
	if (new_f < old_f)return true;
	const float acceptance_probability = exp((old_f - new_f) / temperature);
	bool res = n < acceptance_probability;
	return n < acceptance_probability;
}

 __device__ __forceinline__
void write_back(output_type_cuda_t* results, const output_type_cuda_t* best_out) {
	for (int i = 0; i < 3; i++)results->position[i] = best_out->position[i];
	for (int i = 0; i < 4; i++)results->orientation[i] = best_out->orientation[i];
	for (int i = 0; i < MAX_NUM_OF_LIG_TORSION; i++)results->lig_torsion[i] = best_out->lig_torsion[i];
	for (int i = 0; i < MAX_NUM_OF_FLEX_TORSION; i++)results->flex_torsion[i] = best_out->flex_torsion[i];
	results->lig_torsion_size = best_out->lig_torsion_size;
	results->e = best_out->e;
	for (int i = 0; i < MAX_NUM_OF_ATOMS; i++) {
		for (int j = 0; j < 3; j++) {
			results->coords[i][j] = best_out->coords[i][j];
		}
	}
}

__global__
void kernel(	m_cuda_t*			m_cuda_global,
				ig_cuda_t*			ig_cuda_gpu,
				p_cuda_t*			p_cuda_gpu,
				float*				rand_molec_struc_gpu,
				float*				best_e_gpu,
				int					bfgs_max_steps,
				unsigned int		num_steps,
				float				mutation_amplitude,
				random_maps_t*		rand_maps_gpu,
				float				epsilon_fl,
				float*				hunt_cap_gpu,
				float*				authentic_v_gpu,
				output_type_cuda_t*	results,
				int					search_depth
				// int					e,
				// int					total_wi
)
{
	/* OpenCL function */
	// int gx = get_global_id(0);
	// int gy = get_global_id(1);
	// int gs = get_global_size(0);
	// int gl = get_global_linear_id();

	// CUDA function
	// TODO

	// tid
	int gll = blockIdx.x * blockDim.x + threadIdx.x;
	// int gll = 0;
	// printf("%d", gll);
	float best_e = INFINITY;

	do
	{
		//if (gll % 100 == 0)printf("\nThread %d START", gll);

		m_cuda_t m_cuda_gpu;
		m_cuda_init_with_m_cuda(m_cuda_global, &m_cuda_gpu);

		output_type_cuda_t tmp; // private memory, shared only in work item
		change_cuda_t g;
		output_type_cuda_init(&tmp, rand_molec_struc_gpu + gll * (SIZE_OF_MOLEC_STRUC / sizeof(float)));
		g.lig_torsion_size = tmp.lig_torsion_size;
		// BFGS
		output_type_cuda_t best_out;
		output_type_cuda_t candidate;

		// for (int step = 0; step < search_depth; step++) {
		for (int step = 0; step < search_depth; step++) { //debug
			output_type_cuda_init_with_output(&candidate, &tmp);

			int map_index = (step + gll * search_depth) % MAX_NUM_OF_RANDOM_MAP;
			mutate_conf_cuda(map_index, num_steps, &candidate, rand_maps_gpu->int_map, rand_maps_gpu->sphere_map,
				rand_maps_gpu->pi_map, m_cuda_gpu.ligand.begin, m_cuda_gpu.ligand.end, m_cuda_gpu.atoms,
				&m_cuda_gpu.m_coords, m_cuda_gpu.ligand.rigid.origin[0], epsilon_fl, mutation_amplitude);

			bfgs(&candidate, &g, &m_cuda_gpu, p_cuda_gpu, ig_cuda_gpu, hunt_cap_gpu, epsilon_fl, bfgs_max_steps);

			float n = generate_n(rand_maps_gpu->pi_map, map_index);

			if (step == 0 || metropolis_accept(tmp.e, candidate.e, 1.2, n)) {

				output_type_cuda_init_with_output(&tmp, &candidate);
				set(&tmp, &m_cuda_gpu.ligand.rigid, &m_cuda_gpu.m_coords,
					m_cuda_gpu.atoms, m_cuda_gpu.m_num_movable_atoms, epsilon_fl);
				if (tmp.e < best_e) {
					bfgs(	&tmp,
							&g,
							&m_cuda_gpu,
							p_cuda_gpu,
							ig_cuda_gpu,
							authentic_v_gpu,
							epsilon_fl,
							bfgs_max_steps
					);
					// set
					if (tmp.e < best_e) {
						set(&tmp, &m_cuda_gpu.ligand.rigid, &m_cuda_gpu.m_coords,
							m_cuda_gpu.atoms, m_cuda_gpu.m_num_movable_atoms, epsilon_fl);

						output_type_cuda_init_with_output(&best_out, &tmp);
						get_heavy_atom_movable_coords(&best_out, &m_cuda_gpu); // get coords
						best_e = tmp.e;
					}

				}
			}

		}

		// write the best conformation back to CPU
		write_back(results, &best_out);
		//if (gll % 100 == 0)printf("\nThread %d FINISH", gll);
	}
	while(0);
}

/* Above based on kernel.cl */

/* Below based on monte-carlo.cpp */

#ifdef ENABLE_CUDA

output_type monte_carlo::cuda_to_vina(output_type_cuda_t &results) const {
	printf("entering cuda_to_vina\n");
	printf("%f", results.e);
	conf tmp_c;
	tmp_c.ligands.resize(1);
	// Position
	for (int j = 0; j < 3; j++)tmp_c.ligands[0].rigid.position[j] = results.position[j];
	// Orientation
	qt q(results.orientation[0], results.orientation[1], results.orientation[2], results.orientation[3]);
	tmp_c.ligands[0].rigid.orientation = q;
	output_type result_vina(tmp_c, results.e);
	// torsion
	for (int j = 0; j < results.lig_torsion_size; j++)result_vina.c.ligands[0].torsions.push_back(results.lig_torsion[j]);
	// coords
	for (int j = 0; j < MAX_NUM_OF_ATOMS; j++) {
		vec v_tmp(results.coords[j][0], results.coords[j][1], results.coords[j][2]);
		if (v_tmp[0] * v_tmp[1] * v_tmp[2] != 0) result_vina.coords.push_back(v_tmp);
	}
	return result_vina;
}

__host__
void monte_carlo::operator()(model& m, output_container& out, const precalculate_byatom& p, const igrid& ig, const vec& corner1, const vec& corner2, rng& generator) const {

	/* Definitions from vina1.2 */
	printf("entering CUDA monte_carlo search\n"); //debug

	// int evalcount = 0;
	vec authentic_v(1000, 1000, 1000); // FIXME? this is here to avoid max_fl/max_fl
	conf_size s = m.get_size();
	change g(s);
	output_type tmp(s, 0);

	// tmp.c.randomize(corner1, corner2, generator); // should be moved to gpu
	// fl best_e = max_fl; // should be moved to gpu
	quasi_newton quasi_newton_par;
    const int quasi_newton_par_max_steps = local_steps; // no need to decrease step

	/* Allocate CPU memory and define new data structure */
	printf("Allocating CPU memory\n"); //debug
	m_cuda_t m_cuda;
	assert(m.atoms.size() < MAX_NUM_OF_ATOMS);

	for (int i = 0; i < m.atoms.size(); i++) {
		m_cuda.atoms[i].types[0] = m.atoms[i].el;// To store 4 atoms types (el, ad, xs, sy)
		m_cuda.atoms[i].types[1] = m.atoms[i].ad;
		m_cuda.atoms[i].types[2] = m.atoms[i].xs;
		m_cuda.atoms[i].types[3] = m.atoms[i].sy;
		for (int j = 0; j < 3; j++) {
			m_cuda.atoms[i].coords[j] = m.atoms[i].coords[j];// To store atom coords
		}
	}


	// To store atoms coords
	for (int i = 0; i < m.coords.size(); i++) {
		for (int j = 0; j < 3; j++) {
			m_cuda.m_coords.coords[i][j] = m.coords[i].data[j];
		}
	}

	//To store minus forces
	for (int i = 0; i < m.coords.size(); i++) {
		for (int j = 0; j < 3; j++) {
			m_cuda.minus_forces.coords[i][j] = m.minus_forces[i].data[j];
		}
	}

    // Preparing ligand data
	printf("prepare ligand data\n");
	assert(m.num_other_pairs() == 0); // m.other_paris is not supported!
	assert(m.ligands.size() == 1); // Only one ligand supported!
	m_cuda.ligand.pairs.num_pairs = m.ligands[0].pairs.size();
	for (int i = 0; i < m_cuda.ligand.pairs.num_pairs; i++) {
		m_cuda.ligand.pairs.type_pair_index[i]	= m.ligands[0].pairs[i].type_pair_index;
		m_cuda.ligand.pairs.a[i]					= m.ligands[0].pairs[i].a;
		m_cuda.ligand.pairs.b[i]					= m.ligands[0].pairs[i].b;
	}
	m_cuda.ligand.begin = m.ligands[0].begin; // 0
	m_cuda.ligand.end = m.ligands[0].end; // 29
	ligand m_ligand = m.ligands[0]; // Only support one ligand
	assert(m_ligand.end < MAX_NUM_OF_ATOMS);

	// Store root node
	m_cuda.ligand.rigid.atom_range[0][0] = m_ligand.node.begin;
	m_cuda.ligand.rigid.atom_range[0][1] = m_ligand.node.end;
	for (int i = 0; i < 3; i++) m_cuda.ligand.rigid.origin[0][i] = m_ligand.node.get_origin()[i];
	for (int i = 0; i < 9; i++) m_cuda.ligand.rigid.orientation_m[0][i] = m_ligand.node.get_orientation_m().data[i];
	m_cuda.ligand.rigid.orientation_q[0][0] = m_ligand.node.orientation().R_component_1();
	m_cuda.ligand.rigid.orientation_q[0][1] = m_ligand.node.orientation().R_component_2();
	m_cuda.ligand.rigid.orientation_q[0][2] = m_ligand.node.orientation().R_component_3();
	m_cuda.ligand.rigid.orientation_q[0][3] = m_ligand.node.orientation().R_component_4();
	for (int i = 0; i < 3; i++) {m_cuda.ligand.rigid.axis[0][i] = 0;m_cuda.ligand.rigid.relative_axis[0][i] = 0;m_cuda.ligand.rigid.relative_origin[0][i] = 0;}

	// Store children nodes (in depth-first order)
	printf("store children nodes\n"); //debug

	struct tmp_struct {
		int start_index = 0;
		int parent_index = 0;
		void store_node(tree<segment>& child_ptr, rigid_cuda_t& rigid) {
			start_index++; // start with index 1, index 0 is root node
			rigid.parent[start_index] = parent_index;
			rigid.atom_range[start_index][0] = child_ptr.node.begin;
			rigid.atom_range[start_index][1] = child_ptr.node.end;
			for (int i = 0; i < 9; i++) rigid.orientation_m[start_index][i] = child_ptr.node.get_orientation_m().data[i];
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
			if (child_ptr.children.size() == 0) return;
			else {
				assert(start_index < MAX_NUM_OF_RIGID);
				int parent_index_tmp = start_index;
				for (int i = 0; i < child_ptr.children.size(); i++) {
					this->parent_index = parent_index_tmp; // Update parent index
					this->store_node(child_ptr.children[i], rigid);
				}
			}
		}
	};
	tmp_struct ts;
	for (int i = 0; i < m_ligand.children.size(); i++) {
		ts.parent_index = 0; // Start a new branch, whose parent is 0
		ts.store_node(m_ligand.children[i], m_cuda.ligand.rigid);
	}
	m_cuda.ligand.rigid.num_children = ts.start_index;

	// set children_map
	printf("set children map\n"); //debug
	for (int i = 0; i < MAX_NUM_OF_RIGID; i++)
		for (int j = 0; j < MAX_NUM_OF_RIGID; j++)
			m_cuda.ligand.rigid.children_map[i][j] = false;
	// printf("1"); //debug
	for (int i = 1; i < m_cuda.ligand.rigid.num_children + 1; i++) {
		int parent_index = m_cuda.ligand.rigid.parent[i];
		m_cuda.ligand.rigid.children_map[parent_index][i] = true;
	}
	m_cuda.m_num_movable_atoms = m.num_movable_atoms();
	size_t m_cuda_size = sizeof(m_cuda);
	printf("m_cuda_size=%lu\n", m_cuda_size);

    // Preparing ig related data
	printf("Preparing ig related data\n"); //debug
	ig_cuda_t* ig_cuda_ptr;
	checkCUDA(cudaMallocHost(&ig_cuda_ptr, sizeof(ig_cuda_t)));
	ig_cuda_ptr->atu = ig.get_atu(); // atu
	ig_cuda_ptr->slope = ig.get_slope(); // slope
	std::vector<grid> tmp_grids = ig.get_grids();
	int grid_size = tmp_grids.size();
	assert(GRIDS_SIZE > grid_size); //

	printf("2,%d", grid_size);
	for (int i = 0; i < grid_size; i++) {
		// printf("i=%d\n",i); //debug
		for (int j = 0; j < 3; j++) {
			ig_cuda_ptr->grids[i].m_init[j] = tmp_grids[i].m_init[j];
			ig_cuda_ptr->grids[i].m_factor[j] = tmp_grids[i].m_factor[j];
			ig_cuda_ptr->grids[i].m_dim_fl_minus_1[j] = tmp_grids[i].m_dim_fl_minus_1[j];
			ig_cuda_ptr->grids[i].m_factor_inv[j] = tmp_grids[i].m_factor_inv[j];
		}
		if (tmp_grids[i].m_data.dim0() != 0) {
			ig_cuda_ptr->grids[i].m_i = tmp_grids[i].m_data.dim0(); assert(MAX_NUM_OF_GRID_MI >= ig_cuda_ptr->grids[i].m_i);
			ig_cuda_ptr->grids[i].m_j = tmp_grids[i].m_data.dim1(); assert(MAX_NUM_OF_GRID_MJ >= ig_cuda_ptr->grids[i].m_j);
			ig_cuda_ptr->grids[i].m_k = tmp_grids[i].m_data.dim2(); assert(MAX_NUM_OF_GRID_MK >= ig_cuda_ptr->grids[i].m_k);

			for (int j = 0; j < ig_cuda_ptr->grids[i].m_i * ig_cuda_ptr->grids[i].m_j * ig_cuda_ptr->grids[i].m_k; j++) {
				ig_cuda_ptr->grids[i].m_data[j] = tmp_grids[i].m_data.m_data[j];
			}
		}
		else {
			ig_cuda_ptr->grids[i].m_i = 0;
			ig_cuda_ptr->grids[i].m_j = 0;
			ig_cuda_ptr->grids[i].m_k = 0;
		}
	}
	size_t ig_cuda_size = sizeof(ig_cuda_t);
	printf("ig_cuda_size=%lu\n", ig_cuda_size);

	// Generating random ligand structures
	printf("Generating random ligand structures\n"); //debug
	output_type_cuda_t* rand_molec_struc;
	// rand_molec_struc_vec.resize(thread);
	checkCUDA(cudaMallocHost(&rand_molec_struc, sizeof(output_type_cuda_t)));

	int lig_torsion_size = tmp.c.ligands[0].torsions.size();
	int flex_torsion_size;
	if (tmp.c.flex.size() != 0) flex_torsion_size = tmp.c.flex[0].torsions.size();
	else flex_torsion_size = 0;
	vec uniform_data;

	tmp.c.randomize(corner1, corner2, generator); // generate a random structure
	for (int j = 0; j < 3; j++) rand_molec_struc->position[j] = tmp.c.ligands[0].rigid.position[j];
	assert(lig_torsion_size < MAX_NUM_OF_LIG_TORSION);
	for (int j = 0; j < lig_torsion_size; j++) rand_molec_struc->lig_torsion[j] = tmp.c.ligands[0].torsions[j];// Only support one ligand
	assert(flex_torsion_size < MAX_NUM_OF_FLEX_TORSION);
	for (int j = 0; j < flex_torsion_size; j++) rand_molec_struc->flex_torsion[j] = tmp.c.flex[0].torsions[j];// Only support one flex

	rand_molec_struc->orientation[0] = (float)tmp.c.ligands[0].rigid.orientation.R_component_1();
	rand_molec_struc->orientation[1] = (float)tmp.c.ligands[0].rigid.orientation.R_component_2();
	rand_molec_struc->orientation[2] = (float)tmp.c.ligands[0].rigid.orientation.R_component_3();
	rand_molec_struc->orientation[3] = (float)tmp.c.ligands[0].rigid.orientation.R_component_4();

	rand_molec_struc->lig_torsion_size = lig_torsion_size;


    // Preaparing p related data
	printf("Preaparing p related data\n"); //debug
	p_cuda_t p_cuda;
	p_cuda.m_cutoff_sqr = p.cutoff_sqr();
	p_cuda.factor = p.m_factor;
	p_cuda.n = p.m_n;
	printf("%d, %lu\n", MAX_P_DATA_M_DATA_SIZE, p.m_data.m_data.size());
	// assert(MAX_P_DATA_M_DATA_SIZE > p.m_data.m_data.size());
	printf("FAST_SIZE=%d, fast.size()=%lu\n", FAST_SIZE, p.m_data.m_data[0].fast.size());
	printf("SMOOTH_SIZE=%d, smooth.size()=%lu\n", SMOOTH_SIZE, p.m_data.m_data[0].smooth.size());
	for (int i = 0; i < MAX_P_DATA_M_DATA_SIZE; i++) { // only copy part of it
		p_cuda.m_data[i].factor = p.m_data.m_data[i].factor;

		// assert(FAST_SIZE == p.m_data.m_data[i].fast.size());
		// assert(SMOOTH_SIZE == p.m_data.m_data[i].smooth.size());
		for (int j = 0; j < FAST_SIZE; j++) {
			p_cuda.m_data[i].fast[j] = p.m_data.m_data[i].fast[j];
		}
		for (int j = 0; j < SMOOTH_SIZE; j++) {
			p_cuda.m_data[i].smooth[j][0] = p.m_data.m_data[i].smooth[j].first;
			p_cuda.m_data[i].smooth[j][1] = p.m_data.m_data[i].smooth[j].second;
		}
	}
	size_t p_cuda_size = sizeof(p_cuda_t);

	// Generate random maps
	random_maps_t* rand_maps;
	checkCUDA(cudaMallocHost(&rand_maps, sizeof(random_maps_t)));
	for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
		rand_maps->int_map[i] = random_int(0, int(lig_torsion_size), generator);
		rand_maps->pi_map[i] = random_fl(-pi, pi, generator);
	}
	for (int i = 0; i < MAX_NUM_OF_RANDOM_MAP; i++) {
		vec rand_coords = random_inside_sphere(generator);
		for (int j = 0; j < 3 ; j ++) {
			rand_maps->sphere_map[i][j] = rand_coords[j];
		}
	}
	size_t rand_maps_size = sizeof(random_maps_t);

	fl hunt_cap_float[3] = {hunt_cap[0], hunt_cap[1], hunt_cap[2]};
	float authentic_v_float[3] = {static_cast<float>(authentic_v[0]), static_cast<float>(authentic_v[1]),static_cast<float>(authentic_v[2])};
	float mutation_amplitude_float = static_cast<float>(mutation_amplitude);
	float epsilon_fl_float = static_cast<float>(epsilon_fl);
	// int	total_wi = max_wi_size[0] * max_wi_size[1];

	/* Allocate GPU memory */
	printf("Allocating GPU memory\n");
	// rand_molec_struc_gpu
	float *rand_molec_struc_gpu;
	checkCUDA(cudaMalloc(&rand_molec_struc_gpu, SIZE_OF_MOLEC_STRUC));
	std::vector<float> pos(rand_molec_struc->position, rand_molec_struc->position + 3);
	std::vector<float> ori(rand_molec_struc->orientation, rand_molec_struc->orientation + 4);
	std::vector<float> lig_tor(rand_molec_struc->lig_torsion, rand_molec_struc->lig_torsion + MAX_NUM_OF_LIG_TORSION);
	std::vector<float> flex_tor(rand_molec_struc->flex_torsion, rand_molec_struc->flex_torsion + MAX_NUM_OF_FLEX_TORSION);
	float lig_tor_size = rand_molec_struc->lig_torsion_size;
	checkCUDA(cudaMemcpy(rand_molec_struc_gpu, pos.data(), pos.size() * sizeof(float), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(rand_molec_struc_gpu + pos.size(), ori.data(), ori.size() * sizeof(float), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(rand_molec_struc_gpu + pos.size() + ori.size(), lig_tor.data(), lig_tor.size() * sizeof(float), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(rand_molec_struc_gpu + pos.size() + ori.size() + MAX_NUM_OF_LIG_TORSION, flex_tor.data(),
		flex_tor.size() * sizeof(float), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(rand_molec_struc_gpu + pos.size() + ori.size() + MAX_NUM_OF_LIG_TORSION + MAX_NUM_OF_FLEX_TORSION,
		&lig_tor_size, sizeof(float), cudaMemcpyHostToDevice));
	// best_e_gpu
	float *best_e_gpu;
	checkCUDA(cudaMalloc(&best_e_gpu, sizeof(float)));
	checkCUDA(cudaMemcpy(best_e_gpu, &max_fl, sizeof(float), cudaMemcpyHostToDevice));
	// rand_maps_gpu
	random_maps_t *rand_maps_gpu;
	checkCUDA(cudaMalloc(&rand_maps_gpu, rand_maps_size));
	checkCUDA(cudaMemcpy(rand_maps_gpu, rand_maps, rand_maps_size, cudaMemcpyHostToDevice));
	// hunt_cap_gpu
	float *hunt_cap_gpu;
	checkCUDA(cudaMalloc(&hunt_cap_gpu, 3 * sizeof(float)));
	checkCUDA(cudaMemcpy(hunt_cap_gpu, hunt_cap_float, 3 * sizeof(float), cudaMemcpyHostToDevice));
	// Preparing m related data
	m_cuda_t* m_cuda_gpu;
	printf("m_cuda_size=%lu", m_cuda_size);
	checkCUDA(cudaMalloc(&m_cuda_gpu, m_cuda_size));
	printf("m_cuda_gpu");
	printf("%p\n", m_cuda_gpu);
	checkCUDA(cudaMemcpy(m_cuda_gpu, &m_cuda, m_cuda_size, cudaMemcpyHostToDevice));
	// Preparing p related data
	p_cuda_t *p_cuda_gpu;
	checkCUDA(cudaMalloc(&p_cuda_gpu, p_cuda_size));
	checkCUDA(cudaMemcpy(p_cuda_gpu, &p_cuda, p_cuda_size, cudaMemcpyHostToDevice));
	// Preparing ig related data (cache related data)
	ig_cuda_t *ig_cuda_gpu;
	checkCUDA(cudaMalloc(&ig_cuda_gpu, ig_cuda_size));
	checkCUDA(cudaMemcpy(ig_cuda_gpu, ig_cuda_ptr, ig_cuda_size, cudaMemcpyHostToDevice));
	float *authentic_v_gpu;
	checkCUDA(cudaMalloc(&authentic_v_gpu, sizeof(authentic_v_float)));
	checkCUDA(cudaMemcpy(authentic_v_gpu, authentic_v_float, sizeof(authentic_v_float), cudaMemcpyHostToDevice));
	// Preparing result data
	output_type_cuda_t *results_gpu;
	checkCUDA(cudaMalloc(&results_gpu, sizeof(output_type_cuda_t)));

	/* Launch kernel */
	printf("launch kernel\n");
	kernel<<<1,1>>>(m_cuda_gpu, ig_cuda_gpu, p_cuda_gpu, rand_molec_struc_gpu,
		best_e_gpu, quasi_newton_par_max_steps, global_steps, mutation_amplitude_float,
		rand_maps_gpu, epsilon_fl, hunt_cap_gpu, authentic_v_gpu, results_gpu, global_steps);

	checkCUDA(cudaDeviceSynchronize());

	/* Convert result data. Since thread==1, result_vina can be defined as an object
	 * instead of a vector
	 */
	printf("cuda to vina\n");
	output_type_cuda_t results;
	checkCUDA(cudaMemcpy(&results, results_gpu, sizeof(output_type_cuda_t), cudaMemcpyDeviceToHost));
	output_type result_vina = cuda_to_vina(results);
	add_to_output_container(out, result_vina, min_rmsd, num_saved_mins);
	VINA_CHECK(!out.empty());
	VINA_CHECK(out.front().e <= out.back().e); // make sure the sorting worked in the correct order

	/* Free memory */
	checkCUDA(cudaFree(m_cuda_gpu));
	checkCUDA(cudaFree(ig_cuda_gpu));
	checkCUDA(cudaFree(p_cuda_gpu));
	checkCUDA(cudaFree(rand_molec_struc_gpu));
	checkCUDA(cudaFree(best_e_gpu));
	checkCUDA(cudaFree(hunt_cap_gpu));
	checkCUDA(cudaFree(authentic_v_gpu));
	checkCUDA(cudaFree(results_gpu));
	free(rand_maps);
	free(rand_molec_struc);

	printf("exit monte_carlo\n");

}

/* Above based on monte-carlo.cpp */

#endif
