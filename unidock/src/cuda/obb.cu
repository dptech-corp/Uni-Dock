#include <stdio.h>
#include <cuda_runtime.h>

__global__ void computeCentroidKernel(float* points, float* centroid, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    atomicAdd(&centroid[0], points[3*idx]);
    atomicAdd(&centroid[1], points[3*idx + 1]);
    atomicAdd(&centroid[2], points[3*idx + 2]);
}

__global__ void computeCovarianceKernel(float* points, float* centroid, float* covMatrix, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float x = points[3*idx] - centroid[0];
    float y = points[3*idx + 1] - centroid[1];
    float z = points[3*idx + 2] - centroid[2];

    atomicAdd(&covMatrix[0], x * x);
    atomicAdd(&covMatrix[1], x * y);
    atomicAdd(&covMatrix[2], x * z);
    atomicAdd(&covMatrix[3], y * y);
    atomicAdd(&covMatrix[4], y * z);
    atomicAdd(&covMatrix[5], z * z);
}

int main() {
    const int numPoints = 1024;
    float points[3 * numPoints]; 
    float centroid[3] = {0}; 
    float covMatrix[6] = {0}; 

    // 初始化点集（示例中省略了具体的初始化代码）
    // 初始化随机数生成器
    srand(time(NULL));

    // 初始化点集
    for (int i = 0; i < numPoints; ++i) {
        points[3*i] = (float)rand() / RAND_MAX;   
        points[3*i + 1] = (float)rand() / RAND_MAX; 
        points[3*i + 2] = (float)rand() / RAND_MAX; 
    }

    float *d_points, *d_centroid, *d_covMatrix;
    cudaMalloc(&d_points, 3 * numPoints * sizeof(float));
    cudaMalloc(&d_centroid, 3 * sizeof(float));
    cudaMalloc(&d_covMatrix, 6 * sizeof(float));

    cudaMemcpy(d_points, points, 3 * numPoints * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    computeCentroidKernel<<<numBlocks, blockSize>>>(d_points, d_centroid, numPoints);
    cudaDeviceSynchronize();

    cudaMemcpy(centroid, d_centroid, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // 计算每个点对质心的贡献
    for (int i = 0; i < 3; ++i) {
        centroid[i] /= numPoints;
    }

    cudaMemcpy(d_centroid, centroid, 3 * sizeof(float), cudaMemcpyHostToDevice);

    computeCovarianceKernel<<<numBlocks, blockSize>>>(d_points, d_centroid, d_covMatrix, numPoints);
    cudaDeviceSynchronize();

    cudaMemcpy(covMatrix, d_covMatrix, 6 * sizeof(float), cudaMemcpyDeviceToHost);

    // 清理资源
    cudaFree(d_points);
    cudaFree(d_centroid);
    cudaFree(d_covMatrix);

    // 打印结果（示例中省略了打印代码）
        // 打印质心
    printf("质心坐标: (%f, %f, %f)\n", centroid[0], centroid[1], centroid[2]);

    // 打印协方差矩阵
    // 注意协方差矩阵是对称的，所以这里只计算并打印了上三角部分
    printf("协方差矩阵:\n");
    printf("%f %f %f\n", covMatrix[0], covMatrix[1], covMatrix[2]);
    printf("%f %f\n", covMatrix[3], covMatrix[4]);
    printf("%f\n", covMatrix[5]);

    return 0;
}
