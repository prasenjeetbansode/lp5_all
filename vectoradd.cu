#include <iostream>
#include <vector>

// CUDA kernel for vector addition
__global__ void vectorAdd(float* a, float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int size = 1000000;  // Size of the vectors
    int numBytes = size * sizeof(float);

    // Allocate memory on the host (CPU)
    std::vector<float> h_a(size);
    std::vector<float> h_b(size);
    std::vector<float> h_c(size);

    // Initialize the vectors
    for (int i = 0; i < size; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // Allocate memory on the device (GPU)
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, numBytes);
    cudaMalloc((void**)&d_b, numBytes);
    cudaMalloc((void**)&d_c, numBytes);

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a.data(), numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), numBytes, cudaMemcpyHostToDevice);

    // Set the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // Copy the result back from device to host
    cudaMemcpy(h_c.data(), d_c, numBytes, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < size; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
