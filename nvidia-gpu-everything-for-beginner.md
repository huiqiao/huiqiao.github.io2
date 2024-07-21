# **Nvidia GPU Architectures (2014-2024)**

Here's an overview of Nvidia GPU architectures from the past 10 years, popular GPU cards for each architecture(Brown color for server version):

1. **Maxwell (2014-2015)**
    - Popular GPUs: GTX 980 Ti, Tesla K80
2. **Pascal (2016-2017)**
    - Popular GPUs: GTX 1080 Ti, Tesla P100
3. **Volta (2017-2018)**
    - Popular GPUs: Titan V, Tesla V100
4. **Turing (2018-2019)**
    - Popular GPUs: RTX 2080 Ti, Tesla T4
5. **Ampere (2020-2021)**
    - Popular GPUs: RTX 3090, RTX 3080, RTX 3070, A100, A800(mainly for China), A40, A20
6. **Ada Lovelace (2022-present)**
    - Popular GPUs: RTX 4090, RTX 4090D(**China version**), L20(mainly for China), L40
7. **Hopper (2022-present, focused on data centers)**
    - Popular GPUs: H100, H800, H20(mainly for China)
8. **Blackwell(2023 - present, focused on data centers)**
    - Popular GPUs: GB200
  # **Precision Formats Supported**

Different precision formats offer a trade-off between computational accuracy and performance. They are represented in scientific notation in computers, consisting of three parts:

- Sign bit
    - 0 for positive numbers
    - 1 for negative number
- Exponent
    - Represents the power of 2 in scientific notation
    - Uses a biased representation to handle both positive and negative exponents
    - Bias is 127 for single-precision and 1023 for double-precision
- Mantissa (also called significand or fraction)
    - Represents the significant digits of the number
    - In normalized form, there's an implicit leading 1 before the binary point

Please refer this wiki for how these parts convert number to binary format: https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Exponent_encoding

Here's an explanation of the various formats:

1. FP64 (Double Precision)
- **Bits**: 64 (52-bit mantissa, 11-bit exponent, 1-bit sign)
- **Use Cases**:
    - Scientific computing requiring high precision
    - Climate modeling
    - Computational fluid dynamics
    - Financial modeling with high accuracy requirements
- **Performance**: Slowest but highest precision
2. FP32 (Single Precision)
- **Bits**: 32 (23-bit mantissa, 8-bit exponent, 1-bit sign)
- **Use Cases**:
    - General-purpose computing
    - AI model training (when high precision is needed)
    - Computer graphics
    - Physics simulations
- **Performance**: Balanced between precision and speed
3. TF32 (Tensor Float 32)
- **Bits**: 19 (10-bit mantissa, 8-bit exponent, 1-bit sign)
- **Use Cases**:
    - AI model training (NVIDIA-specific format)
    - Offers a balance between FP32 accuracy and FP16 speed
    - First introduced with the **Ampere** architecture
- **Performance**: Faster than FP32 with minimal accuracy loss for many AI tasks
4. FP16 (Half Precision)
- **Bits**: 16(10-bit mantissa, 5-bit exponent, 1-bit sign)
- **Use Cases**:
    - AI model training and inference
    - Computer vision
    - Image processing
- **Performance**: Faster and more memory-efficient than FP32, with some accuracy trade-offs
5. BF16 (Brain Float 16)
- **Bits**: 16 (7-bit mantissa, 8-bit exponent, 1-bit sign)
- **Use Cases**:
    - AI model training, especially for **Large Language Models**
    - Offers better dynamic range than FP16
    - First introduced with the **Ampere** architecture
- **Performance**: Good balance of speed and accuracy for many AI workloads
6. INT8 (8-bit Integer)
- **Bits**: 8 (0-bit mantissa, 7-bit exponent, 1-bit sign)
- **Use Cases**:
    - AI inference
    - Image processing
    - Low-precision neural network acceleration
- **Performance**: High performance and low memory usage, suitable for deployment on edge devices
7. FP8 (8-bit Floating Point)
- **Bits**: 8 (3-bit mantissa, 4-bit exponent, 1-bit sign)
- **Use Cases**:
    - AI training and inference (emerging format)
    - Potential for very efficient large language model training
- **Performance**: Highest performance, but requires careful handling to maintain accuracy

Illustration(image from nvidia-h100-tensor-core-hopper-whitepaper.pdf):

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f0690f3a-5cae-40ee-8511-4cfa3d154f0d/2e7f7d36-c99d-43fd-adb8-d6571a8baac2/Untitled.png)

Key Points:

1. **AI Model Training**: Typically uses FP32, TF32, FP16, or BF16. The choice depends on the model size, required accuracy, and available hardware.
2. **AI Inference**: Often uses lower precision formats like FP16, INT8, or even INT4 to improve speed and reduce memory usage.
3. **Scientific Computing**: Generally requires higher precision formats like FP64 or FP32 due to the need for high accuracy in calculations.
4. **Graphics and Visualization**: Typically uses FP32 or FP16, balancing visual quality and performance.
5. **Mixed Precision**: Many modern AI workflows use mixed precision training, combining different formats (e.g., FP16 for forward/backward passes and FP32 for weight updates) to balance speed and accuracy.
6. **Hardware Considerations**: The choice of precision often depends on the available hardware. For example, NVIDIA's Tensor Cores can accelerate certain precisions more effectively than others.
7. **Model Size vs. Precision**: As AI models grow larger, there's a trend towards using lower precision formats to manage computational and memory requirements, with techniques developed to maintain accuracy.

Understanding these precision formats and their use cases is crucial for optimizing performance in various computational tasks, especially in the fields of AI and high-performance computing.

# **Key GPU Components**

The fundamental computational units in Nvidia GPU is Streaming Multiprocessors(SM). GH100  full GPU has 144 SMs(image from nvidia-h100-tensor-core-hopper-whitepaper.pdf). 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f0690f3a-5cae-40ee-8511-4cfa3d154f0d/cc102169-3844-4359-9716-e791bbc0484a/Untitled.png)

Let’s talk about the major components of SM:

1. **CUDA Cores:**
    - the primary parallel processing units for general-purpose computing
2. **Tensor Cores:**
    - Specialized cores for matrix multiply-accumulate operations
    - Accelerate AI and deep learning workloads
    - Support various precision formats (FP16, FP32, TF32, BF16, INT8, FP8)
    - Typically 4 Tensor Cores per SM in recent architectures(H100)
    - First introduced with the **Volta** architecture
3. **Load/Store Units(LD/ST):**
    - Handle memory operations for loading and storing data
4. **Special Function Units (SFUs):**
    - Execute transcendental instructions (sin, cos, exp, etc.)
5. **Warp Schedulers:**
    - Manage and dispatch warps (groups of 32 threads) for execution
6. **Register File:**
    - Fast, on-chip memory for storing local variables
7. **L1 Cache / Shared Memory:**
    - On-chip memory that can be configured as either L1 cache or shared memory
8. **Texture Units:**
    - Handle texture sampling and filtering operations
9. **Ray Tracing (RT) Cores(Not for server side GPUs):**
    - Present in recent architectures (e.g., Turing, Ampere, Ada Lovelace)
    - Accelerate ray tracing operations for graphics workloads

# Memory **Hierarchy**

1. **Register File**:
    - Fastest on-chip memory(256 KB per SM for H100)
    - Organized into 32 banks (matching 32 threads in a warp)
    - Directly feeds into CUDA cores and Tensor cores
    - **Latency**: Effectively 0 cycles (immediate access)
    - **Bandwidth**: Highest on-chip bandwidth, up to 15 TB/s
2. **L1 Cache**:
    - On-chip storage for fast access to recently used data(Configurable up to  228 KB per SM for H100)
    - Serves as overflow for register spilling
    - **Latency**: ~28-30 cycles
    - **Bandwidth**: ~128 B/cycle per SM
3. **Shared Memory**:
    - On-chip memory shared by threads in a block(shared with L1 cache)
    - Allows inter-thread communication
    - **Latency**: ~20-30 cycles
    - **Bandwidth**: Very high, up to 5 TB/s
    - **Key diff** **with L1 cache**: managed by programmer, L1 cache is automatically managed by the hardware. Shared memory is limited to a thread block, while L1 cache serves all threads on the SM.
4. **L2 Cache**:
    - Larger, shared cache between all SMs(50MB for H100)
    - Intermediary between on-chip and global memory
    - **Latency**: ~193 cycles
    - **Bandwidth**: ~17.6 B/cycle across all SMs
5. **Global Memory(DRAM)**:
    - Main GPU memory (H100 uses HBM, 80GB)
    - Largest but slowest memory type
    - **Latency**: 400-800 cycles
    - **Bandwidth**: Lower than on-chip memory, but still high (e.g., 900 GB/s for A100)
6. **Constant and Texture Memory**:
    - Read-only memory areas
    - Cached for faster access
    - **Latency**: Similar to L1 cache when data is cached
    - **Bandwidth**: No need to worry?

On chip memory has low latency and high bandwidth, in order to achieve high computational performance, try to better utilize on chip memory and avoid frequent data exchange between global memory and the computational cores.

# Computing Blocks

You probably heard warp, thread, block in previous paragraphs. Let me explain these concepts in detail.

1. **Thread**
    - **Definition**: The smallest unit of execution in a CUDA program.
    - **Execution**: Each thread executes the same program (kernel) but operates on different data.
    - **Identification**: Threads are uniquely identified by their thread ID within a block.
2. **Warp**
    - **Definition**: A group of 32 consecutive threads, number(32) controlled by system, not program
    - **Execution**: Warps are the primary unit of execution in an SM. All threads in a warp execute the same instruction simultaneously (SIMT - Single Instruction, Multiple Threads).
    - **Scheduling**: Each SM contains a warp scheduler that manages the execution of warps. The scheduler can switch between warps to hide latency, such as memory access delays.
    - **Synchronization**: Threads within a warp execute in lock-step, but synchronization across warps requires explicit barriers (e.g., `__syncthreads()`).
3. **Block**
    - **Definition**: A group of threads that execute together and can share data via shared memory.
    - **Configuration**: Blocks are defined by the programmer and can contain up to 1024 threads (depending on the GPU architecture).
    - **Mapping to SM**: Each block is assigned to a single SM for execution. The SM manages the scheduling and execution of all warps within the block.
    - **Synchronization**: Threads within a block can synchronize using `__syncthreads()`. Blocks are independent and do not synchronize with each other.
4. **Grid**
    - **Definition**: A grid is a collection of thread blocks that execute a kernel function.
    - **Structure**: Grids can be one-dimensional, two-dimensional, or three-dimensional arrays of blocks.
    - **Purpose**: Organizes the execution of a kernel across multiple blocks, allowing for large-scale parallelism.
    - **Indexing**: Blocks within a grid are indexed using the built-in variable `blockIdx`.
    - **Kernel Launch**: When a kernel is launched, the grid dimensions and block dimensions are specified, determining the total number of threads.

**Relationships and Hierarchy**

- **Grid**: A collection of blocks. The entire grid is launched by a kernel call.
- **Block**: A collection of threads. Each block is mapped to an SM.
- **Warp**: A collection of 32 threads. Warps are the primary unit of execution within an SM.
- **Thread**: The smallest unit of execution, identified by its thread ID.

**Key Points**

- **Parallelism**: The CUDA execution model leverages massive parallelism by dividing work across many threads, warps, and blocks.
- **Latency Hiding**: The warp scheduler can switch between warps to hide memory access latency and keep the SM's execution units busy.
- **Occupancy**: The number of active warps per SM affects occupancy, which is a measure of how effectively the SM's resources are utilized.
- **Synchronization**: Threads within a warp execute in **lock-step**, while synchronization across warps and blocks requires explicit barriers.

Understanding these concepts is crucial for optimizing CUDA programs and effectively utilizing the computational power of NVIDIA GPUs.

Here is a code example that demonstrates the use of both blocks and warps in CUDA C++. This example will implement a simple vector addition, showcasing how blocks are used for data parallelism and how we can leverage warp-level operations for optimization. 

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;  // Lane ID within the warp
    int warpId = threadIdx.x / 32;

    // Vector addition
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }

    // Warp-level sum reduction (for demonstration)
    __shared__ float warpSums[THREADS_PER_BLOCK / 32];
    float sum = c[tid];

    // Warp shuffle to perform reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes result
    if (lane == 0) {
        warpSums[warpId] = sum;
    }

    __syncthreads();

    // Further reduce the warp sums (only using the first warp)
    if (warpId == 0) {
        float warpSum = (lane < (THREADS_PER_BLOCK / 32)) ? warpSums[lane] : 0;
        
        for (int offset = 16; offset > 0; offset /= 2) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }

        if (lane == 0) {
            atomicAdd(&c[n], warpSum);  // Store the total sum at the end of array c
        }
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size + sizeof(float));  // Extra space for sum

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size + sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size + sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += h_c[i];
    }
    printf("Vector addition sum: %f\n", sum);
    printf("Warp-level reduction sum: %f\n", h_c[N]);

    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

If you need code explanation, please refer to [Claude.ai](http://Claude.ai) or [chatGPT](https://chatgpt.com/)

# Finally, Maximize GPU Performance

To maximize GPU performance, it is essential to understand and optimize several key factors: GPU utilization, memory bandwidth, floating-point operations per second (FLOPS), and memory access latency. Here's a detailed breakdown of how to utilize GPU resources effectively and how these factors interrelate:

**Key Factors for Maximizing GPU Performance**

1. **GPU Utilization**
    
    **GPU utilization** measures the percentage of time your GPU is actively processing tasks. High utilization indicates efficient use of the GPU, while low utilization suggests potential bottlenecks or inefficient resource allocation. Tools like NVIDIA’s system management interface (NVIDIA-smi) can help monitor GPU utilization and identify areas for improvement.
    
2. **Memory Bandwidth**
    
    **Memory bandwidth** is the rate at which data can be read from or written to the GPU's memory. It is a critical factor because GPUs need to move large amounts of data quickly to keep the computation cores busy. High memory bandwidth ensures that data transfer between the memory and computation cores is fast enough to prevent bottlenecks. For example, the NVIDIA A100 GPU has a memory bandwidth of 1555 GB/s, which supports its high computational power.
    
3. **Floating-Point Operations Per Second (FLOPS)**
    
    **FLOPS** is a measure of a GPU's computational performance, specifically the number of floating-point operations it can perform per second. Higher FLOPS indicate greater computational power, which is crucial for tasks requiring intensive calculations, such as deep learning and scientific simulations. 
    
4. **Memory Access Latency**
    
    **Memory access latency** is the time it takes for the GPU to access data from its memory. Lower latency means faster data retrieval, which is essential for maintaining high computational throughput. Techniques like memory prefetching and optimizing memory access patterns can help reduce latency and improve overall performance.
    

**Practical Tips for Maximizing GPU Performance**

- **Optimize Data Transfer**: Use high-bandwidth memory (e.g., GDDR6, HBM2) and ensure that the PCIe bandwidth is sufficient for your workload. For instance, running GPUs at PCI-e 4.0 x8 or above typically ensures minimal performance degradation.
- **Efficient Memory Usage**: Optimize memory access patterns to reduce latency. Techniques like memory prefetching and wide loads can help achieve this.
- **Monitor and Adjust Workloads**: Use tools like NVIDIA-smi to monitor GPU utilization and adjust workloads to ensure that the GPU is used efficiently. This may involve tuning batch sizes or distributing tasks more effectively.
- **Leverage Mixed Precision**: For deep learning tasks, using mixed-precision computation can improve performance by reducing the amount of data that needs to be transferred and processed, thus optimizing both bandwidth and computational efficiency.

By understanding and optimizing these factors, you can maximize the performance of your GPU and ensure that it operates efficiently for your specific workloads.
