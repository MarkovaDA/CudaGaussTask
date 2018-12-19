
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "Common.h"

__global__ void forwardPropagation(float *, float *, int); 

void calculateTriangleMatrix(float *temp_h , int numvar , float *temp1_h) 
{ 
    float *a_d , *b_d; 

    //Memory allocation on the device 
    cudaMalloc(&a_d,sizeof(float)*(numvar)*(numvar+1)); 
    cudaMalloc(&b_d,sizeof(float)*(numvar)*(numvar+1)); 
    
    //Copying data to device from host 
    cudaMemcpy(a_d, temp_h, sizeof(float)*numvar*(numvar+1),cudaMemcpyHostToDevice); 
    
    //Defining size of Thread Block 
    dim3 dimBlock(numvar + 1, numvar, 1); 
    dim3 dimGrid(1,1,1); 
    //Kernel call 
    forwardPropagation<<<dimGrid , dimBlock>>>(a_d, b_d, numvar);
    //Coping data to host from device 
    cudaMemcpy(temp1_h,b_d, sizeof(float)*numvar*(numvar+1), cudaMemcpyDeviceToHost);
    cudaFree(a_d); 
    cudaFree(b_d);
}

__global__ void forwardPropagation(float *a_d , float *b_d ,int size) 
{
    int idx = threadIdx.x; 
    int idy = threadIdx.y; 
    
    __shared__ float temp[16][16]; 
    
    temp[idy][idx] = a_d[(idy * (size+1)) + idx] ; 
    
    for(int i = 1 ; i < size; i++) 
    { 
        if((idy + i) < size) 
        { 
            float var1 =(-1)*(temp[i-1][i-1] / temp[i+idy][i-1]); 
            temp[i+idy][idx] = temp[i-1][idx] + ((var1) * (temp[i+idy][idx]));
        } 
        __syncthreads();
    } 
    
    b_d[idy*(size+1) + idx] = temp[idy][idx]; 
}

__host__ void backPropagation(float **result, float *b_h, int numvar) {
	float sum, rvalue; 
	int j;
	*result = (float*)malloc(sizeof(float)*(numvar)); 
    
	for(int i = 0; i < numvar;i++) 
    { 
        (*result)[i] = 1.0;
    }

    for(int i=numvar-1 ; i>=0 ; i--) 
    { 
        sum = 0.0 ;
		#pragma omp parallel for reduction(+:sum) private(j, numvar) shared(result, b_h)
        for(j = numvar-1; j > i ;j--) 
        { 
            sum = sum + (*result)[j]*b_h[i*(numvar+1) + j]; 
        }
        rvalue = b_h[i*(numvar+1) + numvar] - sum; 
        (*result)[i] = rvalue / b_h[i *(numvar+1) + j];
    } 
}

int main()
{	
	float *a_h = NULL;
    float *b_h = NULL;

    float *result, sum, rvalue; 
    int numvar, j;
	
	numvar = readMatrix(&a_h);	
	b_h = (float*)malloc(sizeof(float) * numvar * (numvar+1));
	result = (float*)malloc(sizeof(float)*(numvar));
	float gpuTime = 0.0f;
	
	/***œŒ—À≈ƒŒ¬¿“≈À‹Õ¿ﬂ ¬≈–—»ﬂ***/
	/*
	double start, end;
	start = omp_get_wtime();

	sequenceForwardPropagation(&a_h, numvar);
	sequenceBackPropagation(a_h, &result, numvar);

	end = omp_get_wtime();
	printf("Work time %f mc\n", (end-start)*1000);
	for(int i = 0; i < numvar; i++) 
    { 
        printf("[X%d] = %+f\n", i , result[i]); 
    }*/

	/***œ¿–¿ÀÀ≈À‹Õ¿ﬂ ¬≈–—»ﬂ***/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	calculateTriangleMatrix(a_h , numvar , b_h); 

	cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&gpuTime, start, stop);
        
	result = (float*)malloc(sizeof(float)*(numvar)); 

    for(int i = 0; i < numvar;i++) 
    { 
        result[i] = 1.0;
    }

    backPropagation(&result, b_h, numvar);
	cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);

	#pragma omp parallel for
    for(int i = 0; i < numvar; i++) 
    { 
        printf("[X%d] = %+f\n", i ,result[i]); 
    }

	system("pause");
    return 0;
}
