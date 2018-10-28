
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include "Common.h"

__global__ void Kernel(float *, float *, int); 

void DeviceFunc(float *temp_h , int numvar , float *temp1_h) 
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
    Kernel<<<dimGrid , dimBlock>>>(a_d, b_d, numvar); 
    
    //Coping data to host from device 
    cudaMemcpy(temp1_h,b_d, sizeof(float)*numvar*(numvar+1), cudaMemcpyDeviceToHost);
    cudaFree(a_d); 
    cudaFree(b_d);
}

__global__ void Kernel(float *a_d , float *b_d ,int size) 
{
    int idx = threadIdx.x ; 
    int idy = threadIdx.y ; 
    
    //Allocating memory in the share memory of the device 
    __shared__ float temp[16][16]; 
    
    //Copying the data to the shared memory 
    temp[idy][idx] = a_d[(idy * (size+1)) + idx] ; 
    
    for(int i =1 ; i<size ;i++) 
    { 
        if((idy + i) < size) // NO Thread divergence here 
        { 
            float var1 =(-1)*( temp[i-1][i-1]/temp[i+idy][i-1]); 
            temp[i+idy][idx] = temp[i-1][idx] +((var1) * (temp[i+idy][idx]));
        } 
        __syncthreads(); //Synchronizing all threads before Next iterat ion 
    } 
    
    b_d[idy*(size+1) + idx] = temp[idy][idx]; 
}

int main()
{	
	float *a_h = NULL;
    float *b_h = NULL;

    float *result, sum, rvalue; 
    int numvar, j;
	
	numvar = readMatrix(&a_h);	
	b_h = (float*)malloc(sizeof(float) * numvar * (numvar+1));
	
	DeviceFunc(a_h , numvar , b_h); 
    
    //printf("End \n"); 
    
	result = (float*)malloc(sizeof(float)*(numvar)); 
    for(int i = 0; i< numvar;i++) 
    { 
        result[i] = 1.0;
    }

    for(int i=numvar-1 ; i>=0 ; i--) 
    { 
        sum = 0.0 ;

        for(j=numvar-1; j > i ;j--) 
        { 
            sum = sum + result[j]*b_h[i*(numvar+1) + j]; 
        }
        rvalue = b_h[i*(numvar+1) + numvar] - sum ; 
        result[i] = rvalue / b_h[i *(numvar+1) + j];
    } 
    
    //Displaying the result 
    for(int i =0; i < numvar;i++) 
    { 
        printf("[X%d] = %+f\n", i ,result[i]); 
    }

	system("pause");
    return 0;
}
