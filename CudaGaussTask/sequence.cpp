#include "Common.h"

/*������ ���*/
void sequenceForwardPropagation(float **sourceMatrix, int numVars) 
{	
	int size = numVars;
	
	for(int k = 0; k < size - 1; k++) 
	{	//���������� x_i �� ����� k+1..n-1
		
		double pivot = (*sourceMatrix)[k*(size+1) + k];

		for(int i = k + 1; i < size; i++) 
		{	//�� ��������� ������ i ���������� ��������� k
			double lik = (*sourceMatrix)[i*(size+1) + k] / pivot;

			for (int j = k; j < size + 1; j++)
				(*sourceMatrix)[i*(size+1) + j] -= lik * ((*sourceMatrix)[k*(size+1) + j]);
		}
	}
}

/*�������� ���*/
void sequenceBackPropagation(float *triangleMatrix , float **result , int numVars) 
{
	int size = numVars;
	for(int k = size - 1; k >= 0; k--) 
	{
		(*result)[k] = triangleMatrix[k * (size + 1) + size];

		for (int i = k + 1; i < size; i++) {
			(*result)[k] -= triangleMatrix[k * (size+1) + i] * (*result)[i];
		}

		(*result)[k] /= triangleMatrix[k * (size+1) + k];
	}
}
