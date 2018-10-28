#include<stdio.h>  
#include<math.h> 
#include <stdlib.h>
#include "Common.h" 

int readMatrix(float **matrix) {
	FILE *sourceFile;
    sourceFile = fopen("data10.txt", "r");

	int i;
	int numVars = 0;

	fscanf(sourceFile, "%d", &numVars);

	*matrix = (float*)malloc(sizeof(float)*numVars*(numVars+1));
	
	float value = 0;
	for(int i = 0; i < numVars; i++) 
	{
		for(int j = 0; j < numVars + 1; j++) 
		{
			fscanf(sourceFile, "%f", &value);
			//printf("%f ", value);

			(*matrix)[i * (numVars + 1) + j] = value;
		}
		printf("\n");
	}

	/*for(i = 0; i < (numVars + 1) * numVars; i++) {
		fscanf(sourceFile, "%f", &(*matrix)[i]);
		//printf("%f\n", *(&(*matrix)[i]));
	}*/

	return numVars;
}