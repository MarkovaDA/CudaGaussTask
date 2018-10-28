#include<stdio.h>  
#include<math.h> 
#include <stdlib.h>
#include "Common.h" 

int readMatrix(float **matrix) {
	FILE *sourceFile;
    sourceFile = fopen("data.txt", "r");

	int i;
	int numVars = 0;

	fscanf(sourceFile, "%d", &numVars);

	*matrix = (float*)malloc(sizeof(float)*numVars*(numVars+1));
	
	for(i = 0; i < (numVars + 1) * numVars; i++) {
		fscanf(sourceFile, "%f", &(*matrix)[i]);
		//printf("%f\n", *(&(*matrix)[i]));
	}

	return numVars;
}