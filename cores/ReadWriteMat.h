#pragma once
#ifndef READ_MAT_H
#define READ_MAT_H

#include <math.h>
#include <iostream>
#include <string.h>
#include <armadillo>
#include "StructureOrientedFiltering.h"
#include <mat.h>

using namespace std;
using namespace arma;

Cube<double> ReadMat2Arma3D(string sourceFile, const char *variableName, int *inlineNum, int *xlineNum, int *times)
{
/*
	Discription : "ReadMat2Arma3D()" read MATLAB .mat file and convert it into armadillo formats.
	Here the defalut size of input is 3D, and we think the dimension is inlineNum * xlineNum * times.
	
	Editor : yyh
	Edit Date: 2019-08-28
*/

	MATFile *_pmatFile = matOpen(sourceFile.c_str(), "r");
	if (_pmatFile == NULL)
	{
		cerr << "Error : file don't exists OR your path has something wrong!";
		throw("Error : file don't exists OR your path has something wrong!");
	}
	mxArray *_mxArray = matGetVariable(_pmatFile, variableName);
	if (_mxArray == NULL)
	{
		cerr << "Error : variable name has something wrong!";
		throw("Error : variable name has something wrong!");
	}
	*inlineNum = mxGetDimensions(_mxArray)[0];
	*xlineNum = mxGetDimensions(_mxArray)[1];
	*times = mxGetDimensions(_mxArray)[2];

	Cube<double> data3D(*inlineNum, *xlineNum, *times);

	double *dataArray = mxGetPr(_mxArray);

	for (int k = 0; k < *times; k++)
	{
		for (int j = 0; j < *xlineNum; j++)
		{
			for (int i = 0; i < *inlineNum; i++)
			{
				data3D(i, j, k) = dataArray[k * *inlineNum * *xlineNum + j * *inlineNum + i];
			}
		}
	}

	matClose(_pmatFile);

	cout << "Reading and conversion completed. Data dimension is " << *inlineNum << " x " << *xlineNum << " x " << *times << '.' << endl;

	return data3D;
}

void WriteMatFromArma3D(string destFile, cube &data3D, const char *variableName, int *inlineNum, int *xlineNum, int *times)
{
/*
	Discription : "WriteMatFromArma3D()" convert armadillo formats into Matlab .mat format.
	Here the defalut size of input is 3D, and we think the dimension is inlineNum * xlineNum * times.
	
	Editor : yyh
	Edit Date: 2019-08-28
*/
	MATFile *_pOutFile = matOpen(destFile.c_str(), "w");

	const SizeCube size = arma::size(data3D);

	*inlineNum = size[0];

	*xlineNum = size[1];

	*times = size[2];

	const size_t dims[3] = { *inlineNum, *xlineNum, *times };

	mxArray *_mxArray = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);

	double *dataArray = mxGetPr(_mxArray);

	for (int k = 0; k < *times; k++)
	{
		for (int j = 0; j < *xlineNum; j++)
		{
			for (int i = 0; i < *inlineNum; i++)
			{
				dataArray[k * *inlineNum * *xlineNum + j * *inlineNum + i] = data3D(i, j, k);
			}
		}
	}
	matPutVariable(_pOutFile, variableName, _mxArray);

	matClose(_pOutFile);

	cout << "Writing completed. Data dimension is " << *inlineNum << " x " << *xlineNum << " x " << *times << '.' << endl;
}

#endif