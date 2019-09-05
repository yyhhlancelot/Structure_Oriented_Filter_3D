/*
	Discription : "main()" include a demo of 3D seismic data processing using Structure Oriented Filter.
	
	The linear algebra library we use is Armadillo, the version we use is armadillo-9.600.6.
	
	The code is written under x64, please NOT running under x86.

	The pipeline of the demo is: 
	1. read.mat -> armadillo formal(cube)
	2. algorithm processing
	3. write & store

	Here the defalut size of input is 3D, and we think the dimension is inlineNum * xlineNum * times.
	
	Editor : yyh
	Edit Date: 2019-08-28
*/

#include "pch.h"
#include <iostream>
#include <math.h>
#include <string>
#include "armadillo"
#include <mat.h>
#include "read_mat.h"
#include "StructureOrientedFiltering.h"

using namespace std;
using namespace arma;

int main()
{
	// your data file index
	string _sourceFile = "D:\\code\\project\\structrue_oriented_filtering\\structure_oriented_filter_matlab\\QX_370_870_286_586_1980_2020.mat";
	string _destFile = "D:\\code\\project\\structrue_oriented_filtering\\structure_oriented_filter_matlab\\QX_370_870_286_586_1980_2020_AfterDenoise.mat";
	
	// variable name when we read and write
	const char* _variableName0 = "Vol";
	const char* _variableName1 = "Vol_AfterDenoise";

	int* _inlineNum = new int(0);
	int* _xlineNum = new int(0);
	int* _times = new int(0);
	
	// read .mat format and convert it into armadillo format
	Cube<double> _data3D = ReadMat2Arma3D(_sourceFile, _variableName0, _inlineNum, _xlineNum, _times);

	// cut part of the data to test and speed the process
	Cube<double> _dataTest = _data3D(span(0, 99), span(0, 99), span(0, *_times - 1));

	_data3D.clear();

	// const parameters of structure oriented filtering
	const double _sigma_tgf = 0.6;
	const double _sigma_tgf_t = 0.6;
	const double _C = 0.05;
	const int _N = 5;

	// algorithm function
	StructureOrientedFiltering(_sigma_tgf, _sigma_tgf_t, _C, _N, _dataTest);

	// write & store
	WriteMatFromArma3D(_destFile, _dataTest, _variableName1, _inlineNum, _xlineNum, _times);

	return 0;
}
