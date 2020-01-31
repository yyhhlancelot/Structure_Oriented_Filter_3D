#pragma once
#ifndef STRUCTURE_ORIENTED_FILTERING_H
#define STRUCTURE_ORIENTED_FILTERING_H

#include <iostream>
#include <string.h>
#include <armadillo>
#include "read_mat.h"
#include <vector>
#include <mat.h>
#include <numeric>
using namespace std;
using namespace arma;

void GradientOfCube(Cube<double> data3D, Cube<double> &InlineGradient, Cube<double> &XlineGradient, Cube<double> &TimesGradient)
{
	/*
		Discription : "GradientOfCube()" compute 3D data' gradient while the data dimension is
		'inlineNum * xlineNum * times' which we default.
		
		Editor : yyh
		Edit Date: 2019-08-28
	*/
	const SizeCube _size = arma::size(data3D);
	const int inlineNum = _size[0];
	const int xlineNum = _size[1];
	const int times = _size[2];
	
	cube _g = zeros<cube>(_size);
	
	vector<vector<int>> loc;
	
	for (int i = 0; i < 3; i++)
	{
		vector<int> tmpVec(_size[i]);
		
		iota(begin(tmpVec), end(tmpVec), 1);
		
		loc.push_back(tmpVec);
	}
	
	vector<int> _h = loc[0];
	
	int _n = _size[0];
	
	if (_n > 1)
	{
		_g(span(0, 0), span(0, xlineNum - 1), span(0, times - 1)) = 
			(data3D(span(1, 1), span(0, xlineNum - 1), span(0, times - 1)) - data3D(span(0, 0), span(0, xlineNum - 1), span(0, times - 1)))
			/ (_h[1] - _h[0]);

		_g(span(inlineNum - 1, inlineNum - 1), span(0, xlineNum - 1), span(0, times - 1)) = 
			(data3D(span(inlineNum - 1, inlineNum - 1), span(0, xlineNum - 1), span(0, times - 1)) - 
				data3D(span(inlineNum - 2, inlineNum - 2), span(0, xlineNum - 1), span(0, times - 1)))
			/ (_h.back() - *(_h.end() - 2));
	}
	
	if (_n > 2)
	{
		for (int i = 1; i <= inlineNum - 2; i++)
		{
			_g(span(i, i), span(0, xlineNum - 1), span(0, times - 1)) =
				(data3D(span(i + 1, i + 1), span(0, xlineNum - 1), span(0, times - 1)) - data3D(span(i - 1, i - 1), span(0, xlineNum - 1), span(0, times - 1)))
				/ (_h[i + 1] - _h[i - 1]);
		}
	}

	InlineGradient = _g;
	
	for (int i = 1; i <= 2; i++)
	{
		_n = _size[i];
		
		Cube<double> data3D_re;
		
		if (i == 1)
			data3D_re = reshape(data3D, _size[0], _size[1], _size[2]);
		else if (i == 2)
			data3D_re = reshape(data3D, _size[0] * _size[1], _size[2], 1);
		
		_h = loc[i];
		
		_g = zeros<cube>(arma::size(data3D_re));
		
		SizeCube _newsize = arma::size(data3D_re);
		
		if (_n > 1)
		{
			_g(span(0, _newsize[0] - 1), span(0, 0), span(0, _newsize[2] - 1)) =
				(data3D_re(span(0, _newsize[0] - 1), span(1, 1), span(0, _newsize[2] - 1)) - 
					data3D_re(span(0, _newsize[0] - 1), span(0, 0), span(0, _newsize[2] - 1)))
				/ (_h[1] - _h[0]);

			_g(span(0, _newsize[0] - 1), span(_newsize[1] - 1, _newsize[1] - 1), span(0, _newsize[2] - 1)) =
				(data3D_re(span(0, _newsize[0] - 1), span(_newsize[1] - 1, _newsize[1] - 1), span(0, _newsize[2] - 1)) -
					data3D_re(span(0, _newsize[0] - 1), span(_newsize[1] - 2, _newsize[1] - 2), span(0, _newsize[2] - 1)))
				/ (_h.back() - *(_h.end() - 2));
		}
		
		if (_n > 2)
		{
			for (int i = 1; i <= _n - 2; i++)
			{
				_g(span(0, _newsize[0] - 1), span(i, i), span(0, _newsize[2] - 1)) =
					(data3D_re(span(0, _newsize[0] - 1), span(i + 1, i + 1), span(0, _newsize[2] - 1)) -
						data3D_re(span(0, _newsize[0] - 1), span(i - 1, i - 1), span(0, _newsize[2] - 1)))
					/ (_h[i + 1] - _h[i - 1]);
			}
		}
		if (i == 1)
			XlineGradient = reshape(_g, _size[0], _size[1], _size[2]);
		else if (i == 2)
			TimesGradient = reshape(_g, _size[0], _size[1], _size[2]);
	}
}

void GaussianLowpassFilter(const double sigma_tgf, const double tgf_sidelen, vector<double>& tgf_coef, vec& tgf_coef_vec)
{
	if (!tgf_coef.empty())
	{
		cerr << "coef gets some wrong." << endl;
		throw("coef gets some wrong.");
	}
	
	for (int i = -tgf_sidelen; i <= tgf_sidelen; i++)
	{
		tgf_coef.push_back(exp(-1 / (2 * pow(sigma_tgf, 2)) * pow(i, 2)));
	}
	vec tmp = conv_to<vec>::from(tgf_coef);

	double sum = accumulate(tgf_coef.begin(), tgf_coef.end(), double(0));

	for (auto it = tgf_coef.begin(); it != tgf_coef.end(); it++)
	{
		*it = *it / sum;
	}

	tgf_coef_vec = conv_to<vec>::from(tgf_coef);

}

cube InlineFilter(cube& SynSeismicVolume, const int inlineNum, const int xlineNum, const int times, vec GaussFilterType)
{
	cube InlineGradientVectorVol = zeros<cube>(arma::size(SynSeismicVolume));
	for (int x_index = 0; x_index < xlineNum; x_index++)
	{
		for (int t_index = 0; t_index < times; t_index++)
		{
			vec tmp = SynSeismicVolume(span(0, inlineNum - 1), span(x_index, x_index), span(t_index, t_index));
			
			InlineGradientVectorVol(span(0, inlineNum - 1), span(x_index, x_index), span(t_index, t_index)) =
				conv(tmp, GaussFilterType, "same");
			tmp.clear();
		}
	}
	return InlineGradientVectorVol;
}

cube XlineFilter(cube& SynSeismicVolume, const int inlineNum, const int xlineNum, const int times, vec GaussFilterType)
{
	cube XlineGradientVectorVol = zeros<cube>(arma::size(SynSeismicVolume));
	for (int in_index = 0; in_index < inlineNum; in_index++)
	{
		for (int t_index = 0; t_index < times; t_index++)
		{
			rowvec tmp = SynSeismicVolume(span(in_index, in_index), span(0, xlineNum - 1), span(t_index, t_index));
			XlineGradientVectorVol(span(in_index, in_index), span(0, xlineNum - 1), span(t_index, t_index)) =
				conv(tmp, GaussFilterType, "same");
		}
	}
	return XlineGradientVectorVol;
}

cube TimeFilter(cube& SynSeismicVolume, const int inlineNum, const int xlineNum, const int times, vec GaussFilterType)
{
	cube TimeGradientVectorVol = zeros<cube>(arma::size(SynSeismicVolume));
	for (int in_index = 0; in_index < inlineNum; in_index++)
	{
		for (int x_index = 0; x_index < xlineNum; x_index++)
		{
			vec tmp = SynSeismicVolume(span(in_index, in_index), span(x_index, x_index), span(0, times - 1));
			TimeGradientVectorVol(span(in_index, in_index), span(x_index, x_index), span(0, times - 1)) =
				conv(tmp, GaussFilterType, "same");
		}
	}
	return TimeGradientVectorVol;
}

void GradTensor(Cube<double> &InlineGradient, Cube<double> &XlineGradient, Cube<double> &TimesGradient, double sigma_tgf, double sigma_tgf_t,
	cube& T00_af_f, cube& T01_af_f, cube& T02_af_f, cube& T11_af_f, cube& T12_af_f, cube& T22_af_f)
{
/*
	Discription : "GradTensor()".

	Convolution processing and get 6 tensor.

	Here the defalut size of input is 3D, and we think the dimension is inlineNum * xlineNum * times.

	Editor : yyh
	Edit Date: 2019-08-28
*/
	const SizeCube _size = arma::size(InlineGradient);
	const int xlineNum = _size[0];
	const int inlineNum = _size[1];
	const int times = _size[2];
	cube T00 = zeros<cube>(_size);
	cube T01 = T00; cube T02 = T00; cube T11 = T00; cube T12 = T00; cube T22 = T00;

	for (int t = 0; t < times; t++)
	{
		for (int n = 0; n < inlineNum; n++)
		{
			for (int m = 0; m < xlineNum; m++)
			{
				vec _g = { InlineGradient(m, n, t), XlineGradient(m, n, t), TimesGradient(m, n, t) };
				mat T = _g * trans(_g);
				T00(m, n, t) = T(0, 0);
				T01(m, n, t) = T(0, 1);
				T02(m, n, t) = T(0, 2);
				T11(m, n, t) = T(1, 1);
				T12(m, n, t) = T(1, 2);
				T22(m, n, t) = T(2, 2);
			}
		}
	}
	const double tgf_sidelen = ceil(4 * sqrt(sigma_tgf));
	const double tgf_t_sidelen = ceil(4 * sqrt(sigma_tgf_t));

	vector<double> tgf_coef, tgf_coef_t;
	vec _tgf_coef, _tgf_coef_t;
	GaussianLowpassFilter(sigma_tgf, tgf_sidelen, tgf_coef, _tgf_coef);
	GaussianLowpassFilter(sigma_tgf_t, tgf_t_sidelen, tgf_coef_t, _tgf_coef_t);

	T00_af_f = InlineFilter(T00, xlineNum, inlineNum, times, _tgf_coef);
	T00_af_f = XlineFilter(T00_af_f, xlineNum, inlineNum, times, _tgf_coef);
	T00_af_f = TimeFilter(T00_af_f, xlineNum, inlineNum, times, _tgf_coef_t);
	T00.clear();

	T01_af_f = InlineFilter(T01, xlineNum, inlineNum, times, _tgf_coef);
	T01_af_f = XlineFilter(T01_af_f, xlineNum, inlineNum, times, _tgf_coef);
	T01_af_f = TimeFilter(T01_af_f, xlineNum, inlineNum, times, _tgf_coef_t);
	T01.clear();

	T02_af_f = InlineFilter(T02, xlineNum, inlineNum, times, _tgf_coef);
	T02_af_f = XlineFilter(T02_af_f, xlineNum, inlineNum, times, _tgf_coef);
	T02_af_f = TimeFilter(T02_af_f, xlineNum, inlineNum, times, _tgf_coef_t);
	T02.clear();

	T11_af_f = InlineFilter(T11, xlineNum, inlineNum, times, _tgf_coef);
	T11_af_f = XlineFilter(T11_af_f, xlineNum, inlineNum, times, _tgf_coef);
	T11_af_f = TimeFilter(T11_af_f, xlineNum, inlineNum, times, _tgf_coef_t);
	T11.clear();

	T12_af_f = InlineFilter(T12, xlineNum, inlineNum, times, _tgf_coef);
	T12_af_f = XlineFilter(T12_af_f, xlineNum, inlineNum, times, _tgf_coef);
	T12_af_f = TimeFilter(T12_af_f, xlineNum, inlineNum, times, _tgf_coef_t);
	T12.clear();

	T22_af_f = InlineFilter(T22, xlineNum, inlineNum, times, _tgf_coef);
	T22_af_f = XlineFilter(T22_af_f, xlineNum, inlineNum, times, _tgf_coef);
	T22_af_f = TimeFilter(T22_af_f, xlineNum, inlineNum, times, _tgf_coef_t);
	T22.clear();

}

void DiffBySobel(cube& Vol1, cube& Vol2, cube& Vol3, cube& sobel_x, cube& sobel_y, cube& sobel_z)
{
	const SizeCube size = arma::size(Vol1);

	const int inlineNum = size[0];

	const int xlineNum = size[1];

	const int times = size[2];

	cube sT_x = zeros<cube>(3, 3, 3);
	cube sT_y = zeros<cube>(3, 3, 3);
	cube sT_z = zeros<cube>(3, 3, 3);

	sT_x.slice(0) = { {-2, -4, -2}, {0, 0, 0}, {2, 4, 2} };
	sT_x.slice(1) = { {-4, -8, -4}, {0, 0, 0}, {4, 8, 4} };
	sT_x.slice(2) = { {-2, -4, -2}, {0, 0, 0}, {2, 4, 2} };

	sT_y.slice(0) = { {-2, 0, 2}, {-4, 0, 4}, {-2, 0, 2} };
	sT_y.slice(1) = { {-4, 0, 4}, {-8, 0, 8}, {-4, 0, 4} };
	sT_y.slice(2) = { {-2, 0, 2}, {-4, 0, 4}, {-2, 0, 2} };

	sT_z.slice(0) = { {-2, -4, -2}, {-4, -8, -4}, {-2, -4, -2} };
	sT_z.slice(1) = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
	sT_z.slice(2) = { {2, 4, 2}, {4, 8, 4}, {2, 4, 2} };

	for (int i = 1; i <= inlineNum - 2; i++)
	{
		for (int j = 1; j <= xlineNum - 2; j++)
		{
			for (int k = 1; k <= times - 2; k++)
			{
				double v1 = 0;
				double v2 = 0;
				double v3 = 0;

				for (int m = 0; m <= 2; m++)
				{
					for (int n = 0; n <= 2; n++)
					{
						for (int p = 0; p <= 2; p++)
						{
							v1 += sT_x(m, n, p) * Vol1(i - 1 + m, j - 1 + n, k - 1 + p);
							v2 += sT_y(m, n, p) * Vol2(i - 1 + m, j - 1 + n, k - 1 + p);
							v3 += sT_z(m, n, p) * Vol3(i - 1 + m, j - 1 + n, k - 1 + p);
						}
					}
				}

				if (isnan(v1) || isinf(v1))
					v1 = 0;
				if (isnan(v2) || isinf(v2))
					v2 = 0;
				if (isnan(v3) || isinf(v3))
					v3 = 0;

				sobel_x(i, j, k) = v1;
				sobel_y(i, j, k) = v2;
				sobel_z(i, j, k) = v3;

			}
		}
	}

}

void StructureOrientedFiltering(const double sigma_tgf, const double sigma_tgf_t, const double C, const double N, Cube<double>& data3D)
{
/*
	Discription : "StructureOrientedFiltering()".

	Const parameters can be changed outside of the function.

	parameter meaning;
	sigma_tgf/sigma_tgf_t : param of timeline gaussian filter, can be changed to get better performance :)
	N : iteration nums.
	C : a const num.

	Here the defalut size of input is 3D, and we think the dimension is inlineNum * xlineNum * times.
	
	Editor : yyh
	Edit Date: 2019-08-28
*/
	cout << "Start processing..." << endl;

	const SizeCube size = arma::size(data3D);
	
	const int inlineNum = size[0];
	
	const int xlineNum = size[1];
	
	const int times = size[2];
	
	Cube<double> InlineGradient;
	
	Cube<double> XlineGradient;
	
	Cube<double> TimesGradient;
	
	// compute 3 gradients of each direction.
	GradientOfCube(data3D, InlineGradient, XlineGradient, TimesGradient);

	cube T11_af_f0, T12_af_f0, T13_af_f0, T22_af_f0, T23_af_f0, T33_af_f0;
	
	GradTensor(InlineGradient, XlineGradient, TimesGradient, sigma_tgf, sigma_tgf_t,
		T11_af_f0, T12_af_f0, T13_af_f0, T22_af_f0, T23_af_f0, T33_af_f0);

	cube T11_af_f, T12_af_f, T13_af_f, T22_af_f, T23_af_f, T33_af_f;
	
	// update the amplitude
	for (int iter = 0; iter < N; iter++)
	{
		GradientOfCube(data3D, InlineGradient, XlineGradient, TimesGradient);
		
		GradTensor(InlineGradient, XlineGradient, TimesGradient, sigma_tgf, sigma_tgf_t,
			T11_af_f, T12_af_f, T13_af_f, T22_af_f, T23_af_f, T33_af_f);

		cube div_vol_x = zeros<cube>(size);
		cube div_vol_y = zeros<cube>(size);
		cube div_vol_z = zeros<cube>(size);

		cube J0 = zeros<cube>(size);
		cube J1 = zeros<cube>(size);
		cube J2 = zeros<cube>(size);

		for (int t = 0; t < times; t++)
		{
			for (int n = 0; n < xlineNum; n++)
			{
				for (int m = 0; m < inlineNum; m++)
				{
					if (isnan(T11_af_f(m, n, t)) || isinf(T11_af_f(m, n, t)))
					{
						T11_af_f(m, n, t) = 0;
					}

					if (isnan(T12_af_f(m, n, t)) || isinf(T12_af_f(m, n, t)))
					{
						T12_af_f(m, n, t) = 0;
					}

					if (isnan(T13_af_f(m, n, t)) || isinf(T13_af_f(m, n, t)))
					{
						T13_af_f(m, n, t) = 0;
					}

					if (isnan(T22_af_f(m, n, t)) || isinf(T22_af_f(m, n, t)))
					{
						T22_af_f(m, n, t) = 0;
					}

					if (isnan(T23_af_f(m, n, t)) || isinf(T23_af_f(m, n, t)))
					{
						T23_af_f(m, n, t) = 0;
					}

					if (isnan(T33_af_f(m, n, t)) || isinf(T33_af_f(m, n, t)))
					{
						T33_af_f(m, n, t) = 0;
					}

					mat T_af_f = { {T11_af_f(m, n, t), T12_af_f(m, n, t), T13_af_f(m, n, t)},
					{T12_af_f(m, n, t), T22_af_f(m, n, t), T23_af_f(m, n, t)},
					{T13_af_f(m, n, t), T23_af_f(m, n, t), T33_af_f(m, n, t)} };

					vec eig_val;
					mat eig_mat;

					eig_sym(eig_val, eig_mat, T_af_f);

					vec v0 = eig_mat.col(0);
					vec v1 = eig_mat.col(1);

					mat D_mat = v0 * v0.t() + v1 * v1.t();

					mat S0_mat = { {T11_af_f0(m, n, t), T12_af_f0(m, n, t), T13_af_f0(m, n, t)},
					{T12_af_f0(m, n, t), T22_af_f0(m, n, t), T23_af_f0(m, n, t)},
					{T13_af_f0(m, n, t), T23_af_f0(m, n, t), T33_af_f0(m, n, t)} };

					mat S0S1 = S0_mat * T_af_f;

					double f = (sum(S0S1.diag())) / (sum(S0_mat.diag()) * sum(T_af_f.diag()));
					
					vec deltaU = { InlineGradient(m, n, t), XlineGradient(m, n, t), TimesGradient(m, n, t) };

					J0(m, n, t) = (D_mat(0, 0) * deltaU(0) + D_mat(0, 1) * deltaU(1) + D_mat(0, 2) * deltaU(2)) * f;
					J1(m, n, t) = (D_mat(1, 0) * deltaU(0) + D_mat(1, 1) * deltaU(1) + D_mat(1, 2) * deltaU(2)) * f;
					J2(m, n, t) = (D_mat(2, 0) * deltaU(0) + D_mat(2, 1) * deltaU(1) + D_mat(2, 2) * deltaU(2)) * f;

				}
			}
		}

		// gradients of each direction
		cube sobel_x = zeros<cube>(size);
		cube sobel_y = zeros<cube>(size);
		cube sobel_z = zeros<cube>(size);

		DiffBySobel(J0, J1, J2, sobel_x, sobel_y, sobel_z);

		// compute the divergence
		cube div_vol = sobel_x + sobel_y + sobel_z;

		// update the amplitude
		data3D += C * div_vol;

	}
	cout << "Process finished." << endl;
}

#endif
