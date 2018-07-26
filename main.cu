#include <stdio.h>
#include <Windows.h>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\remove.h>
#include <thrust\execution_policy.h>
#include <thrust\remove.h>
#include <thrust\copy.h>
#include <thrust\extrema.h>
#include <list>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <ctime>
extern "C" {
#include <math.h>
#include "floating_number_helper.h"
#include "input_output.h"
}
#define INIT_LINE 100000
#define ERROR 0.000001
#define _CRT_SECURE_NO_WARNINGS
#define M_PI 3.14159265358979323846

//
// this function is what you need to finish
// @Usage : to solve the problem
// @Input : input containing all data needed
// @Output: answer containing all necessary data
//  you can find the definition of the two structs above in
//      input_output.h
//

double rm_error(double a) {
	long long b = a * 1000000000000;
	return b / 1000000000000.0;
}

struct rotation {
	double a0;
	double b0;

	rotation(double x, double y) { a0 = x, b0 = y; }

	__host__ __device__
		thrust::tuple<double, double, double> operator()(thrust::tuple<double, double, double> v) {
		double a = thrust::get<0>(v);
		double b = thrust::get<1>(v);
		double c = thrust::get<2>(v);
		double theta = atan2(b0, a0);
		double alpha = atan2(b, a);
		theta = M_PI / 2 - theta;
		double new_a = cos(theta)*a - sin(theta)*b;
		double new_b = sin(theta)*a + cos(theta)*b;
		double new_c = c;
		return thrust::make_tuple(new_a, new_b, new_c);
	}
};

struct if_plus {
	__host__ __device__
		bool operator()(thrust::tuple<double, double, double> v) {
		double b = thrust::get<1>(v);
		return (b > 0);
	}
};

struct if_minus {
	__host__ __device__
		bool operator()(thrust::tuple<double, double, double> v) {
		double b = thrust::get<1>(v);
		return (b < 0);
	}
};

struct slope {
	__host__ __device__
		line operator()(line v) {
		v.slope_value = -v.param_a/v.param_b;;
		return v;
	}
};

struct height {
	double x;
	height(double a) { x = a; }
	__host__ __device__
		double operator()(line v) {
		double a = v.param_a;
		double b = v.param_b;
		double c = v.param_c;
		double y = (c - a*x) / b;
		return y;
	}
};

typedef struct boundary {
	double x;
	double y_plus;
	double y_minus;
	line line_plus;
	line line_minus;
}boundary;

answer * compute(inputs * input) {
	int num = input->number;
	line ** lines = input->lines;
	answer * ans = (answer *)malloc(sizeof(answer));
	double start, stop, time;


	//
	// @todo write your codes here!
	//

	thrust::host_vector<double> h_a(num);
	thrust::host_vector<double> h_b(num);
	thrust::host_vector<double> h_c(num);

	for (int i = 0; i < num; i++) {
		h_a[i] = lines[i]->param_a;
		h_b[i] = lines[i]->param_b;
		h_c[i] = lines[i]->param_c;
	}

	thrust::device_vector<double> a = h_a;
	thrust::device_vector<double> b = h_b;
	thrust::device_vector<double> c = h_c;

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin(), c.begin()));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end(), c.end()));

	double ao = input->obj_function_param_a;
	double bo = input->obj_function_param_b;

	rotation pred(ao, bo);
	thrust::transform(begin, end, begin, pred);

	thrust::device_vector<double> a_plus(num);
	thrust::device_vector<double> b_plus(num);
	thrust::device_vector<double> c_plus(num);
	thrust::device_vector<double> a_minus(num);
	thrust::device_vector<double> b_minus(num);
	thrust::device_vector<double> c_minus(num);

	auto plus_begin  = thrust::make_zip_iterator(thrust::make_tuple(a_plus.begin(), b_plus.begin(), c_plus.begin()));
	auto minus_begin = thrust::make_zip_iterator(thrust::make_tuple(a_minus.begin(), b_minus.begin(), c_minus.begin()));
	auto plus_end    = thrust::make_zip_iterator(thrust::make_tuple(a_plus.end(), b_plus.end(), c_plus.end()));
	auto minus_end   = thrust::make_zip_iterator(thrust::make_tuple(a_minus.end(), b_minus.end(), c_minus.end()));

	plus_end = thrust::copy_if(begin, end, plus_begin, if_plus());
	minus_end = thrust::copy_if(begin, end, minus_begin, if_minus());

	int plus_count = num - thrust::count(a_plus.begin(), a_plus.end(), 0);
	int minus_count = num - thrust::count(a_minus.begin(), a_minus.end(), 0);

	//这里还需释放cuda内存
	/*
	thrust::device_vector<double> a_Ip(plus_count);
	thrust::device_vector<double> b_Ip(plus_count);
	thrust::device_vector<double> c_Ip(plus_count);
	thrust::device_vector<double> a_Im(minus_count);
	thrust::device_vector<double> b_Im(minus_count);
	thrust::device_vector<double> c_Im(minus_count);
	*/
	
	a_plus.erase(a_plus.begin() + plus_count, a_plus.begin() + num);
	b_plus.erase(b_plus.begin() + plus_count, b_plus.begin() + num);
	c_plus.erase(c_plus.begin() + plus_count, c_plus.begin() + num);
	a_minus.erase(a_minus.begin() + minus_count, a_minus.begin() + num);
	b_minus.erase(b_minus.begin() + minus_count, b_minus.begin() + num);
	c_minus.erase(c_minus.begin() + minus_count, c_minus.begin() + num);


	thrust::host_vector<line> h_I_plus(plus_count);
	thrust::host_vector<line> h_I_minus(minus_count);

	start = std::clock();
	for (int i = 0; i < plus_count; i++) {
		h_I_plus[i].param_a = a_plus[i];
		h_I_plus[i].param_b = b_plus[i];
		h_I_plus[i].param_c = c_plus[i];
	}
	for (int i = 0; i < minus_count; i++) {
		h_I_minus[i].param_a = a_minus[i];
		h_I_minus[i].param_b = b_minus[i];
		h_I_minus[i].param_c = c_minus[i];
	}
	stop = std::clock();
	time = ((double)(stop - start)) / CLK_TCK;
	std::cout << time << std::endl;

	thrust::device_vector<line> I_plus = h_I_plus;
	thrust::device_vector<line> I_minus = h_I_minus;

	thrust::transform(I_plus.begin(), I_plus.end(), I_plus.begin(), slope());
	thrust::transform(I_minus.begin(), I_minus.end(), I_minus.begin(), slope());

	boundary left_line;
	boundary right_line;
	boundary test_line;

	left_line.x = -INIT_LINE;
	right_line.x = INIT_LINE;
	test_line.x = 0;

	thrust::device_vector<double> LI_plus_height(plus_count);
	thrust::device_vector<double> LI_minus_height(minus_count);
	thrust::device_vector<double> RI_plus_height(plus_count);
	thrust::device_vector<double> RI_minus_height(minus_count);
	thrust::device_vector<double> TI_plus_height(plus_count);
	thrust::device_vector<double> TI_minus_height(minus_count);

	thrust::transform(I_plus.begin(), I_plus.end(), LI_plus_height.begin(), height(left_line.x));
	thrust::transform(I_minus.begin(), I_minus.end(), LI_minus_height.begin(), height(left_line.x));
	thrust::transform(I_plus.begin(), I_plus.end(), RI_plus_height.begin(), height(right_line.x));
	thrust::transform(I_minus.begin(), I_minus.end(), RI_minus_height.begin(), height(right_line.x));
	thrust::transform(I_plus.begin(), I_plus.end(), TI_plus_height.begin(), height(test_line.x));
	thrust::transform(I_minus.begin(), I_minus.end(), TI_minus_height.begin(), height(test_line.x));
	// 这里可以更快

	

	unsigned int position;
	thrust::device_vector<double>::iterator iter;
	iter = thrust::max_element(LI_plus_height.begin(), LI_plus_height.end());
	position = iter - LI_plus_height.begin();
	left_line.line_plus = I_plus[position],   left_line.y_plus=LI_plus_height[position];
	iter = thrust::min_element(LI_minus_height.begin(), LI_minus_height.end());
	position = iter - LI_minus_height.begin();
	left_line.line_minus = I_minus[position], left_line.y_minus = LI_minus_height[position];		//L
	iter = thrust::max_element(RI_plus_height.begin(), RI_plus_height.end());
	position = iter - RI_plus_height.begin();
	right_line.line_plus = I_plus[position],  right_line.y_plus = RI_plus_height[position];
	iter = thrust::min_element(RI_minus_height.begin(), RI_minus_height.end());
	position = iter - RI_minus_height.begin();
	right_line.line_minus = I_minus[position],right_line.y_minus = RI_minus_height[position];		//R
	iter = thrust::max_element(TI_plus_height.begin(), TI_plus_height.end());
	position = iter - TI_plus_height.begin();
	test_line.line_plus = I_plus[position],   test_line.y_plus = TI_plus_height[position];
	iter = thrust::min_element(TI_minus_height.begin(), TI_minus_height.end());
	position = iter - TI_minus_height.begin();
	test_line.line_minus = I_minus[position], test_line.y_minus = TI_minus_height[position];		//T

	bool if_get_ans = FALSE;
	double res_x, res_y;
	while (!if_get_ans) {
		double plus_slope = test_line.line_plus.slope_value;
		double minus_slope = test_line.line_minus.slope_value;

		if (rm_error(test_line.y_plus) >rm_error(test_line.y_minus)) {
			if (plus_slope > 0) {
				if (minus_slope > plus_slope)
					left_line = test_line;
				else
					right_line = test_line;

			}
			else if (plus_slope < 0) {
				if (minus_slope < plus_slope)
					right_line = test_line;
				else
					left_line = test_line;
			}
			test_line.x = generate_intersection_point(&test_line.line_minus, &test_line.line_plus)->pos_x;
			thrust::transform(I_plus.begin(), I_plus.end(), TI_plus_height.begin(), height(test_line.x));
			thrust::transform(I_minus.begin(), I_minus.end(), TI_minus_height.begin(), height(test_line.x));
			iter = thrust::max_element(TI_plus_height.begin(), TI_plus_height.end());
			position = iter - TI_plus_height.begin();
			test_line.line_plus = I_plus[position], test_line.y_plus = TI_plus_height[position];
			iter = thrust::min_element(TI_minus_height.begin(), TI_minus_height.end());
			position = iter - TI_minus_height.begin();
			test_line.line_minus = I_minus[position], test_line.y_minus = TI_minus_height[position];
		}
		else if(rm_error(test_line.y_plus) <=rm_error(test_line.y_minus)){
			if (plus_slope > 0) {
				double inter_x = generate_intersection_point(&test_line.line_minus, &test_line.line_plus)->pos_x;
				if (minus_slope > plus_slope && inter_x > right_line.x) {
					if (rm_error(test_line.y_plus) == rm_error(test_line.y_minus)) {
						res_x = test_line.x;
						res_y = test_line.y_plus;
						ans->line1 = &test_line.line_plus;
						ans->line2 = &test_line.line_minus;
						if_get_ans = TRUE;
					}
					test_line.x = generate_intersection_point(&test_line.line_minus, &test_line.line_plus)->pos_x;
					thrust::transform(I_plus.begin(), I_plus.end(), TI_plus_height.begin(), height(test_line.x));
					thrust::transform(I_minus.begin(), I_minus.end(), TI_minus_height.begin(), height(test_line.x));
					iter = thrust::max_element(TI_plus_height.begin(), TI_plus_height.end());
					position = iter - TI_plus_height.begin();
					test_line.line_plus = I_plus[position], test_line.y_plus = TI_plus_height[position];
					iter = thrust::min_element(TI_minus_height.begin(), TI_minus_height.end());
					position = iter - TI_minus_height.begin();
					test_line.line_minus = I_minus[position], test_line.y_minus = TI_minus_height[position];
				}
				else {
					right_line = test_line;
					double intersection_x = generate_intersection_point(&right_line.line_plus, &left_line.line_plus)->pos_x;
					if (intersection_x == NULL) {
						res_x = test_line.x;
						res_y = test_line.y_plus;
						ans->line1 = &left_line.line_plus;
						ans->line2 = &right_line.line_plus;
						if_get_ans = TRUE;
					}
					else if(rm_error(intersection_x)== rm_error(test_line.x)) {
						res_x = test_line.x;
						res_y = test_line.y_plus;
						ans->line1 = &left_line.line_plus;
						ans->line2 = &right_line.line_plus;
						if_get_ans = TRUE;
					}
					else {
						test_line.x = intersection_x;
						thrust::transform(I_plus.begin(), I_plus.end(), TI_plus_height.begin(), height(test_line.x));
						thrust::transform(I_minus.begin(), I_minus.end(), TI_minus_height.begin(), height(test_line.x));
						iter = thrust::max_element(TI_plus_height.begin(), TI_plus_height.end());
						position = iter - TI_plus_height.begin();
						test_line.line_plus = I_plus[position], test_line.y_plus = TI_plus_height[position];
						iter = thrust::min_element(TI_minus_height.begin(), TI_minus_height.end());
						position = iter - TI_minus_height.begin();
						test_line.line_minus = I_minus[position], test_line.y_minus = TI_minus_height[position];
					}
				}
			}
			else if (plus_slope < 0) {
				double inter_x = generate_intersection_point(&test_line.line_minus, &test_line.line_plus)->pos_x;
				if (minus_slope < plus_slope &&  inter_x < right_line.x) {
					if (rm_error(test_line.y_plus) == rm_error(test_line.y_minus)) {
						res_x = test_line.x;
						res_y = test_line.y_plus;
						ans->line1 = &test_line.line_plus;
						ans->line2 = &test_line.line_minus;
						if_get_ans = TRUE;
					}
					test_line.x = generate_intersection_point(&test_line.line_minus, &test_line.line_plus)->pos_x;
					thrust::transform(I_plus.begin(), I_plus.end(), TI_plus_height.begin(), height(test_line.x));
					thrust::transform(I_minus.begin(), I_minus.end(), TI_minus_height.begin(), height(test_line.x));
					iter = thrust::max_element(TI_plus_height.begin(), TI_plus_height.end());
					position = iter - TI_plus_height.begin();
					test_line.line_plus = I_plus[position], test_line.y_plus = TI_plus_height[position];
					iter = thrust::min_element(TI_minus_height.begin(), TI_minus_height.end());
					position = iter - TI_minus_height.begin();
					test_line.line_minus = I_minus[position], test_line.y_minus = TI_minus_height[position];
				}
				else {
					left_line = test_line;
					double intersection_x = generate_intersection_point(&right_line.line_plus, &left_line.line_plus)->pos_x;
					if (intersection_x == NULL) {
						res_x = test_line.x;
						res_y = test_line.y_plus;
						ans->line1 = &left_line.line_plus;
						ans->line2 = &right_line.line_plus;
						if_get_ans = TRUE;
					}
					else if (rm_error(intersection_x) == rm_error(test_line.x)) {
						res_x = test_line.x;
						res_y = test_line.y_plus;
						ans->line1 = &left_line.line_plus;
						ans->line2 = &right_line.line_plus;
						if_get_ans = TRUE;
					}
					else {
						test_line.x = intersection_x;
						thrust::transform(I_plus.begin(), I_plus.end(), TI_plus_height.begin(), height(test_line.x));
						thrust::transform(I_minus.begin(), I_minus.end(), TI_minus_height.begin(), height(test_line.x));
						iter = thrust::max_element(TI_plus_height.begin(), TI_plus_height.end());
						position = iter - TI_plus_height.begin();
						test_line.line_plus = I_plus[position], test_line.y_plus = TI_plus_height[position];
						iter = thrust::min_element(TI_minus_height.begin(), TI_minus_height.end());
						position = iter - TI_minus_height.begin();
						test_line.line_minus = I_minus[position], test_line.y_minus = TI_minus_height[position];
					}
				}
			}
		}
	}

	entry:
	double theta = atan2(bo, -ao);
	theta = M_PI / 2 - theta;
	double new_x = cos(theta)*res_x - sin(theta)*res_y;
	double new_y = sin(theta)*res_x + cos(theta)*res_y;
	double a1 = cos(theta)*ans->line1->param_a - sin(theta)*ans->line1->param_b;
	double b1 = sin(theta)*ans->line1->param_a + cos(theta)*ans->line1->param_b;
	double a2 = cos(theta)*ans->line2->param_a - sin(theta)*ans->line2->param_b;
	double b2 = sin(theta)*ans->line2->param_a + cos(theta)*ans->line2->param_b;
	double c1 = ans->line1->param_c;
	double c2 = ans->line2->param_c;
	ans->line1 = generate_line_from_abc(a1, b1, c1);
	ans->line2 = generate_line_from_abc(a2, b2, c2);
	ans->intersection_point = generate_point_from_xy(new_x, new_y);
	ans->answer_b = ans->intersection_point->pos_x*input->obj_function_param_a + ans->intersection_point->pos_y*input->obj_function_param_b;
	//这里还需释放cuda内存

	return ans;
}

int main() {
	// 1. get the input data
	inputs * input = read_from_file("../test_cases/100000_0.dat");
	// 2. get the answer
	answer * ans = compute(input);
	for (int i = 0; i < 10; i++) {
		ans = compute(input);
	}
	// 3. display result and free memory
	char * ans_string = generate_ans_string(ans);
	printf("%s", ans_string);
	getchar();

	free_inputs(&input);
	free_ans(&ans);
	free(ans_string);
	return 0;
}