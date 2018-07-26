//
// Created by 唐艺峰 on 2018/7/14.
//

#ifndef INC_2D_LINEAR_PROBLEM_MODELS_H
#define INC_2D_LINEAR_PROBLEM_MODELS_H

#include "floating_number_helper.h"

typedef struct line {
    // ax + by >= c
    double param_a;
    double param_b;
    double param_c;
    double slope_value;
} line;

typedef struct point {
    // (x, y)
    double pos_x;
    double pos_y;
} point;

// the functions below may not be all used

line * generate_line_from_abc(double param_a, double param_b, double param_c); // ax + by = c
line * generate_line_from_kb(double k, double b); // y = kx + b
line * generate_line_from_2points(point * p1, point * p2); //

point * generate_point_from_xy(double pos_x, double pos_y);
point * generate_intersection_point(line * line1, line * line2);

double compute_slope(line * line);
int is_parallel(line * line1, line * line2);

#endif //INC_2D_LINEAR_PROBLEM_MODELS_H
