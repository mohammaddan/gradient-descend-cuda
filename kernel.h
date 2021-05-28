#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
const double epsilon = 0.0001;

void tink4(double* theta, double* train_x, double* train_y, int n, int p,int blocksize);