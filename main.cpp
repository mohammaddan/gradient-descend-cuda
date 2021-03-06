#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <iomanip>
#include "public-lib.h"
#include "kernel.h"

using namespace std;

const int n = 1000;
const int p = 100;
double params[100];// = { 100, 2, 4, 0.5, 1, 4, 3, 2, 1, 2,-2,5,7,14,23,37 };
int attrbite_range[100];// = { 10, 100, 50000, 5, 10, 3, 50, 20, 50, 100,10,10,10,20,30,50 };
double norm1[p];
double train_x[n][p] = { 0 }, train_y[n] = { 0 };
double theta[p];

void set_train_set();
void tink3(double* theta, int thread_num);
double cal_y(double* x);
double predict_y(double* x);

int main(int argc, const char** argv) {
	for(int i=0;i<p;i++){
		params[i]=(rand()%10)*10+1;
		attrbite_range[i]=(rand()%10+1)*10;
	}
	print_vector("original theta",params,p,7,2);
	vector<vector<double>> times;
	vector<string> functions;
	vector<double> temp;
	srand(time(NULL));

	setcolor(1);
	set_train_set();

	system_clock::time_point start, end;

	// functions.push_back("Parallel third implement (tink3 function)");
	// temp.clear();
	// for (int i = 1; i <= 8; i++) {
	// 	start = system_clock::now();
	// 	tink3(theta, i);
	// 	end = system_clock::now();
	// 	temp.push_back(diffclock(end, start) / 1000.0);
	// }
	// times.push_back(temp);
	// setcolor(1);
	// cout << functions.back() << endl;
	// print_vector("theta", theta, p, 2, 0);
	// cout << endl;

	functions.push_back("Parallel forth implement (tink4 function)");
	temp.clear();
	for(int i=1;i<=32;i<<=1){
		start = system_clock::now();
		tink4(theta,(double*)train_x,train_y,n,p,i);
		end = system_clock::now();
		temp.push_back(diffclock(end, start) / 1000.0);
	}
	times.push_back(temp);
	setcolor(1);
	cout << functions.back() << endl;
	print_vector("theta", theta, p, 2, 0);
	cout << endl;
	setcolor(1);
	cout << functions.back() << endl;
	print_vector("theta", theta, p, 2, 0);
	cout << endl;

	cout << setw(45) << left << "Implements/Number block size";
	for (int i = 1; i <= 32; i<<=1) cout << setw(7) << right << i;
	cout << endl;
	for (int i = 0; i < functions.size(); i++) {
		cout << setw(45) << left << functions[i];
		for (auto t : times[i]) {
			cout << setw(7) << right << t;
		}
		cout << endl;
	}
	// double x[p];
	// x[0] = 1;
	// for (int i = 1; i < p; i++)
	//   x[i] = attrbite_range[i] / (1 + rand() % 5);
	// print_vector("x", x, p, 7, 4);
	// double cy = cal_y(x), py = predict_y(x);
	// setcolor(2, 0);
	// cout << "  y=" << cy << "  predict_y=" << py;
	// setcolor(1);
	// cout << " diff=" << fabs(cy - py) << "\n\n";
	setcolor(7);
	return 0;
}

double cal_y(double* x) {
	double result = theta[0];
	for (int i = 1; i < p; i++)
		result += x[i] * theta[i];
	return result;
}

double predict_y(double* x) {
	double result = params[0];
	for (int i = 1; i < p; i++)
		result += x[i] * params[i];
	return result;
}

void set_train_set() {
	double t = 0;
	for (int i = 0; i < n; i++) {
		train_x[i][0] = 1;
		train_y[i] = train_x[i][0] * params[0];
		for (int j = 1; j < p; j++) {
			double noise = (rand() % 100 - 50) / 10000.0;
			train_x[i][j] = rand() % attrbite_range[j];
			train_x[i][j] /= attrbite_range[j];
			train_y[i] += params[j] * train_x[i][j] + noise;
		}
	}
}

void tink3(double* theta, int thread_num) {
	double alpha = 0.01;
	double err_0[n];

	for (int i = 0; i < p; i++)
		theta[i] = (rand() % 10) / 1000.0;

	double delta[p] = { 0 };
	double y_pred[n] = { 0 };
	int cnt = 2000000, it = 0;
	double last_error, error = 100;
	do {
		last_error = error;
		int j;
		double error2 = 0;

#pragma omp parallel for private(j) num_threads(thread_num) schedule(static,8) reduction(+:error2)
		for (int i = 0; i < n; i++) {
			y_pred[i] = 0;
			for (j = 0; j < p; j++) {
				y_pred[i] += theta[j] * train_x[i][j];
			}
			err_0[i] = y_pred[i] - train_y[i];
			error2 += err_0[i] * err_0[i];
		}

#pragma omp parallel for num_threads(thread_num) schedule(static,1)
		for (int j = 0; j < p; j++) {
			double temp = 0;
			for (int i = 0; i < n; i++)
				temp += train_x[i][j] * err_0[i];
			theta[j] -= alpha * temp / n;
		}

		error = sqrt(error2);
		if (error > last_error)
			alpha /= 1.2;

	} while (it++ < cnt && fabs(error - last_error) > epsilon);
	printf("in %d iterations \n", it);
}