#include "cute.h"
#include "flatField_test.h"
#include <stdio.h>
#include <tiffio.h>
#include <assert.h>
#include <iostream>


#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

//typedef unsigned short GreyscaleValue; //unsigned char for 8-bit and unsigned short for 16-bit tiff
//typedef double Real;

#include <flatFieldCorrect_cpu.h>
#include <flatFieldCorrect_gpu.h>

void flatFieldCorrect_cpu_test_16bit() {

	const char* lightFile = "./test-data/16-bit/MED_light.tif";
	const char* darkFile = "./test-data/16-bit/MED_dark.tif";
	const char* fileToCorrect = "./test-data/16-bit/lamb1_";
	int nImages = 1;

	float milli =0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	FieldImages fi(lightFile, darkFile);
	flatFieldCorrect_cpu<unsigned short, double>(fi, fileToCorrect,nImages);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);

	printf("testing execution of sequential flat field correction (16-bit): %.1f (ms)", milli);
	ASSERTM("execution failed", true);
}


void flatFieldCorrect_gpu_test_16bit() {

	const char* lightFile = "./test-data/16-bit/MED_light.tif";
	const char* darkFile = "./test-data/16-bit/MED_dark.tif";
	const char* fileToCorrect = "./test-data/16-bit/lamb1_";
	int nImages = 1;

	float milli =0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	FieldImages fi(lightFile, darkFile);
	flatFieldCorrect_gpu<unsigned short, double>(fi, fileToCorrect,nImages);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);

	printf("testing execution of parallel flat field correction (16-bit): %.1f (ms)", milli);
	ASSERTM("execution failed", true);
}

void flatFieldCorrect_cpu_test_8bit() {

	const char* lightFile = "./test-data/8-bit/light-median-gimp.tif";
	const char* darkFile = "./test-data/8-bit/dark-median-gimp.tif";
	const char* fileToCorrect = "./test-data/8-bit/sloth1_";
	int nImages = 1;

	float milli =0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	FieldImages fi(lightFile, darkFile);
	flatFieldCorrect_cpu<unsigned char, double>(fi, fileToCorrect, nImages);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);

	printf("testing execution of sequential flat field correction (8-bit): %.1f (ms)", milli);
	ASSERTM("execution failed", true);
}

void flatFieldCorrect_gpu_test_8bit() {

	const char* lightFile = "./test-data/8-bit/light-median-gimp.tif";
	const char* darkFile = "./test-data/8-bit/dark-median-gimp.tif";
	const char* fileToCorrect = "./test-data/8-bit/sloth1_";
	int nImages = 1;

	float milli =0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	FieldImages fi(lightFile, darkFile);
	flatFieldCorrect_gpu<unsigned char, double>(fi, fileToCorrect,nImages);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milli, start, stop);

	printf("testing execution of parallel flat field correction (8-bit): %.1f (ms)", milli);
	ASSERTM("execution failed", true);
}

cute::suite make_suite_flatFieldCorrection(){
	cute::suite s;
	s.push_back(CUTE(flatFieldCorrect_cpu_test_16bit));
	s.push_back(CUTE(flatFieldCorrect_gpu_test_16bit));
	s.push_back(CUTE(flatFieldCorrect_cpu_test_8bit));
	s.push_back(CUTE(flatFieldCorrect_gpu_test_8bit));
	return s;
}



