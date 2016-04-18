#ifndef FACEDETECT_DLL_H
#define FACEDETECT_DLL_H

#ifdef FACEDETECTDLL_EXPORTS
#define FACEDETECTDLL_API __declspec(dllexport) 
#else
#define FACEDETECTDLL_API __declspec(dllimport) 
#endif

FACEDETECTDLL_API int * facedetect_frontal(unsigned char * gray_image_data, int width, int height, int step, //input image, it must be gray (single-channel) image!
	float scale, //scale factor for scan windows
	int min_neighbors, //how many neighbors each candidate rectangle should have to retain it
	int min_size, //Minimum possible face size. Faces smaller than that are ignored.
	int max_size = 0); //Maximum possible face size. Faces larger than that are ignored. It is the largest posible when max_size=0.

FACEDETECTDLL_API int * facedetect_multiview(unsigned char * gray_image_data, int width, int height, int step, //input image, it must be gray (single-channel) image!
	float scale, //scale factor for scan windows
	int min_neighbors, //how many neighbors each candidate rectangle should have to retain it
	int min_size, //Minimum possible face size. Faces smaller than that are ignored.
	int max_size = 0); //Maximum possible face size. Faces larger than that are ignored. It is the largest posible when max_size=0.

FACEDETECTDLL_API int * facedetect_multiview_reinforce(unsigned char * gray_image_data, int width, int height, int step, //input image, it must be gray (single-channel) image!
	float scale, //scale factor for scan windows
	int min_neighbors, //how many neighbors each candidate rectangle should have to retain it
	int min_size, //Minimum possible face size. Faces smaller than that are ignored.
	int max_size = 0); //Maximum possible face size. Faces larger than that are ignored. It is the largest posible when max_size=0.
#endif