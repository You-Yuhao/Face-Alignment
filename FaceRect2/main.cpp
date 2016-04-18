#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "facedetect-dll.h"
#include "face_x.h"

#pragma comment(lib, "libfacedetect.lib")

using namespace cv;
using namespace std;


const string ModelFileName = "model_small.xml.gz";
const string FacePath      = "face/";
const string leftEyePath   = "leftEye/";
const string rightEyePath  = "rightEye/";
const string EyesPath      = "eyes/";
const string leftBrowPath  = "leftBrow/";
const string rightBrowPath = "rightBrow/";
const string BrowsPath     = "brows/";
const string NosePath      = "nose/";
const string MouthPath     = "mouth/";
const int    ImageSize     = 200;


string TestImage;
//int testImageNum = 0;
//int finalImageNum = 0;

/*---- DoG函数 -------
带通滤波器，去除高频噪声和低频噪声
------------------------------------------------------*/
Mat DoG(Mat img, double sigma1 = 2.0, double sigma2 = 4.0)
{
	Mat img1, img2, img3;
	GaussianBlur(img, img1, Size(5, 5), sigma1);
	GaussianBlur(img, img2, Size(5, 5), sigma2);
	img3 = img1 - img2;
	normalize(img3, img3, 255, 0, CV_MINMAX);
	return img3;
}

/*---- rotateFace函数 -------
以左眼的中心点作为旋转的输入原点，getRotationMatrix2D()
生成了表示仿射变换的2 * 3矩阵
------------------------------------------------------*/
void rotateFace(Mat& src, Point2d& pt, double angle, Mat& dst)
{
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
}


/*---- cropFaceBasedOnEye函数 -------
人脸几何归一化
------------------------------------------------------*/
Mat cropFaceBasedOnEye(Mat img, Point2d leftEye, Point2d rightEye, float offset, int outputWidth, int outputHeight)
{
	int offset_h = floor(offset * outputWidth);
	int offset_v = floor(offset * outputHeight);

	int eyegap_h = rightEye.x - leftEye.x;
	int eyegap_v = rightEye.y - leftEye.y;

	float eye_distance = sqrt(pow(eyegap_h, 2) + pow(eyegap_v, 2));  // 实际瞳距
	float eye_reference = outputWidth - 2 * offset_h;                // 归一化后的瞳距
	float scale = eye_distance / eye_reference;                      // 缩放比例

	// 根据眼睛坐标旋转图片
	Mat rotatedImage;
	if (eyegap_v != 0)
	{
		double rotation = atan2f((float)eyegap_v, (float)eyegap_h);  // actan求倾斜角度
		double degree = rotation * 180 / CV_PI;
		rotateFace(img, leftEye, degree, rotatedImage);
	}

	// 切割出人脸
	Point2d crop_xy(leftEye.x - scale*offset_h, leftEye.y - scale*offset_v);
	Size crop_size(outputWidth*scale, outputHeight*scale);
	Rect crop_area(crop_xy, crop_size);
	Mat cropFace;
	if (eyegap_v == 0)
		cropFace = img(crop_area);
	else
		cropFace = rotatedImage(crop_area);

	// 归一化图片大小
	resize(cropFace, cropFace, Size(outputWidth, outputHeight));
	return cropFace;
	/*Mat croppedGray;
	cvtColor(cropFace, croppedGray, CV_BGR2GRAY);
	equalizeHist(croppedGray, croppedGray);
	return croppedGray;*/
}


/*---- circleCutFace函数 -------
在原图上标记出关键点，得到人脸图中关键点坐标，切割人脸图片
------------------------------------------------------*/
void circleCutFace(Mat img_grayzone, Mat img, Rect face, const FaceX & face_x, int mode)
{
	int count = 0;
	int squreSize;
	Point2d leftEye, rightEye, Nose, leftMouth, rightMouth;
	// 标记点
	vector<Point2d> landmarks = face_x.Alignment(img_grayzone, face);
	for (Point2d landmark : landmarks)
	{
		//circle(img, landmark, 1, Scalar(0, 255, 0), 2);

		switch (count++){
		case 0:
			leftEye.x = (int) landmark.x;
			leftEye.y = (int) landmark.y;
			//cout << "leftEye: (" << leftEye.x << ", " << leftEye.y << ")" << endl;
			break;
		case 1:
			rightEye.x = (int) landmark.x;
			rightEye.y = (int) landmark.y;
			//cout << "rightEye: (" << rightEye.x << ", " << rightEye.y << ")" << endl;
			break;
		default:
			break;
		}

	}

	if (mode == 1 || mode == 2){
		squreSize = 200;
	}
	else if (mode == 3){
		squreSize = 100;
	}
	Mat img_size = cropFaceBasedOnEye(img, leftEye, rightEye, 0.3, squreSize, squreSize);
	imwrite(FacePath + TestImage, img_size);

	Mat img_sizeGray;
	Rect face1;
	face1.x = 0;
	face1.y = 0;
	face1.width = squreSize;
	face1.height = squreSize;
	cvtColor(img_size, img_sizeGray, COLOR_BGR2GRAY);
	vector<Point2d> landmarks1 = face_x.Alignment(img_sizeGray, face1);
	int count1 = 0;
	for (Point2d landmark : landmarks1)
	{
		//circle(img, landmark, 1, Scalar(0, 255, 0), 2);
		
		// 将原图中标记点的坐标改成200*200下的点坐标
		switch (count1++){
		case 0:
			leftEye.x = (int) landmark.x;
			leftEye.y = (int) landmark.y;
			cout << "leftEye: (" << leftEye.x << ", " << leftEye.y << ")" << endl;
			break;
		case 1:
			rightEye.x = (int) landmark.x;
			rightEye.y = (int) landmark.y;
			cout << "rightEye: (" << rightEye.x << ", " << rightEye.y << ")" << endl;
			break;
		case 2:
			Nose.x = (int) landmark.x;
			Nose.y = (int) landmark.y;
			//cout << "Nose: (" << Nose.x << ", " << Nose.y << ")" << endl;
			break;
		case 3:
			leftMouth.x = (int) landmark.x;
			leftMouth.y = (int) landmark.y;
			//cout << "leftMouth: (" << leftMouth.x << ", " << leftMouth.y << ")" << endl;
			break;
		case 4:
			rightMouth.x = (int) landmark.x;
			rightMouth.y = (int) landmark.y;
			//cout << "rightMouth: (" << rightMouth.x << ", " << rightMouth.y << ")" << endl;
			break;
		default:
			break;
		}

	}


	if (mode == 1){
		Mat img_lefteye = img_size(Rect((leftEye.x - 30), (leftEye.y - 16), 60, 40));
		imwrite(leftEyePath + TestImage, img_lefteye);

		Mat img_righteye = img_size(Rect((rightEye.x - 30), (rightEye.y - 16), 60, 40));
		imwrite(rightEyePath + TestImage, img_righteye);

		Mat img_eyes = img_size(Rect((leftEye.x - 30), (leftEye.y - 16), 140, 40));
		imwrite(EyesPath + TestImage, img_eyes);

		Mat img_leftbrow = img_size(Rect((leftEye.x - 30), (leftEye.y - 40), 50, 30));
		imwrite(leftBrowPath + TestImage, img_leftbrow);

		Mat img_rightbrow = img_size(Rect((rightEye.x - 30), (rightEye.y - 40), 50, 30));
		imwrite(rightBrowPath + TestImage, img_rightbrow);

		Mat img_brows = img_size(Rect((leftEye.x - 30), (leftEye.y - 40), 140, 30));
		imwrite(BrowsPath + TestImage, img_brows);

		Mat img_nose = img_size(Rect((leftEye.x + 15), (leftEye.y + 2), 50, 70));
		imwrite(NosePath + TestImage, img_nose);

		Mat img_mouth = img_size(Rect((leftMouth.x - 10), (leftMouth.y - 20), 80, 40));
		imwrite(MouthPath + TestImage, img_mouth);
	}
	else if (mode == 2){
		Mat img_lefteye = img_size(Rect((leftEye.x - 32), (leftEye.y - 20), 64, 44));
		imwrite(leftEyePath + TestImage, img_lefteye);

		Mat img_righteye = img_size(Rect((rightEye.x - 32), (rightEye.y - 20), 64, 44));
		imwrite(rightEyePath + TestImage, img_righteye);

		Mat img_eyes = img_size(Rect((leftEye.x - 32), (leftEye.y - 20), 144, 44));
		imwrite(EyesPath + TestImage, img_eyes);

		Mat img_leftbrow = img_size(Rect((leftEye.x - 32), (leftEye.y - 44), 54, 34));
		imwrite(leftBrowPath + TestImage, img_leftbrow);

		Mat img_rightbrow = img_size(Rect((rightEye.x - 32), (rightEye.y - 44), 54, 34));
		imwrite(rightBrowPath + TestImage, img_rightbrow);

		Mat img_brows = img_size(Rect((leftEye.x - 32), (leftEye.y - 44), 144, 34));
		imwrite(BrowsPath + TestImage, img_brows);

		Mat img_nose = img_size(Rect((leftEye.x + 13), (leftEye.y), 54, 74));
		imwrite(NosePath + TestImage, img_nose);

		Mat img_mouth = img_size(Rect((leftMouth.x - 12), (leftMouth.y - 20), 84, 44));
		imwrite(MouthPath + TestImage, img_mouth);
	}
	else if (mode == 3){
		Mat img_lefteye = img_size(Rect((leftEye.x - 16), (leftEye.y - 8), 34, 24));
		imwrite(leftEyePath + TestImage, img_lefteye);

		Mat img_righteye = img_size(Rect((rightEye.x - 16), (rightEye.y - 8), 34, 24));
		imwrite(rightEyePath + TestImage, img_righteye);

		Mat img_eyes = img_size(Rect((leftEye.x - 16), (leftEye.y - 8), 74, 24));
		imwrite(EyesPath + TestImage, img_eyes);

		Mat img_leftbrow = img_size(Rect((leftEye.x - 16), (leftEye.y - 23), 29, 19));
		imwrite(leftBrowPath + TestImage, img_leftbrow);

		Mat img_rightbrow = img_size(Rect((rightEye.x - 16), (rightEye.y - 23), 29, 19));
		imwrite(rightBrowPath + TestImage, img_rightbrow);

		Mat img_brows = img_size(Rect((leftEye.x - 16), (leftEye.y - 23), 74, 19));
		imwrite(BrowsPath + TestImage, img_brows);

		Mat img_nose = img_size(Rect((leftEye.x + 7), (leftEye.y), 29, 39));
		imwrite(NosePath + TestImage, img_nose);

		Mat img_mouth = img_size(Rect((leftMouth.x - 6), (leftMouth.y - 9), 44, 24));
		imwrite(MouthPath + TestImage, img_mouth);
	}

}


/*---- detectSaveFace函数 -------
检测灰度图片中的人脸，返回人脸矩形坐标(x,y,width,height)，
并保存人脸图片，调用circleCutFace
------------------------------------------------------*/
Rect detectSaveFace(Mat img_gray, Mat img, const FaceX & face_x, int mode)
{
	int * pResults = NULL;
	Rect face;
	
	pResults = facedetect_frontal((unsigned char*)(img_gray.ptr(0)), img_gray.cols, img_gray.rows, img_gray.step, 1.2f, 3, 24);
	//printf("%d frontal faces detected.\n", (pResults ? *pResults : 0));

	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 6 * i;
		face.x = p[0];
		face.y = p[1];
		face.width = p[2];
		face.height = p[3];
	}
	//printf("face_rect = [%d, %d, %d, %d]\n", face.x, face.y, face.width, face.height);  // 测试用，调整图片阈值


	if (!*pResults){
		// cout << "第" << testImageNum << "张人脸未检测到!!!!!!!!!!!!!!!!" << endl;
		Rect noFace;
		noFace.x = 0;
		noFace.y = 0;
		noFace.width = 0;
		noFace.height = 0;
		return noFace;
	}	
	else{
			//printf("face_rect = [%d, %d, %d, %d]\n", face.x, face.y, face.width, face.height);
		    // cout << "第" << testImageNum << "张人脸检测成功！" << endl;
			circleCutFace(img_gray, img, face, face_x, mode);
			//finalImageNum++;
		}

	//// 专门针对有些身份证检测不到人脸打的补丁
	//if (face.width < ImageSize * 0.75 || face.height < ImageSize * 0.75){
	//	Mat img_smallgray = img_gray(Rect(1300, 200, 640, 840));  // Rect(1300, 200, 640, 840)是针对身份证图片的特定值
	//	Mat img_small = img(Rect(1300, 200, 640, 840));
	//	pResults = facedetect_frontal((unsigned char*)(img_smallgray.ptr(0)), img_smallgray.cols, img_smallgray.rows, img_smallgray.step, 1.2f, 3, 24);

	//	//printf("%d frontal faces detected.\n", (pResults ? *pResults : 0));
	//	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	//	{
	//		short * p = ((short*)(pResults + 1)) + 6 * i;
	//		face.x = p[0];
	//		face.y = p[1];
	//		face.width = p[2];
	//		face.height = p[3];
	//	}
	//	
	//	if (*pResults){
	//		printf("face_rect = [%d, %d, %d, %d]\n", face.x, face.y, face.width, face.height);
	//		circleCutFace(img_smallgray, img_small, face, face_x);
	//	}	
	//	// 这个else还有要注意的地方：这是针对身份证图片，所以必然有人脸，那么在检测不到人脸时强行给个区域是合理的
	//	// 但是，也要考虑人脸确实不存在的情况。（以后添加一个是否是“身份证”的判断？）
	//	else{
	//		face.x = 100;
	//		face.y = 250;
	//		face.width = 420;
	//		face.height = 420;
	//		printf("face_rect = [%d, %d, %d, %d]\n", face.x, face.y, face.width, face.height);
	//		//imwrite("E:/faces/222.jpg", img_smallgray(face));  // 测试用
	//		circleCutFace(img_smallgray, img_small, face, face_x);
	//	}
	//}
	//else{
	//	printf("face_rect = [%d, %d, %d, %d]\n", face.x, face.y, face.width, face.height);
	//	circleCutFace(img_gray, img, face, face_x);
	//}

	return face;
}





int main()
{
	int picNumber = 0;
	int mode = 1;
	cout << "1.200*200，小块大小不变（默认）\n2.200*200，小块各边加4\n3.100*100，小块各边加4\n请选择：";
	cin >> mode;
	cout << "\n请输入需要处理的图片数目：";
	cin >> picNumber;
	cout << endl;

	char s[100];
	for (int testImageNum = 0; testImageNum < picNumber; ++testImageNum){
		sprintf_s(s, "%d.jpg", testImageNum);
		TestImage = s;
		
		//cout << TestImage << endl;
		//读取图像，转化为灰度图，再进行直方图均衡化
		Mat img = imread(TestImage);
		if (img.empty())
		{
			fprintf(stderr, "Can not load the image file.\n\n");
			continue;
		}
		Mat img_gray;
		cvtColor(img, img_gray, COLOR_BGR2GRAY);
		equalizeHist(img_gray, img_gray);

		FaceX face_x(ModelFileName);
		Rect face = detectSaveFace(img_gray, img, face_x, mode);

		if (face.x == 0){
			cout << "第" << testImageNum << "张人脸未检测到!!!!!!!!!!!!!!!!" << endl;
		}
		else{
			cout << "第" << testImageNum << "张人脸检测成功！" << endl;
		}
		cout << endl;
	}
	

	//cout << "有" << testImageNum - finalImageNum << "张人脸未检测到" << endl << endl;
	system("pause"); 
	return 0;
}


