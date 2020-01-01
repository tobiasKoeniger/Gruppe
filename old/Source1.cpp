#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using std::cout;
using namespace std;

int main()
{
	/*
	// Read image as grayscale image
	// C:\\Users\\Tobias\\OneDrive\\AT Projekt\\Project1
	Mat im1 = imread("1.png", IMREAD_GRAYSCALE);
	Mat im2 = imread("2.png", IMREAD_GRAYSCALE);



	// Threshold image
	threshold(im1, im1, 128, 255, THRESH_BINARY);
	threshold(im2, im2, 128, 255, THRESH_BINARY);

	// Calculate Moments
	Moments moments1 = cv::moments(im1, false);
	Moments moments2 = cv::moments(im2, false);

	// Calculate Hu Moments
	double huMoments1[7];
	HuMoments(moments1, huMoments1);

	double huMoments2[7];
	HuMoments(moments2, huMoments2);

	// Log scale hu moments
	for (int i = 0; i < 7; i++)
	{
		huMoments1[i] = -1 * copysign(1.0, huMoments1[i]) * log10(abs(huMoments1[i]));
		cout << huMoments1[i] << "\n";


		huMoments2[i] = -1 * copysign(1.0, huMoments2[i]) * log10(abs(huMoments2[i]));
		cout << huMoments2[i] << "\n";


	}

	double d1 = matchShapes(im1, im2, CONTOURS_MATCH_I1, 0);
	double d2 = matchShapes(im1, im2, CONTOURS_MATCH_I2, 0);
	double d3 = matchShapes(im1, im2, CONTOURS_MATCH_I3, 0);

	cout << d1 << "\n";
	cout << d2 << "\n";
	cout << d3 << "\n";

	return 0;
	*/
}