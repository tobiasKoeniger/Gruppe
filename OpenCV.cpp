
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;


int main()
{

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

	cout << "\n\n";

	// Log scale hu moments
	for (int i = 0; i < 7; i++)
	{
		huMoments1[i] = -1 * copysign(1.0, huMoments1[i]) * log10(abs(huMoments1[i]));
		cout << "Hu Moment " << i + 1 << ", Bild 1: " << huMoments1[i] << "\n";
		
		huMoments2[i] = -1 * copysign(1.0, huMoments2[i]) * log10(abs(huMoments2[i]));
		cout << "Hu Moment " << i + 1 << ", Bild 2: " << huMoments2[i] << "\n\n";

	}

	// 
	double d1 = matchShapes(im1, im2, CONTOURS_MATCH_I1, 0);
	double d2 = matchShapes(im1, im2, CONTOURS_MATCH_I2, 0);
	double d3 = matchShapes(im1, im2, CONTOURS_MATCH_I3, 0);

	//
	cout << "\n\n" << "matchShapes, CONTOUR_MATCH_I1: " << d1 << "\n";
	cout << "matchShapes, CONTOUR_MATCH_I2: " << d2 << "\n";
	cout << "matchShapes, CONTOUR_MATCH_I3: " << d3 << "\n";

	//
	imshow("Image 1", im1);
	imshow("Image 2", im2);

	waitKey(0);

	//
	return 0;
}
