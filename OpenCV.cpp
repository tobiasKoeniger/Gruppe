// compile opencv4 with: sudo cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. -DOPENCV_GENERATE_PKGCONFIG=ON
// for compilation see also: https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html

// compile and run with: g++ OpenCV.cpp `pkg-config --cflags --libs opencv4` & ./a.out 

// commit and push to git with: git add . ; git commit -m "Next commit3" ; git push origin master


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;


// Main Function
int main()
{

	// Read image as grayscale image
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

	// MatchShapes berechnen 
	double d1 = matchShapes(im1, im2, CONTOURS_MATCH_I1, 0);
	double d2 = matchShapes(im1, im2, CONTOURS_MATCH_I2, 0);
	double d3 = matchShapes(im1, im2, CONTOURS_MATCH_I3, 0);

	//
	cout << "\n" << "matchShapes, CONTOUR_MATCH_I1: " << d1 << "\n";
	cout << "matchShapes, CONTOUR_MATCH_I2: " << d2 << "\n";
	cout << "matchShapes, CONTOUR_MATCH_I3: " << d3 << "\n";

	//
	imshow("image 1", im1);
	imshow("Image 2", im2);

	waitKey(0);

	// Zeige OpenCV Version 
	cout << "\n" << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "\n\n";

	// Programm Ende
	return 0;
}
