// compile opencv4 with: sudo cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. -DOPENCV_GENERATE_PKGCONFIG=ON
// for compilation see also: https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html

// compile and run with: g++ OpenCV.cpp `pkg-config --cflags --libs opencv4` ; ./a.out 

// commit and push to git with: git add . ; git commit -m "Next commit3" ; git push origin master


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <experimental/filesystem>

#include "Scene_Generator.hpp"

using namespace cv;
using namespace std;

namespace fs = experimental;


void getFiles ()
{
	string s;

	for (const auto& entry : fs::directory_iterator("/Kanten/")) 
	{
		const auto filenameStr = entry.path().filename().string();

		if (entry.is_directory()) {
		    cout << "dir:  " << filenameStr << '\n';
		}

		else if (entry.is_regular_file()) {
		    cout << "file: " << filenameStr << '\n';
		}

		else
		    cout << "??    " << filenameStr << '\n';
	}
}



