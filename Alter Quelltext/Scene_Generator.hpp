// compile opencv4 with: sudo cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. -DOPENCV_GENERATE_PKGCONFIG=ON
// for compilation see also: https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html

// compile and run with: g++ OpenCV.cpp `pkg-config --cflags --libs opencv4` ; ./a.out 

// commit and push to git with: git add . ; git commit -m "Next commit3" ; git push origin master


#include <string>

using namespace std;


class Scene_Generator
{
	private:
	string files;

	public:
	void getFiles ();
};

