#include "opencv2/opencv.hpp"
using namespace cv;
Mat denoise(Mat src)
{
	/*Mat denoiseFigure, dstFigure, denoiseFigure1, out;
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
	medianBlur(src, denoiseFigure, 7);
	denoiseFigure1 = denoiseFigure.clone();
	GaussianBlur(denoiseFigure, denoiseFigure1, Size(3, 1), 0.0);
	erode(denoiseFigure1, dstFigure, element);
	dilate(dstFigure, out, element);
	return out;*/
	Mat out;
	Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
	medianBlur(src, out, 7);
	GaussianBlur(out, out, Size(3, 1), 0.0);
	erode(out, out, element);
	dilate(out, out, element);
	return out;
}