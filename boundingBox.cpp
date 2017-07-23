#include "opencv2/opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;
vector<Vec4f> boundingBoxInf;//用于存储一帧中筛选出来的boundingBox信息，Vec4f的第一、二个元素为包围盒左上角坐标，第二、三个元素代表包围盒的长和宽
//vector<Vec4f> preciseboundingBoxInf;//更精确地用于存储一帧中筛选出来的boundingBox信息，Vec4f的第一、二个元素为包围盒左上角坐标，第二、三个元素代表包围盒的长和宽
void drawBoundingBox(Mat src)
{
	boundingBoxInf.clear();//清空vector数组。
	//定义一些参数
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;


	// 找出轮廓
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	// 多边形逼近轮廓 + 获取矩形和圆形边界框
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	//vector<Point2f>center(contours.size());
	//vector<float>radius(contours.size());

	//一个循环，遍历所有部分，进行本程序最核心的操作
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);//用指定精度逼近多边形曲线 
		boundRect[i] = boundingRect(Mat(contours_poly[i]));//计算点集的最外面（up-right）矩形边界
	}

	// 绘制多边形轮廓 + 包围的矩形框 + 圆形框
	double area = (double)(src.cols * src.rows);
	double minArea = area / 500;
	double maxArea = area / 50;
	for (int unsigned i = 0; i<contours.size(); i++)
	{
		double boundingWidth = (double)(double)(boundRect[i].br().x - boundRect[i].tl().x);
		double boundingHeight = (double)(boundRect[i].br().y - boundRect[i].tl().y);
		double boundingboxArea = boundingWidth * boundingHeight;
		double boxRatio = boundingHeight / boundingWidth;
		if (boundingboxArea >  minArea && boundingboxArea <  maxArea && boxRatio > 1.5 && boxRatio < 4)//去除置信度不高的boundingBox
		{
			/*Vec4f bB0(boundRect[i].tl().x, boundRect[i].tl().y, boundingWidth, boundingHeight);
			preciseboundingBoxInf.push_back(bB0);*/
			int temp_x = boundRect[i].tl().x, temp_y = boundRect[i].tl().y;//temp_x代表包围盒左上角x坐标，temp_y代表包围盒左上角y坐标
			if (temp_x - boundingWidth / 2 > 0)
				temp_x = temp_x - boundingWidth / 2;
			if (temp_x + boundingWidth * 2 < src.cols)//扩展后包围盒右边界未越界，注意Mat对象的原点（0，0）位于图像左上角！
				boundingWidth = (int)(boundingWidth * 2);
			if (temp_y - boundingHeight / 2 > 0)
				temp_y = temp_y - boundingHeight / 2;
			if (temp_y + boundingHeight * 2 < src.rows)//扩展后包围盒下边界未越界
				boundingHeight = (int)(boundingHeight * 2);
			Vec4f bB(temp_x, temp_y, boundingWidth, boundingHeight);//将扩展后的包围盒信息存入bB中
			boundingBoxInf.push_back(bB);
		}
	}
	//for (int i = 0; i < boundingBoxInf.size(); i++)
	//{
	//	cout << "x = " << boundingBoxInf[i][0] << ",y = " << boundingBoxInf[i][1] << ",length = " << boundingBoxInf[i][2] << ",wide = " << boundingBoxInf[i][3] << endl;
	//}

}