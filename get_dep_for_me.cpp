  #include <iostream> //used for testing
#include "ros/ros.h"
#include <image_transport/image_transport.h>   //image_transport
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>    //图像编码格式
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "std_msgs/String.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "../include/color.hpp"//颜色提取
#include "../include/EDLib.h"
#include "../include/rec_o_l.hpp"
#include "../include/rec_line_new.hpp"
#include <zxing/LuminanceSource.h>
#include <zxing/common/Counted.h>
#include <zxing/Reader.h>
#include <zxing/ReaderException.h>
#include <zxing/Exception.h>
#include <zxing/aztec/AztecReader.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/common/IllegalArgumentException.h>
#include <zxing/DecodeHints.h>
#include <zxing/BinaryBitmap.h>
#include <zxing/DecodeHints.h>
#include <zxing/datamatrix/DataMatrixReader.h>
#include <zxing/MultiFormatReader.h>
#include <zxing/pdf417/PDF417Reader.h>
#include <zxing/qrcode/QRCodeReader.h>
#include <zxing/MatSource.h>
#include <zxing/oned/Code128Reader.h>


using namespace cv;
using namespace std;
using namespace message_filters;
/*双目
cv::Mat left_image;
cv::Mat right_image;

void image_callback(const sensor_msgs::ImageConstPtr &left_img, const sensor_msgs::ImageConstPtr &right_img)
{
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(left_img, sensor_msgs::image_encodings::TYPE_8UC1);
    left_image = cv_ptr1->image;
    cv_bridge::CvImagePtr cv_ptr2 = cv_bridge::toCvCopy(right_img, sensor_msgs::image_encodings::TYPE_8UC1);
    right_image = cv_ptr2->image;
}
*/
//深度�???
//深度图和彩图获取
cv::Mat image_dep;
cv::Mat img;
cv::Mat img_p;
//圆两边的深度
int ll = 0;
double dep1;
double dep2;
double dep_gan;
double dep1_min = 4000; 
double dep2_min = 4000;
double dep_gan_min = 4000;
double x_p;
double y_p;
double x_m;
double y_m;
double angle1;
int x_middle,y_middle;
//两个话题同步获取


//识别椭圆
// void image_callback(const sensor_msgs::ImageConstPtr &dep_image,const sensor_msgs::ImageConstPtr &color_image)
// {
//     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(dep_image, "16UC1");
//     image_dep = cv_ptr->image;
//     cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(color_image, "bgr8");
//     img = cv_ptr1->image;
//     Mat imghsv;
// 	Mat testImg, cdst;
// 	cv::cvtColor(img, imghsv, cv::COLOR_BGR2HSV);
// 	cv::inRange(imghsv, cv::Scalar(35, 43, 46), cv::Scalar(77, 255, 255), testImg); //绿色
// 	// cv::Mat elementRect;
// 	// elementRect = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
// 	// cv::morphologyEx(testImg, testImg, cv::MORPH_CLOSE, elementRect);
// 	//cv::inRange(imghsv, cv::Scalar(29, 50, 0), cv::Scalar(46, 255, 255), testImg); //黄色
// 	//Call ED constructor
// 	ED testED = ED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true); // apply ED algorithm
//     EDCircles testEDCircles = EDCircles(testImg);
// 	Mat circleImg = testEDCircles.drawResult(false, ImageStyle::ELLIPSES);
//     vector<mEllipse> ellipses = testEDCircles.getEllipses();
//     if(ellipses.size()>0)
//     {
//         cout << ellipses.size() << endl;
//         cout << ellipses[0].center.x << endl;
// 	    cout << ellipses[0].center.y << endl;
// 	    cout << ellipses[0].axes.width<< endl;
// 	    cout << ellipses[0].axes.height << endl;
//         for(int ii = int(ellipses[0].center.x) - int(ellipses[0].axes.width) - 10;ii < int(ellipses[0].center.x) - int(ellipses[0].axes.width ) + 10;ii++)
//         {
//             dep1 = image_dep.at<ushort>(int(ellipses[0].center.y), ii );
//             if(dep1 < dep1_min && dep1 > 1000)
//             {
//                 dep1_min = dep1;
//             }
            
//         }
//         dep1 = dep1_min;
//         for(int jj = int(ellipses[0].center.x) + int(ellipses[0].axes.width) - 10;jj < int(ellipses[0].center.x) + int(ellipses[0].axes.width ) + 10;jj++)
//         {
//             dep2 = image_dep.at<ushort>(int(ellipses[0].center.y), jj );
//             if(dep2 < dep2_min && dep2 > 1000)
//             {
//                 dep2_min = dep2;
//             }
            
//         }
//         dep2 = dep2_min;
//         //dep1 = image_dep.at<ushort>(int(ellipses[0].center.y),(int(ellipses[0].center.x) - int(ellipses[0].axes.width )));
//         //dep2 = image_dep.at<ushort>(int(ellipses[0].center.y),(int(ellipses[0].center.x) + int(ellipses[0].axes.width )));
//         double dd = (ellipses[0].center.x - ellipses[0].axes.width - 320) * dep1 / 388;
//         angle1 = acos((dep1 - dep2) / 1100);
//         x_p = ((dd + 550 / sin(angle1) - dep1 / tan(angle1)) * sin(angle1)) * cos(angle1);
//         y_p = ((dd + 550 / sin(angle1) - dep1 / tan(angle1)) * sin(angle1)) * sin(angle1);
//         if(dep1 > dep2)
//         {
//             x_m = (dep1 + dep2) / 2 + 700 * cos(angle1);
//             y_m = dd + 550 * sin(angle1) + 700 * sin(angle1);
//         }
//         else if(dep1 <= dep2)
//         {
//             x_m = (dep1 + dep2) / 2 - 700 * cos(angle1);
//             y_m = dd + 550 * sin(angle1) - 700 * sin(angle1);
//         }
//         //dep1 = image_dep.at<ushort>(int(ellipses[0].center.y),(int(ellipses[0].center.x) - int(ellipses[0].axes.width )));
//         //dep2 = image_dep.at<ushort>(int(ellipses[0].center.y),(int(ellipses[0].center.x) + int(ellipses[0].axes.width )));
//         //double d = (ellipses[0].center.x - ellipses[0].axes.width / 2 - 320) * dep1 / 428.5;
//         //double angle = acos((dep1 - dep2) / 1100);
//         //double x_p = d + 550 / sin(angle) - dep1 / tan(angle);
//         //cout << int(ellipses[0].center.y) <<endl;
//         cout << angle1 << endl;
//         cout << dd << endl;
//         cout << dep1 << endl;
//         cout << dep2 << endl;
//         cout << int(x_p) <<endl;
//         cout << int(y_p) <<endl;
//         cout << int(x_m) <<endl;
//         cout << int(y_m) <<endl;
//         cout << -x_p/1000 <<endl;
//         cout << -1-y_p/1000 <<endl;
//         cout << x_m/1000 <<endl;
//         cout << -1-y_m/1000 <<endl;

//         cout << "--------------------------" << endl;
//     } 
// }

// //识别杆子
// void image_callback1(const sensor_msgs::ImageConstPtr &dep_image,const sensor_msgs::ImageConstPtr &color_image)
// {

//     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(dep_image, "16UC1");
//     image_dep = cv_ptr->image;
//     cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(color_image, "bgr8");
//     img = cv_ptr1->image;
//     Mat image_source = img;
//     Mat img_hsv,img_test;
//     cv::Mat grayImage;
//     cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
//     //cv::imshow("gray", grayImage);
//     // 进行边缘检�???
//     cv::Mat edges;
//     cv::Canny(grayImage, edges, 50, 150);
//     rec_l(edges,image_dep,dep_gan,dep_gan_min,x_middle,y_middle);

//     // //cv::imshow("canny", edges);
//     // //cvtColor(img, img_hsv, COLOR_BGR2HSV);
//     // //cv::inRange(img_hsv, cv::Scalar(0, 0, 59), cv::Scalar(67, 23, 188), img_test); //黄色
//     // std::vector<cv::Vec4i> lines;
//     // cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 300, 60);
    
//     // for (size_t i = 0; i < lines.size(); i++)
//     // {
//     //     if (abs(lines[i][0] - lines[i][2]) < 20 && lines[i][0] >50 && lines[i][0] <440)
//     //     {

//     //         x_middle = (lines[i][0] + lines[i][2]) / 2;
//     //         y_middle = (lines[i][1] + lines[i][3]) / 2;
//     //         break;            
//     //     }
//     // }
//     //     //在原始图像上绘制直线�???
//     // // for (size_t i = 0; i < lines.size(); i++)
//     // // {
//     // //     if (abs(lines[i][0] - lines[i][2]) < 20)
//     // //     {
//     // //         cv::Vec4i line = lines[i];
//     // //         cv::line(image_source, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
//     // //         break;
//     // //     }

//     // // }
//     // dep_gan = 0;
//     // dep_gan_min = 4000;
//     // for(int ii = x_middle - 10;ii < x_middle + 10 ;ii++)
//     // {
//     //     dep_gan = image_dep.at<ushort>(y_middle , ii );
//     //     if(dep_gan < dep_gan_min && dep_gan > 700)
//     //     {
//     //         dep_gan_min = dep_gan;
//     //     }   
//     // }
//     // dep_gan = dep_gan_min;

//     // //dep_gan = image_dep.at<ushort>(y_middle,x_middle);
//     // if(dep_gan > 0 && dep_gan < 3000)
//     // {
//     //     cout << "------------------------" << endl;
//     //     cout << x_middle << endl;
//     //     cout << y_middle << endl;
//     //     cout << dep_gan << endl;
//     //     //cout << lines.size() << endl;       
//     // }
//     // // cout << "------------------------" << endl;
//     // // cv::imshow("123", image_source);
//     // // cv::waitKey(3);
//     // // while(ll == 0)
//     // // {
//     // //     cv::imwrite("~/back/ganzi.png", img);
//     // //     cv::imwrite("~/back/ganzi_dep.png", image_dep);
//     // //     cout << "ok" << endl;
//     // //     ll++;
//     // // }


// }

//定位二维�???
// void image_callback1(const sensor_msgs::ImageConstPtr &dep_image,const sensor_msgs::ImageConstPtr &color_image)
// {

//     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(dep_image, "16UC1");
//     image_dep = cv_ptr->image;
//     cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(color_image, "bgr8");
//     img = cv_ptr1->image;
//     Mat src = img;
// 	Mat srcCopy = img.clone();
// 	//canvas为画�??? 将找到的定位特征画出�???
// 	Mat canvas;
// 	canvas = Mat::zeros(src.size(), CV_8UC3);
// 	Mat srcGray;
// 	//center_all获取特性中�???
// 	vector<Point> center_all;
// 	// 转化为灰度图
// 	cvtColor(src, srcGray, COLOR_BGR2GRAY);
// 	// 3X3模糊
// 	//blur(srcGray, srcGray, Size(3, 3));
// 	// 计算直方�???
// 	convertScaleAbs(src, src);
// 	equalizeHist(srcGray, srcGray);
// 	int s = srcGray.at<Vec3b>(0, 0)[0];
// 	// 设置阈值根据实际情�??? 如视图中已找不到特征 可适量调整
// 	//threshold(srcGray, srcGray, 25, 255, THRESH_BINARY);
// 	threshold(srcGray, srcGray, 150, 255, THRESH_BINARY);
// 	//imshow("threshold", srcGray);
// 	//waitKey();
// 	/*contours是第一次寻找轮�???*/
// 	/*contours2是筛选出的轮�???*/
// 	vector<vector<Point>> contours;
// 	//	用于轮廓检�???
// 	vector<Vec4i> hierarchy;
// 	findContours(srcGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
// 	// 小方块的数量
// 	int numOfRec = 0;
// 	// 检测方�???
// 	int ic = 0;
// 	int parentIdx = -1;
// 	for (int i = 0; i < contours.size(); i++)
// 	{
// 		if (hierarchy[i][2] != -1 && ic == 0)
// 		{
// 			parentIdx = i;
// 			ic++;
// 		}
// 		else if (hierarchy[i][2] != -1)
// 		{
// 			ic++;
// 		}
// 		else if (hierarchy[i][2] == -1)
// 		{
// 			parentIdx = -1;
// 			ic = 0;
// 		}
// 		if (ic >= 2 && ic <= 2)
// 		{
// 			if (IsQrPoint(contours[parentIdx], src)) 
//             {
// 				RotatedRect rect = minAreaRect(Mat(contours[parentIdx]));
// 				// 画图部分
// 				Point2f points[4];
// 				rect.points(points);
// 				for (int j = 0; j < 4; j++) 
//                 {
// 					line(src, points[j], points[(j + 1) % 4], Scalar(0, 255, 0), 2);
// 				}
// 				drawContours(canvas, contours, parentIdx, Scalar(0, 0, 255), -1);
// 				// 如果满足条件则存�???
// 				center_all.push_back(rect.center);
// 				numOfRec++;
// 			}
// 				ic = 0;
// 				parentIdx = -1;
// 		}
// 	}
// 		//vector<Point> center_all1 = center_all;
// 		if (center_all.size() < 3)
// 		{	
// 			cout << "未找到二维码" << endl;
// 		}
// 		if (center_all.size()>=3)
// 		{
// 			int leftTopPointIndex = leftTopPoint(center_all);
// 			Point left_top = center_all[leftTopPointIndex];
// 			cout << "二维码的坐标为："<< left_top<< endl;
//         }
// }
//使用zxing来识别二维码
// void image_callback1(const sensor_msgs::ImageConstPtr &dep_image,const sensor_msgs::ImageConstPtr &color_image)
// {

//     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(dep_image, "16UC1");
//     image_dep = cv_ptr->image;
//     cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(color_image, "bgr8");
//     img = cv_ptr1->image;
// 	Mat src = imread("/home/drone/acfly_ws/src/linetracing/img/qrcode1.png");
// 	Mat srcGray;
// 	cvtColor(src, srcGray, COLOR_BGR2GRAY);
// 	equalizeHist(srcGray, srcGray);
// 	threshold(srcGray, srcGray, 100, 255, THRESH_BINARY);
//     zxing::Ref<zxing::LuminanceSource> source = MatSource::create(srcGray);
//     // int width = source->getWidth();
// 	// int height = source->getHeight();
//     // fprintf(stderr, "image width: %d, height: %d\n", width, height);
//     zxing::Ref<zxing::Reader> reader;
//     //二维�??
//     reader.reset(new zxing::qrcode::QRCodeReader);
//     //条形�??
//     //reader.reset(new zxing::oned::Code128Reader);
//     zxing::Ref<zxing::Binarizer> binarizer(new zxing::GlobalHistogramBinarizer(source));
//     zxing::Ref<zxing::BinaryBitmap> bitmap(new zxing::BinaryBitmap(binarizer));
//     //二维�??
//     zxing::Ref<zxing::Result> result(reader -> decode(bitmap, zxing::DecodeHints(zxing::DecodeHints::QR_CODE_HINT)));
//     //条形�??
//     //zxing::Ref<zxing::Result> result(reader -> decode(bitmap, zxing::DecodeHints(zxing::DecodeHints::CODE_128_HINT)));
//     std::string str = result -> getText() -> getText();
//     fprintf(stderr, "recognization result: %s\n", str.c_str());
// }
//保存RGB照片
// void image_callback1(const sensor_msgs::ImageConstPtr &dep_image,const sensor_msgs::ImageConstPtr &color_image)
// {
//     cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(dep_image, "16UC1");
//     image_dep = cv_ptr->image;
//     cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(color_image, "rgb8");
//     img = cv_ptr1->image;
//     // img.empty();
//     if(!(img.empty()))
//     {
//     cv::imwrite("/home/drone/acfly_ws/src/linetracing/img/qrcode123.jpg", cv_ptr1->image);    
//     cout << "ok" << endl;
//     }

// }
//深度图转换为灰度�??
void image_callback1(const sensor_msgs::ImageConstPtr &dep_image,const sensor_msgs::ImageConstPtr &color_image)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(dep_image, "16UC1");
    image_dep = cv_ptr->image;
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(color_image, "rgb8");
    img = cv_ptr1->image;
    Mat src = image_dep;
    Mat Gray = Mat(image_dep.rows, image_dep.cols, CV_8UC1);
    for (size_t i = 0; i < src.rows; i++)
    {
        uchar* data_gray = Gray.ptr<uchar>(i);
        ushort* data_src = src.ptr<ushort>(i); //16λ��ushort�����Ϊ8λ���Ϊuchar
        for (size_t j = 0; j < src.cols; j++)
        {
            if (data_src[j]<3000 && data_src[j]>500) //max,min��ʾ��Ҫ�����ֵ�����/Сֵ
            {
                //cout << data_src[j] << endl;
                //cout << (data_src[j] - 400) / (800 - 400) * 255.0f << endl;
                data_gray[j] = (data_src[j] - 500) / (3000 - 500) * 255.0f;
                //cout << int(data_gray[j]) << endl;
                //cout << 20/400 * 255.0f << endl;
            }
            else
            {
                data_gray[j] = 255;
            }
        }

    }
    if(!(Gray.empty()))
    {
    cv::imwrite("/home/drone/acfly_ws/src/linetracing/img/gray.jpg", Gray);    
    cout << "ok" << endl;
    }

}
//单独图像识别
// void image_callback1(const sensor_msgs::ImageConstPtr &color_image)
// {
//     // try
//     // {
//     cv_bridge:CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(color_image, "bgr8");
//     img = cv_ptr1->image;
// }
int main(int argc, char** argv)
{
    ros::init(argc, argv, "dep_img_get");
    ros::NodeHandle nh;
    //message_filters::Subscriber sub1 = nh.subscribe("/camera/depth/image_rect_raw", 1000, image_callback);
    // ros::Subscriber sub2 = nh.subscribe("/camera/color/image_raw", 1000, image_callback1);
    // message_filters::Subscriber<sensor_msgs::Image> sub_color_image(nh, "/camera/aligned_depth_to_color/image_raw", 2000, ros::TransportHints().tcpNoDelay());
    message_filters::Subscriber<sensor_msgs::Image> sub_color_image(nh, "/camera/depth/image_rect_raw", 2000, ros::TransportHints().tcpNoDelay());
    message_filters::Subscriber<sensor_msgs::Image> sub_right_image(nh, "/camera/color/image_raw", 2000, ros::TransportHints().tcpNoDelay());
    typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
    Synchronizer<syncPolicy> sync(syncPolicy(10), sub_color_image, sub_right_image);
    //指定一个回调函数，就可以实现两个话题数据的同步获取
    sync.registerCallback(boost::bind(&image_callback1, _1, _2));
    ros::Rate rate(10);
    // for( int o = 0 ; o < 10;o++)
    // {
    //     ros::spinOnce();
    // }
    ros::spin();
    // Mat src = imread("/home/drone/acfly_ws/src/linetracing/img/qrcode1.png");
    // cv::imwrite("/home/drone/acfly_ws/src/linetracing/img/qrcode11.jpg", src);
    return 0;
}
