#include <stdio.h>
#include <string>
#include <cv.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>

using namespace std;

int main(int argc, char** argv) {
	/*vector<CvRect> faceWindow;
	FaceDetection  *face_detector = new FaceDetection("cascade Data\\haarcascade.xml");
	IplImage *img = cvLoadImage("001_41_505_130.bmp");
	faceWindow = face_detector->Detect(img);
	*/
	IplImage* img = cvLoadImage(argv[1]);
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
	cvCvtColor(img, gray, CV_BGR2GRAY);
    cvEqualizeHist(gray, gray);

	string cascade_name = "cascade Data\\haarcascade_frontalface_alt2.xml";
	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*) cvLoad(cascade_name.c_str());

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* faces = cvHaarDetectObjects(gray, cascade, storage, 1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/, cvSize(30, 30));

	for(int i = 0; i < faces->total; i++ )
    {
        /* extract the rectangles only */
        CvRect* face_rect = (CvRect*) cvGetSeqElem(faces, i);        
        cvRectangle( img, cvPoint(face_rect->x,face_rect->y),
                     cvPoint((face_rect->x+face_rect->width),
                             (face_rect->y+face_rect->height)),
                     CV_RGB(255,0,0), 3 );
    } 
/*
	IplImage* img = cvLoadImage(argv[1]);
	cvNamedWindow("Example1", CV_WINDOW_AUTOSIZE);
	cvShowImage("Example1", img);
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow("Example1");*/
	cvNamedWindow("Example1", CV_WINDOW_AUTOSIZE);
	cvShowImage("Example1", img);
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvReleaseImage(&gray);
}