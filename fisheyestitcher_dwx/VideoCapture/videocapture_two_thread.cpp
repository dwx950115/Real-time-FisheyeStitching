#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <unistd.h>

using namespace std;
using namespace cv;

// HANDLE HThread1, HThread2;
cv::Mat g_matFrame1, g_matFrame2;
void thread1()
{
	VideoCapture capture;
	capture.open(0);
	while (1)
	{
		capture >> g_matFrame1;
	}
}
void thread2()
{
	VideoCapture capture;
	capture.open(1);
	while (1)
	{
		capture >> g_matFrame2;
	}
}

int main()
{
	// HThread1 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)thread1, NULL, 0, 0);
	// HThread2 = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)thread2, NULL, 0, 0);
	thread thread_obj1(thread1);
	thread thread_obj2(thread2);
	sleep(1);
	int iFrameCount = 0;
	while (1)
	{
		if (g_matFrame1.empty() || g_matFrame2.empty())
		{
			printf("empty frame!\n");
			continue;
		}
		printf("Frame: %d\n", iFrameCount);
		imshow("camera 1:", g_matFrame1);
		imshow("camera 2:", g_matFrame2);
		waitKey(30);
		iFrameCount++;
	}

	thread_obj1.join();
	thread_obj2.join();
	// CloseHandle(HThread1);
	// CloseHandle(HThread2);
	return 0;
}


