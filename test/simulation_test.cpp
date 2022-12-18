#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <memory>
#include <thread>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 1e4;
string sConfig_path = "../config/";
string sData_path = "/home/dataset/EuRoC/MH-05/mav0/";
std::shared_ptr<System> simSystem;

void PubImuData() {
  std::string imu_data_file = sConfig_path + "imu_data.txt";
  cout << "1 PubImuData start imu_data_file: " << imu_data_file << endl;
  ifstream fsimu;
  fsimu.open(imu_data_file.c_str());
  if (!fsimu.is_open()) {
    cerr<<"Failed tp oen imu file!"<<imu_data_file<<endl;
    return;
  }

  std::string imu_line;
  double dstampNsec = 0.0;
  Vector3d acc, gyr;
  while (getline(fsimu, imu_line) && !imu_line.empty()) { // read imu data
    std::istringstream ssimudata(imu_line);
    ssimudata >> dstampNsec >> gyr.x() >> gyr.y() >> gyr.z() 
              >> acc.x() >> acc.y() >> acc.z();
    simSystem->PubImuData(dstampNsec, gyr, acc);
    usleep(nDelayTimes);
  }
  fsimu.close();
}

void PubImageData() {
  string sImage_file = sConfig_path + "sim_cam.txt";
  cout<<"1 PubImageData start sImage_file: "<<sImage_file<<endl;
  ifstream fsImage;
  fsImage.open(sImage_file.c_str());
  if (!fsImage.is_open()) {
    cerr<<"Failed to open image file"<<sImage_file<<endl;
    return;
  }

  std::string feature_points, feature_file;
  double dStampNsec;
  while (std::getline(fsImage, feature_points) && !feature_points.empty()) {
    std::istringstream ssfeaturedata(feature_points);
    ssfeaturedata>>dStampNsec>>feature_file;
    string featurePath = sData_path + feature_file;
    ifstream featurePath_data;
    featurePath_data.open(featurePath.c_str());
    if (!fsImage.is_open()) {
      cerr<<"Failed to open image file"<<featurePath<<endl;
      return;
    }
    string key_frame_points;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points_per_cam;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> features_per_cam;
    while(std::getline(featurePath_data, key_frame_points) && !key_frame_points.empty()){//read feature points in camera
      std::istringstream ssfeature(key_frame_points);
      Eigen::Vector4d cam_points;
      Eigen::Vector2d features;
      ssfeature>>cam_points(0)>>cam_points(1)>>
          cam_points(2)>>cam_points(3)>>features(0)>>features(1);
      // cout<<featurePath<<" "<<dStampNsec<<" "<<cam_points(1)<<" "
      //       <<cam_points(2)<<" "<<cam_points(3)<<endl;
      points_per_cam.push_back(cam_points);
      features_per_cam.push_back(features);
    }
    featurePath_data.close();
    simSystem->PubImageData(dStampNsec, points_per_cam, features_per_cam);
    usleep(7*nDelayTimes);
  }
  fsImage.close();
}

int main(int argc, char **argv)
{
	if(argc != 3)
	{
		cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n" 
			<< "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/"<< endl;
		return -1;
	}
	sData_path = argv[1];
	sConfig_path = argv[2];

	simSystem.reset(new System(sConfig_path + "sim_config.yaml"));
	
	std::thread thd_BackEnd(&System::ProcessBackEnd, simSystem);
		
	// sleep(5);
	std::thread thd_PubImuData(PubImuData);

	std::thread thd_PubImageData(PubImageData);
// #ifdef __linux__	
	std::thread thd_Draw(&System::Draw, simSystem);
// #elif __APPLE__
// 	DrawIMGandGLinMainThrd();
// #endif
	thd_PubImuData.join();
	thd_PubImageData.join();

	thd_BackEnd.join();
	thd_Draw.join();

	cout << "main end... see you ..." << endl;
	return 0;
}