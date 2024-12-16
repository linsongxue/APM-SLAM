/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

struct MapImage
{
    Eigen::Quaterniond Q_cw;
    Eigen::Vector3d t_cw;
    std::string image_name;
    std::map<int, cv::Point2f> points2D;
    std::vector<int> descriptors_idx;
    std::unordered_map<int, Eigen::Vector3d> points3D_c;  // map point coordinate (3-dim) in cam frame
    Eigen::Quaterniond Q_wc;
    Eigen::Vector3d t_wc;

    MapImage()
    {
        image_name = "";
    }

    bool operator==(const MapImage &other) const
    {
        return image_name == other.image_name;
    }
};

class FeatureTracker
{
public:
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    void setMask();
    void setMapMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();
    bool inBorder(const cv::Point2f &pt);
    // bool setPairs(const std::string &pair_path);
    // bool parsePoints3DTxt(const std::string &points3D_txt);
    // bool parseImagesTxt(const std::string &images_txt);
    // map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImageWithMap(const string &img_name, double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());
    // void projMapPointOF(const cv::Mat &mapImg, const MapImage &imgInfo, vector<cv::Point2f> &cur_pts, vector<int> &points3D_id);

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time;
    double prev_time;
    bool stereo_cam;
    int n_id;
    bool hasPrediction;
    // std::unordered_map<std::string, std::string> pairs;
    // std::string mapBaseDir;
    // std::string ImgName;
    // cv::Mat refImg;
    // std::unordered_map<int, Eigen::Vector3d> points3D;
    // std::unordered_map<int, double> points3D_error;
    // std::unordered_map<std::string, MapImage> images;
    // vector<int> map_ids, map_ids_right;
    // std::unordered_map<int, int> feature_mapp3d;
    // MapImage cur_refimg;
    // Eigen::Matrix3d coarse_R = Eigen::Matrix3d::Identity();
    // Eigen::Vector3d coarse_t = Eigen::Vector3d::Zero();
    // bool estimate_T = false;
};
