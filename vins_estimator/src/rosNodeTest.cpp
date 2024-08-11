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

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include <geometry_msgs/PoseStamped.h>

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;
queue<geometry_msgs::PoseStamped::ConstPtr> coarse_pose_buf;
std::mutex m_pose_buf;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

void coarse_pose_callback(const geometry_msgs::PoseStamped::ConstPtr &pose_msg)
{
    m_pose_buf.lock();
    coarse_pose_buf.push(pose_msg);
    m_pose_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

void getPoseFromMsg(const geometry_msgs::PoseStamped::ConstPtr &pose_msg, Eigen::Vector3d &position, Eigen::Quaterniond &quaternion)
{
    position.x() = pose_msg->pose.position.x;
    position.y() = pose_msg->pose.position.y;
    position.z() = pose_msg->pose.position.z;

    quaternion.w() = pose_msg->pose.orientation.w;
    quaternion.x() = pose_msg->pose.orientation.x;
    quaternion.y() = pose_msg->pose.orientation.y;
    quaternion.z() = pose_msg->pose.orientation.z;
}

// extract images with same timestamp from two topics
void sync_process()
{
    while(1)
    {
        if(STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // 0.003s sync tolerance
                if(time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                    //printf("find img0 and img1\n");
                }
            }
            m_buf.unlock();
            // recieve pose data
            Eigen::Vector3d coarse_position;
            Eigen::Quaterniond coarse_quaternion;
            bool got_coarse_pose = false;
            m_pose_buf.lock();
            while(!coarse_pose_buf.empty() && coarse_pose_buf.front()->header.stamp.toSec() < time)
            {
                coarse_pose_buf.pop();
            }
            if (!coarse_pose_buf.empty())
            {
                double pose_time = coarse_pose_buf.front()->header.stamp.toSec();
                if(abs(pose_time - time) < 0.05)
                {
                    getPoseFromMsg(coarse_pose_buf.front(), coarse_position, coarse_quaternion);
                    coarse_pose_buf.pop();
                    got_coarse_pose = true;
                }
            }
            m_pose_buf.unlock();
            if(got_coarse_pose)
            {
                estimator.setCoarsePose(coarse_position, coarse_quaternion);
            }
            // end recieve pose data
            if(!image0.empty())
            {
                estimator.inputImage(time, image0, image1);
                // write result to file with kitti format
                ofstream foutKitti(VINS_RESULT_KITTI, ios::app);
                foutKitti.setf(ios::fixed, ios::floatfield);
                foutKitti.precision(5);
                Eigen::Matrix<double, 4, 4> pose;
                estimator.getPoseInWorldFrame(pose);
                foutKitti << pose(0, 0) << " "
                          << pose(0, 1) << " "
                          << pose(0, 2) << " "
                          << pose(0, 3) << " "
                          << pose(1, 0) << " "
                          << pose(1, 1) << " "
                          << pose(1, 2) << " "
                          << pose(1, 3) << " "
                          << pose(2, 0) << " "
                          << pose(2, 1) << " "
                          << pose(2, 2) << " "
                          << pose(2, 3) << endl;
                foutKitti.close();
            }
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if(!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();
            // recieve pose data
            Eigen::Vector3d coarse_position;
            Eigen::Quaterniond coarse_quaternion;
            bool got_coarse_pose = false;
            m_pose_buf.lock();
            while (!coarse_pose_buf.empty() && coarse_pose_buf.front()->header.stamp.toSec() < time)
            {
                coarse_pose_buf.pop();
            }
            if (!coarse_pose_buf.empty())
            {
                double pose_time = coarse_pose_buf.front()->header.stamp.toSec();
                if (abs(pose_time - time) < 0.05)
                {
                    getPoseFromMsg(coarse_pose_buf.front(), coarse_position, coarse_quaternion);
                    coarse_pose_buf.pop();
                    got_coarse_pose = true;
                }
            }
            m_pose_buf.unlock();
            if (got_coarse_pose)
            {
                estimator.setCoarsePose(coarse_position, coarse_quaternion);
            }
            // end recieve pose data
            if(!image.empty())
            {
                estimator.inputImage(time, image);
                // write result to file with kitti format
                ofstream foutKitti(VINS_RESULT_KITTI, ios::app);
                foutKitti.setf(ios::fixed, ios::floatfield);
                foutKitti.precision(5);
                Eigen::Matrix<double, 4, 4> pose;
                estimator.getPoseInWorldFrame(pose);
                foutKitti << pose(0, 0) << " "
                          << pose(0, 1) << " "
                          << pose(0, 2) << " "
                          << pose(0, 3) << " "
                          << pose(1, 0) << " "
                          << pose(1, 1) << " "
                          << pose(1, 2) << " "
                          << pose(1, 3) << " "
                          << pose(2, 0) << " "
                          << pose(2, 1) << " "
                          << pose(2, 2) << " "
                          << pose(2, 3) << endl;
                foutKitti.close();
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu;
    if(USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1;
    if(STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    ros::Subscriber sub_coarse_pose;
    if(!COARSE_POSE.empty())
    {
        ROS_INFO("start recieve reference pose from %s", COARSE_POSE.c_str());
        sub_coarse_pose = n.subscribe(COARSE_POSE, 1000, coarse_pose_callback);
    }
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;
}
