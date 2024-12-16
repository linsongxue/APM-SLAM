#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <stdio.h>
#include <ros/ros.h>
#include <loop_fusion/QueryToMatch.h>
#include <colmap/estimators/pose.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/scene/image.h>
#include <colmap/scene/point3d.h>
#include <cmath>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <numeric>
#include "keyframe.h"
#include "pose_graph.h"
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

// enum INIT_TYPE
// {
//     ODO,
//     VL
// };

const int MAP_WINDOW_SIZE = 20;

class MapLocalization
{
public:
    MapLocalization();
    ~MapLocalization();
    void prepare_process(ros::NodeHandle &n, const std::string &calib_file, const std::string &colmap_database_path);
    void loadSfM(const std::string &colmap_database_path);
    void addKeyFrame(KeyFrame *keyframe);
    void process();
    bool visualLocalization(KeyFrame *keyframe);
    bool initialization();
    bool checkValid(KeyFrame *keyframe);
    void tightlyCoupledOptimization();
    void computePoseLoss();
    void computePointLoss();
    void slideWindow();
    void pointStatistic();
    void readIntrinsicParameter(const std::string &calib_file);
    cv::Point2f pixel2cam(const Eigen::Vector2d &p);
    void loadParameters();
    void updateParameters();
    void publish();
    void publish_path(const std_msgs::Header &header);
    void publish_point_cloud(const std_msgs::Header& header);
    void publish_TransedVIO(const std_msgs::Header &header);
    void publish_visualLoc(const std_msgs::Header &header);
    // void publish_validLoc(const std_msgs::Header &header);

    void count_vl();
    // void read_gt();

    void registerPub(ros::NodeHandle &n);
    void registerClient(ros::NodeHandle &n);
    void decision_candidate_pose(vector<float> &pos, const Eigen::Vector3d &vio_t);
    void PnPwithIntrinsic(
        const KeyFrame *keyframe, 
        const vector<cv::Point2f> &imagePoints, 
        const vector<cv::Point3f> &objectPoints, 
        Eigen::Matrix3d &R_b2w, 
        Eigen::Vector3d &t_b2w, 
        vector<uchar> &status, 
        bool useIntrinsic = false);

    bool running;
    int window_index;
    bool initialized;
    queue<KeyFrame*> keyframe_buffer;
    KeyFrame* optim_window[MAP_WINDOW_SIZE];
    // vector<INIT_TYPE> init_type;
    double param_window_qs[MAP_WINDOW_SIZE][4];  // w qx qy qz
    double param_window_ts[MAP_WINDOW_SIZE][3];  // x y z

    vector<vector<double>> pointsCLoud;

    std::mutex m_buffer;

    ros::ServiceClient client;
    ros::Publisher pub_path;
    ros::Publisher pub_point_cloud;
    ros::Publisher pub_transed_vio;
    ros::Publisher pub_visual_loc;
    // ros::Publisher pub_valid_loc;
    // ros::Publisher pub_gt;
    // ros::Publisher pub_camera_direct;
    nav_msgs::Path path;
    nav_msgs::Path transedPath;
    nav_msgs::Path visualLocPath;
    // nav_msgs::Path gtPath;
    // visualization_msgs::MarkerArray markerArray_msg;

    colmap::Reconstruction reconstruction;
    camodocal::CameraPtr camera;

    Eigen::Vector3d t_vio2vlw;
    Eigen::Matrix3d R_vio2vlw;
    Eigen::Vector3d last_t;
    Eigen::Matrix3d last_R;
    std::thread t_localization;

    // void test_client();
    // std::unordered_map<long, std::vector<double>> gt_data;
    int all_keyframes = 0;
    int vl_succ_keyframes = 0;
};

struct ReprojectionError
{
    ReprojectionError(
        double observed_x_, double observed_y_, 
        Eigen::Matrix3d qic_,
        Eigen::Vector3d tic_) : observed_x(observed_x_), observed_y(observed_y_)
        {
            qci = Eigen::Quaterniond(qic_.transpose());
            tci = qic_.transpose() * (-tic_);
        }

    template <typename T>
    bool operator()(const T *const rotation, const T *const translation, const T *const point, T *residuals) const
    {
        T q_i2c[4] = {T(qci.w()), T(qci.x()), T(qci.y()), T(qci.z())};
        T t_i2c[3] = {T(tci.x()), T(tci.y()), T(tci.z())};

        T q_i2w[4] = {T(rotation[0]), T(rotation[1]), T(rotation[2]), T(rotation[3])};
        T t_i2w[3] = {T(translation[0]), T(translation[1]), T(translation[2])};

        T q_w2i[4];
        T t_w2i[3];
        // world frame to body frame
        QuaternionInverse(q_i2w, q_w2i);
        ceres::QuaternionRotatePoint(q_w2i, t_i2w, t_w2i);
        t_w2i[0] = -t_w2i[0];
        t_w2i[1] = -t_w2i[1];
        t_w2i[2] = -t_w2i[2]; 

        // world frame to camera frame
        T q_w2c[4];
        T t_w2c[3];
        ceres::QuaternionProduct(q_i2c, q_w2i, q_w2c);
        ceres::QuaternionRotatePoint(q_i2c, t_w2i, t_w2c);
        t_w2c[0] += t_i2c[0];
        t_w2c[1] += t_i2c[1];
        t_w2c[2] += t_i2c[2];

        // projection
        T p[3];
        ceres::QuaternionRotatePoint(q_w2c, point, p);
        p[0] += t_w2c[0];
        p[1] += t_w2c[1];
        p[2] += t_w2c[2];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        residuals[0] = T(500.0) * (xp - T(observed_x));
        residuals[1] = T(500.0) * (yp - T(observed_y));

        return true;
    }

    static ceres::CostFunction *Create(
        double observed_x, double observed_y,
        Eigen::Matrix3d qic, Eigen::Vector3d tic)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
            new ReprojectionError(observed_x, observed_y, qic, tic)));
    }

    double observed_x, observed_y;
    Eigen::Quaterniond qci;
    Eigen::Vector3d tci;
};

struct PointPriori
{
    PointPriori(double x, double y, double z) : x(x), y(y), z(z) {}

    template <typename T>
    bool operator()(const T *const point, T *residuals) const
    {
        residuals[0] = point[0] - T(x);
        residuals[1] = point[1] - T(y);
        residuals[2] = point[2] - T(z);

        return true;
    }

    static ceres::CostFunction *Create(double x, double y, double z)
    {
        return (new ceres::AutoDiffCostFunction<PointPriori, 3, 3>(
            new PointPriori(x, y, z)));
    }

    double x, y, z;
};

struct ReferenceError
{
    ReferenceError(Eigen::Matrix3d R_i2w_, Eigen::Vector3d t_i2w_, double t_var_, double q_var_) : Q_ref(R_i2w_), T_ref(t_i2w_), t_var(t_var_), q_var(q_var_) {}

    template <typename T>
    bool operator()(const T *const rotation, const T *const translation, T *residuals) const
    {
        T q_ref[4] = {T(Q_ref.w()), T(Q_ref.x()), T(Q_ref.y()), T(Q_ref.z())};
        T t_ref[3] = {T(T_ref.x()), T(T_ref.y()), T(T_ref.z())};

        residuals[0] = (translation[0] - t_ref[0]) / T(t_var);
        residuals[1] = (translation[1] - t_ref[1]) / T(t_var);
        residuals[2] = (translation[2] - t_ref[2]) / T(t_var);

        T q_ref_inv[4];
        QuaternionInverse(q_ref, q_ref_inv);

        T error_q[4];
        ceres::QuaternionProduct(q_ref_inv, rotation, error_q);

        residuals[3] = T(2) * error_q[1] / T(q_var);
        residuals[4] = T(2) * error_q[2] / T(q_var);
        residuals[5] = T(2) * error_q[3] / T(q_var);

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Matrix3d R_i2w_, Eigen::Vector3d t_i2w_, double t_var_, double q_var_)
    {
        return (new ceres::AutoDiffCostFunction<ReferenceError, 6, 4, 3>(
            new ReferenceError(R_i2w_, t_i2w_, t_var_, q_var_)));
    }

    Eigen::Quaterniond Q_ref;
    Eigen::Vector3d T_ref;
    double q_var, t_var;
};
