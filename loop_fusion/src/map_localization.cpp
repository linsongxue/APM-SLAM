#include "map_localization.h"

nav_msgs::Path path;

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void RotationAvg(const std::vector<Eigen::Matrix3d> &Rs, Eigen::Matrix3d &avg_R)
{
    avg_R = Eigen::Matrix3d::Zero();
    for (const auto &R : Rs)
    {
        avg_R += R;
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(avg_R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    avg_R = svd.matrixU() * svd.matrixV().transpose();
}

void computeReprojectError(
    const vector<cv::Point3f> &points_3d,
    const vector<cv::Point2f> &points_2d,
    const Eigen::Matrix3d &R_w2c,
    const Eigen::Vector3d &t_w2c,
    vector<double> &errors,
    vector<double> &depths,
    const camodocal::CameraPtr &camera)
{
    errors.clear();
    errors.resize(points_3d.size());
    depths.clear();
    depths.resize(points_3d.size());
    for (int i = 0; i < points_3d.size(); ++i)
    {
        Eigen::Vector3d p3d(points_3d[i].x, points_3d[i].y, points_3d[i].z);
        p3d = R_w2c * p3d + t_w2c;
        depths[i] = p3d.z();
        p3d = p3d / p3d.z();
        Eigen::Vector2d p2d;
        camera->spaceToPlane(p3d, p2d);
        // Eigen::Vector2d p2d = p3d.head<2>();
        Eigen::Vector2d dist = p2d - Eigen::Vector2d(points_2d[i].x, points_2d[i].y);
        errors[i] = dist.norm();
    }
}

void MapLocalization::PnPwithIntrinsic(
    const KeyFrame* keyframe,
    const vector<cv::Point2f> &imagePoints,
    const vector<cv::Point3f> &objectPoints, 
    Eigen::Matrix3d &R_b2w, 
    Eigen::Vector3d &t_b2w,
    vector<uchar> &status,
    bool useIntrinsic)
{
    boost::shared_ptr<camodocal::PinholeCamera> pinhole_camera = boost::dynamic_pointer_cast<camodocal::PinholeCamera>(camera);
    camodocal::PinholeCamera::Parameters camera_param = pinhole_camera->getParameters();
    double fx = camera_param.fx();
    double fy = camera_param.fy();
    double cx = camera_param.cx();
    double cy = camera_param.cy();
    double k1 = camera_param.k1();
    double k2 = camera_param.k2();
    double p1 = camera_param.p1();
    double p2 = camera_param.p2();

    cv::Mat rvec, tvec, R, tmp_r;
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_vio_c = keyframe->vio_R_w_i * qic;
    Vector3d T_vio_c = keyframe->vio_T_w_i + keyframe->vio_R_w_i * tic;
    Matrix3d R_w_c = R_vio2vlw * R_vio_c;
    Vector3d T_w_c = R_vio2vlw * T_vio_c + t_vio2vlw;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, tvec);

    cv::Mat K;
    cv::Mat D;
    if(useIntrinsic)
    {
        K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1.0);
        D = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);
    }
    else
    {
        K = (cv::Mat_<double>(3, 3) << 1., 0, 0, 0, 1., 0, 0, 0, 1.);
        D = (cv::Mat_<double>(1, 4) << 0, 0, 0, 0);
    }
    cv::Mat inliers;
    cv::solvePnPRansac(objectPoints, imagePoints, K, D, rvec, tvec, true, 100, 10.0 / 460.0, 0.99, inliers);

    for (int i = 0; i < (int)imagePoints.size(); i++)
        status.push_back(0);

    for (int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, R);
    Matrix3d R_pnp, R_c2w;
    cv::cv2eigen(R, R_pnp);
    R_c2w = R_pnp.transpose();
    Vector3d T_pnp, t_c2w;
    cv::cv2eigen(tvec, T_pnp);
    t_c2w = R_c2w * (-T_pnp);

    R_b2w = R_c2w * qic.transpose();
    t_b2w = t_c2w - R_b2w * tic;
}

// void MapLocalization::test_client()
// {
//     loop_fusion::QueryToMatch srv;
//     vector<float> pos(3, 0.);

//     if (client.waitForExistence(ros::Duration(5.0)))
//     {
//         ROS_INFO("Service is available.");
//     }
//     else
//     {
//         ROS_ERROR("Service did not become available within the timeout.");
//         exit(1);
//     }

//     srv.request.query_name = "cam0/1586247632040008192.png";
//     srv.request.candidate_pos[0] = pos[0];
//     srv.request.candidate_pos[1] = pos[1];
//     srv.request.candidate_pos[2] = pos[2];
//     ROS_INFO("query_name: %s", srv.request.query_name.c_str());
//     ROS_INFO("candidate_pos: %f, %f, %f", srv.request.candidate_pos[0], srv.request.candidate_pos[1], srv.request.candidate_pos[2]);

//     if (client.call(srv))
//     {
//         printf("num_pairs: %d\n", srv.response.num_pairs);
//     }
//     else
//     {
//         ROS_ERROR("Failed to call service QueryToMatch");
//         exit(1);
//     }
// }

MapLocalization::MapLocalization()
{
    initialized = false;
    running = false;
    R_vio2vlw = Eigen::Matrix3d::Identity();
    t_vio2vlw = Eigen::Vector3d::Zero();
    // read_gt();
}

MapLocalization::~MapLocalization()
{
    running = false;
    if(t_localization.joinable()) t_localization.join();
}

// void MapLocalization::read_gt()
// {
//     std::ifstream file("/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-04-07_10-20-32/gt_tum_ns.txt");
//     if (!file.is_open())
//     {
//         std::cerr << "Failed to open file" << std::endl;
//         return;
//     }

//     std::string line;
//     while (std::getline(file, line))
//     {
//         std::istringstream iss(line);
//         long key;
//         double value;
//         std::vector<double> values;

//         // 读取第一个整数作为键
//         if (!(iss >> key))
//         {
//             std::cerr << "Error reading key from line: " << line << std::endl;
//             continue;
//         }

//         // 读取后面的7个浮点数作为值
//         for (int i = 0; i < 7; ++i)
//         {
//             if (!(iss >> value))
//             {
//                 std::cerr << "Error reading value from line: " << line << std::endl;
//                 continue;
//             }
//             values.push_back(value);
//         }

//         // 将键值对插入到字典中
//         gt_data[key] = values;
//     }

//     file.close();
// }

void MapLocalization::prepare_process(
    ros::NodeHandle &n, 
    const std::string &calib_file, 
    const std::string &colmap_database_path)
{
    registerPub(n);
    registerClient(n);
    readIntrinsicParameter(calib_file);
    loadSfM(colmap_database_path);
    running = true;
    // test_client();
    t_localization = std::thread(&MapLocalization::process, this);
}

void MapLocalization::registerPub(ros::NodeHandle &n)
{
    pub_path = n.advertise<nav_msgs::Path>("map_localization_path", 1000);
    // pub_transed_vio = n.advertise<nav_msgs::Path>("map_localization_transed_vio", 1000);
    pub_visual_loc = n.advertise<nav_msgs::Path>("map_localization_visual_loc", 1000);
    // pub_gt = n.advertise<nav_msgs::Path>("map_localization_gt", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("map_localization_point_cloud", 1000);
    // pub_camera_direct = n.advertise<visualization_msgs::MarkerArray>("map_localization_camera_direct", 1000);
    ROS_INFO("register publisher finish\n");
}

void MapLocalization::loadSfM(const std::string &colmap_database_path)
{
    reconstruction.Read(colmap_database_path);
    ROS_INFO("load colmap database finish\n");
}

void MapLocalization::registerClient(ros::NodeHandle &n)
{
    client = n.serviceClient<loop_fusion::QueryToMatch>("/match_info");
    ROS_INFO("register client finish\n");
}

void MapLocalization::addKeyFrame(KeyFrame* keyframe)
{
    m_buffer.lock();
    keyframe_buffer.push(keyframe);
    m_buffer.unlock();
}

void MapLocalization::process()
{
    while(running)
    {
        if (keyframe_buffer.empty())
        {
            std::chrono::milliseconds dura(500);
            std::this_thread::sleep_for(dura);
            continue;
        }
        m_buffer.lock();
        KeyFrame* keyframe = keyframe_buffer.front();
        keyframe_buffer.pop();
        m_buffer.unlock();

        bool vl_succ = visualLocalization(keyframe);
        keyframe->VL_res_succ = vl_succ;
        float succ_ratio = float(vl_succ_keyframes) / float(all_keyframes);
        ROS_DEBUG("visual localization success: %f", succ_ratio);

        if(window_index < MAP_WINDOW_SIZE)
        {
            optim_window[window_index] = keyframe;
            window_index++;
        }
        else
        {
            optim_window[window_index - 1] = keyframe;
        }

        if (initialized)
        {
            bool VL_valid = checkValid(keyframe);
            keyframe->VL_res_valid = VL_valid;
            tightlyCoupledOptimization();
            publish();
            slideWindow();
        }
        else
        {
            initialized = initialization();
            if (initialized)
            {
                tightlyCoupledOptimization();
                publish();
                slideWindow();
            }
            else
            {
                if(window_index == MAP_WINDOW_SIZE)
                {
                    slideWindow();
                }
            }
        }

        std::chrono::milliseconds dura(500);
        std::this_thread::sleep_for(dura);
    }
}

bool MapLocalization::visualLocalization(KeyFrame *keyframe)
{
    all_keyframes++;
    loop_fusion::QueryToMatch srv;
    vector<float> pos(3, 0.);
    decision_candidate_pose(pos, keyframe->vio_T_w_i);

    srv.request.query_name = "cam0/" + keyframe->nanosecond + ".png";
    srv.request.candidate_pos = {pos[0], pos[1], pos[2]};

    if (client.call(srv))
    {
        int num_pairs = srv.response.num_pairs;
        int num_points = srv.response.num_points;
        vector<int> match0_ids = srv.response.match0_ids;
        vector<int> match0 = srv.response.match0;
        vector<float> pts2d = srv.response.pts2d;
        vector<float> scores = srv.response.score0;
        vector<cv::Point2f> pts2d_img_i;
        vector<cv::Point2f> pts2d_norm_cam_i;
        vector<cv::Point3f> pts3d_w_i;
        vector<int> pts3d_id;
        vector<double> pts_map_error;
        Eigen::Vector3d ref_pos;
        // begin debug
        // string query_img_path = "/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-04-07_10-20-32/undistorted_images/" + srv.request.query_name;
        // cv::Mat query_image = cv::imread(query_img_path, cv::IMREAD_COLOR);
        // cv::Mat combined_image;
        // std::vector<cv::Point2f> query_points;
        // std::vector<cv::Point2f> ref_points;
        // string ref_img_name;
        // end debug
        for (int ipair = 0; ipair < num_pairs; ipair++)
        {
            uint32_t matched_id = static_cast<uint32_t>(match0_ids[ipair]);
            const colmap::Image& image = reconstruction.Image(matched_id);
            ref_pos = image.CamFromWorld().rotation.inverse() * image.CamFromWorld().translation;
            ref_pos = -ref_pos;
            // begin debug
            // ref_img_name = image.Name();
            // end debug
            for (int jpoint = 0; jpoint < num_points; jpoint++)
            {
                int gpi = ipair * num_points + jpoint; // global point index in match0
                if (match0[gpi] == -1)
                {
                    continue;
                }
                if(scores[gpi] < 0.85)
                {
                    continue;
                }
                uint32_t p2d_id = static_cast<uint32_t>(match0[gpi]);
                if (!image.Point2D(p2d_id).HasPoint3D())
                {
                    continue;
                }
                // get point in world frame
                pts3d_id.emplace_back(image.Point2D(p2d_id).point3D_id);
                const Eigen::Vector3d &pt3d = reconstruction.Point3D(image.Point2D(p2d_id).point3D_id).xyz;
                pts3d_w_i.emplace_back(cv::Point3f(pt3d.x(), pt3d.y(), pt3d.z()));
                pts_map_error.emplace_back(reconstruction.Point3D(image.Point2D(p2d_id).point3D_id).error);
                // get pixel point in camera frame
                cv::Point2f pt2d = pixel2cam(Eigen::Vector2d(pts2d[2 * jpoint], pts2d[2 * jpoint + 1]));
                pts2d_norm_cam_i.emplace_back(pt2d);
                pts2d_img_i.emplace_back(cv::Point2f(pts2d[2 * jpoint], pts2d[2 * jpoint + 1]));

                // begin debug
                // if(ipair == 0)
                // {
                //     query_points.push_back(pts2d_img_i.back());
                //     ref_points.push_back(cv::Point2f(image.Point2D(p2d_id).xy[0], image.Point2D(p2d_id).xy[1]));
                // }
                // end debug
            }
            // begin debug
            // if(ipair == 0)
            // {
            //     string ref_img_path = "/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-03-24_17-36-22/sfm_map/" + image.Name();
            //     cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
            //     cv::hconcat(query_image, ref_image, combined_image);
            //     for (size_t i = 0; i < query_points.size(); i++)
            //     {
            //         cv::Point2f pt_query = query_points[i];
            //         cv::Point2f pt_ref = ref_points[i] + cv::Point2f(query_image.cols, 0); // Adjust reference points' x-coordinate

            //         cv::circle(combined_image, pt_query, 3, cv::Scalar(0, 255, 0), -1);
            //         cv::circle(combined_image, pt_ref, 3, cv::Scalar(0, 0, 255), -1);
            //         cv::line(combined_image, pt_query, pt_ref, cv::Scalar(255, 0, 0), 1);
            //     }
            //     ref_pos = image.CamFromWorld().rotation.inverse() * image.CamFromWorld().translation;
            //     ref_pos = -ref_pos;
            //     Eigen::Vector3d dist = Eigen::Vector3d(pos[0], pos[1], pos[2]) - ref_pos;
            //     string dist_str = ", Dist: " + to_string(dist.norm());
            //     cv::putText(
            //         combined_image, 
            //         "Num matches: " + to_string(query_points.size()) + dist_str, 
            //         cv::Point(10, 30), 
            //         cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
            // }
            // end debug
        }
        if (pts2d_norm_cam_i.size() < 30)
        {
            keyframe->R_w_vl = Eigen::Matrix3d::Identity();
            keyframe->T_w_vl = Eigen::Vector3d::Zero();
            return false;
        }
        std::vector<uchar> status;
        Eigen::Matrix3d R_w_vl;
        Eigen::Vector3d t_w_vl;
        // keyframe->PnPRANSAC(pts2d_norm_cam_i, pts3d_w_i, status, t_w_vl, R_w_vl);
        PnPwithIntrinsic(keyframe, pts2d_norm_cam_i, pts3d_w_i, R_w_vl, t_w_vl, status);
        if ((ref_pos - t_w_vl).norm() > 10)
        {
            status.clear();
            PnPwithIntrinsic(keyframe, pts2d_img_i, pts3d_w_i, R_w_vl, t_w_vl, status, true);
        }
        if((ref_pos - t_w_vl).norm() > 10)
        {
            keyframe->R_w_vl = Eigen::Matrix3d::Identity();
            keyframe->T_w_vl = Eigen::Vector3d::Zero();
            return false;
        }
        reduceVector(pts2d_norm_cam_i, status);
        reduceVector(pts3d_w_i, status);
        reduceVector(pts2d_img_i, status);
        reduceVector(pts3d_id, status);
        reduceVector(pts_map_error, status);

        Eigen::Matrix3d R_c2w = R_w_vl * qic;
        Eigen::Vector3d t_c2w = t_w_vl + R_w_vl * tic;
        Eigen::Matrix3d R_w2c = R_c2w.transpose();
        Eigen::Vector3d t_w2c = R_c2w.transpose() * (-t_c2w);
        vector<double> cur_errors;
        vector<double> depths;
        computeReprojectError(pts3d_w_i, pts2d_img_i, R_w2c, t_w2c, cur_errors, depths, camera);
        // std::vector<uchar> refine_status(pts2d_norm_cam_i.size(), 0);
        // begin debug
        // reduceVector(query_points, status);
        // reduceVector(ref_points, status);
        // Eigen::Matrix3d R_c2w = R_w_vl * qic;
        // Eigen::Vector3d t_c2w = t_w_vl + R_w_vl * tic;
        // Eigen::Matrix3d R_w2c = R_c2w.transpose();
        // Eigen::Vector3d t_w2c = R_c2w.transpose() * (-t_c2w);
        // vector<double> cur_errors;
        // vector<double> depths;
        // computeReprojectError(pts3d_w_i, pts2d_img_i, R_w2c, t_w2c, cur_errors, depths, camera);
        // for (size_t i = 0; i < cur_errors.size(); i++)
        // {
        //     if (cur_errors[i] < 1 && depths[i] < 30. && depths[i] > 3.0)
        //     {
        //         refine_status[i] = 1;
        //     }
        // }
        // double all_avg_depth = accumulate(depths.begin(), depths.end(), 0.0) / depths.size();
        // int small_error_cnt = 0;
        // double small_depth = 0.0;
        // for (int i = 0; i < errors.size(); i++)
        // {
        //     if(errors[i] < 1)
        //     {
        //         small_error_cnt++;
        //         small_depth += depths[i];
        //     }
        // }
        // if(small_error_cnt > 0)
        // {
        //     small_depth /= small_error_cnt;
        // }
        // vector<size_t> idxs(pts3d_w_i.size());
        // std::iota(idxs.begin(), idxs.end(), 0);
        // std::sort(idxs.begin(), idxs.end(), [&cur_errors](size_t i1, size_t i2)
        //           { return cur_errors[i1] < cur_errors[i2]; });

        // check the opencv pnp result
        // std::vector<uchar> status_opencv;
        // Eigen::Matrix3d R_w_vl_opencv;
        // Eigen::Vector3d t_w_vl_opencv;
        // PnPwithIntrinsic(pts2d_img_i, pts3d_w_i, R_w_vl_opencv, t_w_vl_opencv, status_opencv);
        // Eigen::Matrix3d R_rel = R_w_vl.transpose() * R_w_vl_opencv;
        // Eigen::Vector3d t_rel = R_w_vl.transpose() * (t_w_vl_opencv - t_w_vl);
        // Eigen::AngleAxisd rvec(R_rel);
        // double angle = rvec.angle() * 180.0 / M_PI;
        // double rel_dist = t_rel.norm();
        // Eigen::Vector3d qr_dist;
        // qr_dist = ref_pos - t_w_vl;
        // if (qr_dist.norm() > 10)
        // {
        //     ofstream debugfile(DEBUG_FILE_PATH, ios::app);
        //     debugfile.setf(ios::fixed, ios::floatfield);
        //     debugfile << srv.request.query_name << " " << ref_img_name << endl;
        //     for (size_t i = 0; i < match0.size(); i++)
        //     {
        //         if(match0[i] == -1)
        //         {
        //             continue;
        //         }
        //         if (scores[i] < 0.85)
        //         {
        //             continue;
        //         }
        //         debugfile << i << " ";
        //     }
        //     debugfile << endl;
        //     for(size_t i = 0; i < match0.size(); i++)
        //     {
        //         if (match0[i] == -1)
        //         {
        //             continue;
        //         }
        //         if (scores[i] < 0.85)
        //         {
        //             continue;
        //         }
        //         debugfile << match0[i] << " ";
        //     }
        //     debugfile << endl;
        //     debugfile.close();

        // string ref_img_path = "/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-03-24_17-36-22/sfm_map/" + ref_img_name;
        // cv::Mat ref_image = cv::imread(ref_img_path, cv::IMREAD_COLOR);
        // cv::hconcat(query_image, ref_image, combined_image);
        // double avg_error = 0;
        // double avg_depth = 0;
        // for(int i = 0; i < 5; ++i)
        // {
        //     int j = int(idxs.size()) - 1 - i;
        //     if(j < 0) break;
        //     cv::Point2f pt_query = query_points[idxs[j]];
        //     cv::Point2f pt_ref = ref_points[idxs[j]] + cv::Point2f(query_image.cols, 0);
        //     cv::circle(combined_image, pt_query, 3, cv::Scalar(0, 255, 0), -1);
        //     cv::circle(combined_image, pt_ref, 3, cv::Scalar(0, 255, 0), -1);
        //     cv::line(combined_image, pt_query, pt_ref, cv::Scalar(255, 0, 0), 1);
        // }
        // for (int i = 0; i < 5; i++)
        // {
        //     if(i >= idxs.size())
        //     {
        //         break;
        //     }
        //     avg_error *= i;
        //     avg_error += errors[idxs[i]];
        //     avg_error /= (i + 1);
        //     avg_depth *= i;
        //     avg_depth += depths[idxs[i]];
        //     avg_depth /= (i + 1);
        //     cv::Point2f pt_query = query_points[idxs[i]];
        //     cv::Point2f pt_ref = ref_points[idxs[i]] + cv::Point2f(query_image.cols, 0); 
        //     cv::circle(combined_image, pt_query, 3, cv::Scalar(0, 0, 255), -1);
        //     cv::circle(combined_image, pt_ref, 3, cv::Scalar(0, 0, 255), -1);
        //     cv::line(combined_image, pt_query, pt_ref, cv::Scalar(255, 0, 0), 1);
        // }
        // cv::putText(
        //     combined_image,
        //     "Avg error top 10: " + to_string(avg_error).substr(0, 4) + " depth: " + to_string(avg_depth).substr(0, 4),
        //     cv::Point(10, 30),
        //     cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        // vector<double> gt_info = gt_data[stol(keyframe->nanosecond)];
        // Eigen::Vector3d gt_trans(gt_info[0], gt_info[1], gt_info[2]);
        // Eigen::Vector3d align_trans(-3.02596633, 1.11718326, 0.0506536);
        // Eigen::Vector3d trans_dist = gt_trans - t_w_vl - align_trans;
        // cv::putText(
        //     combined_image,
        //     "Localization error: " + to_string(trans_dist.norm()).substr(0, 4) + " all depth: " + to_string(all_avg_depth).substr(0, 4),
        //     cv::Point(10, 60),
        //     cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        // cv::putText(
        //     combined_image,
        //     "Smaller than 1 pixel: " + to_string(small_error_cnt) + " depth: " + to_string(small_depth).substr(0, 4),
        //     cv::Point(10, 90),
        //     cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        // cv::imwrite(keyframe->save_img_path, combined_image);

        // cv::Mat filtered_img;
        // cv::hconcat(query_image, ref_image, filtered_img);
        //     int filtered_cnt = 0;
        //     for (size_t i = 0; i < query_points.size(); i++)
        //     {
        //         if (status[i] == 0)
        //         {
        //             continue;
        //         }
        //         filtered_cnt++;
        //         cv::Point2f pt_query = query_points[i];
        //         cv::Point2f pt_ref = ref_points[i] + cv::Point2f(query_image.cols, 0); // Adjust reference points' x-coordinate

        //         cv::circle(filtered_img, pt_query, 3, cv::Scalar(0, 255, 0), -1);
        //         cv::circle(filtered_img, pt_ref, 3, cv::Scalar(0, 0, 255), -1);
        //         cv::line(filtered_img, pt_query, pt_ref, cv::Scalar(255, 0, 0), 1);
        //     }
        //     cv::putText(
        //         filtered_img,
        //         "Num filtered matches: " + to_string(filtered_cnt),
        //         cv::Point(10, 30),
        //         cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

        //     cv::putText(
        //         filtered_img,
        //         "Check two(angle, trans): " + to_string(angle) + ", " + to_string(rel_dist),
        //         cv::Point(10, 60),
        //         cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

        //     cv::putText(
        //         filtered_img,
        //         "Check qr dist: " + to_string((t_w_vl_opencv - ref_pos).norm()),
        //         cv::Point(10, 90),
        //         cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

        //     string qrd = to_string(qr_dist.norm());
        //     cv::putText(
        //         combined_image,
        //         "visualLoc succ, Num inliers: " + to_string(pts2d_img_i.size()) + ", Final dist: " + qrd.substr(0, qrd.find(".") + 2), // + " GT dist: " + qgd.substr(0, qgd.find(".") + 2),
        //         cv::Point(10, 60),
        //         cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        //     cv::imwrite(keyframe->save_img_path, combined_image);
        //     cv::imwrite(keyframe->save_img_path.substr(0, keyframe->save_img_path.size() - 4) + "_filtered.png", filtered_img);
        //     return false;
        // }
        // end debug
        keyframe->R_w_vl = R_w_vl;
        keyframe->T_w_vl = t_w_vl;
        keyframe->map_pts3d_id = pts3d_id;
        keyframe->map_pts2d = pts2d_img_i;
        keyframe->map_pts2d_norm = pts2d_norm_cam_i;
        keyframe->map_pts3d = pts3d_w_i;
        keyframe->cur_errors = cur_errors;
        keyframe->depths = depths;
        vl_succ_keyframes++;
        return true;
    }
    else
    {
        ROS_ERROR("Failed to call service QueryToMatch");
        exit(1);
    }
}

cv::Point2f MapLocalization::pixel2cam(const Eigen::Vector2d &p)
{
    Eigen::Vector3d norm_p;
    m_camera->liftProjective(p, norm_p);
    return cv::Point2f(norm_p.x() / norm_p.z(), norm_p.y() / norm_p.z());
}

void MapLocalization::decision_candidate_pose(vector<float> &pos, const Eigen::Vector3d &vio_t)
{
    if(window_index < MAP_WINDOW_SIZE || !initialized)
    {
        return;
    }
    else
    {
        // float x, y, z;
        // x = float(last_t.x());
        // y = float(last_t.y());
        // z = float(last_t.z());
        // pos[0] = x;
        // pos[1] = y;
        // pos[2] = z;
        // ROS_DEBUG("start decision candidate pose");
        Eigen::Vector3d t_vlw_i = R_vio2vlw * vio_t + t_vio2vlw;
        // ROS_DEBUG("candidate pos: %f, %f, %f", t_vlw_i.x(), t_vlw_i.y(), t_vlw_i.z());
        pos[0] = float(t_vlw_i.x());
        pos[1] = float(t_vlw_i.y());
        pos[2] = float(t_vlw_i.z());
        return;
    }
}

bool MapLocalization::initialization()
{
    if(window_index < MAP_WINDOW_SIZE)
    {
        return false;
    }
    vector<Eigen::Matrix3d> Rs_vio2vlw(MAP_WINDOW_SIZE, Eigen::Matrix3d::Identity());
    vector<Eigen::Vector3d> ts_vio2vlw(MAP_WINDOW_SIZE, Eigen::Vector3d::Zero());
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        Eigen::Vector3d t_i2vlw, t_i2vio;
        Eigen::Matrix3d R_i2vlw, R_i2vio;
        optim_window[i]->getVioPose(t_i2vio, R_i2vio);
        R_i2vlw = optim_window[i]->R_w_vl;
        t_i2vlw = optim_window[i]->T_w_vl;
        Rs_vio2vlw[i] = R_i2vlw * R_i2vio.transpose();
        ts_vio2vlw[i] = t_i2vlw - Rs_vio2vlw[i] * t_i2vio;
    }
    vector<unordered_set<int>> inlier_ids(MAP_WINDOW_SIZE, unordered_set<int>());
    vector<int> inlier_cnt(MAP_WINDOW_SIZE, 0);
    int max_cnt = 0;
    int max_cnt_idx = 0;
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        for (int j = 0; j < MAP_WINDOW_SIZE; ++j)
        {
            if (i == j)
            {
                inlier_ids[i].insert(j);
                inlier_cnt[i] += 1;
                continue;
            }
            Eigen::Matrix3d res_R = Rs_vio2vlw[j].transpose() * Rs_vio2vlw[i];
            Eigen::Vector3d res_t = Rs_vio2vlw[i].transpose() * (ts_vio2vlw[i] - ts_vio2vlw[j]);
            Eigen::AngleAxisd rotation_vector(res_R);
            double angle = rotation_vector.angle() * 180. / M_PI;
            if(angle < 15. && res_t.norm() < 5)
            {
                inlier_ids[i].insert(j);
                inlier_cnt[i] += 1;
            }
        }
        if(max_cnt < inlier_cnt[i])
        {
            max_cnt = inlier_cnt[i];
            max_cnt_idx = i;
        }
    }
    ROS_DEBUG("initialized max inliers: %d", max_cnt);
    if(max_cnt < 0.5 * float(MAP_WINDOW_SIZE))
    {
        return false;
    }

    // use the max inliers to initialize the optimization window
    R_vio2vlw = Rs_vio2vlw[max_cnt_idx];
    t_vio2vlw = ts_vio2vlw[max_cnt_idx];
    // start debug
    // Eigen::Quaterniond q_vio2vlw(R_vio2vlw);
    // ROS_DEBUG("q_vio2vlw: %f, %f, %f, %f", q_vio2vlw.w(), q_vio2vlw.x(), q_vio2vlw.y(), q_vio2vlw.z());
    // ROS_DEBUG("t_vio2vlw: %f, %f, %f", t_vio2vlw.x(), t_vio2vlw.y(), t_vio2vlw.z());
    // end debug
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        if (inlier_ids[max_cnt_idx].find(i) != inlier_ids[max_cnt_idx].end())
        {
            optim_window[i]->VL_res_valid = true;
            optim_window[i]->R_optimal = optim_window[i]->R_w_vl;
            optim_window[i]->T_optimal = optim_window[i]->T_w_vl;
        }
        else
        {
            optim_window[i]->VL_res_valid = false;
            optim_window[i]->R_optimal = R_vio2vlw * optim_window[i]->vio_R_w_i;
            optim_window[i]->T_optimal = R_vio2vlw * optim_window[i]->vio_T_w_i + t_vio2vlw;
        }
    }
    return true;
}

bool MapLocalization::checkValid(KeyFrame *keyframe)
{
    if (!keyframe->VL_res_succ)
    {
        // keyframe->VL_res_valid = false;
        keyframe->R_optimal = R_vio2vlw * keyframe->vio_R_w_i;
        keyframe->T_optimal = R_vio2vlw * keyframe->vio_T_w_i + t_vio2vlw;
        return false;
    }
    // Eigen::Vector3d t_w_vli = optim_window[window_index - 1]->T_w_vl;
    // Eigen::Matrix3d R_w_vli = optim_window[window_index - 1]->R_w_vl;
    // Eigen::Vector3d t_w_vlj = optim_window[window_index - 2]->T_optimal;
    // Eigen::Matrix3d R_w_vlj = optim_window[window_index - 2]->R_optimal;

    // Eigen::Vector3d t_vio_i = optim_window[window_index - 1]->vio_T_w_i;
    // Eigen::Matrix3d R_vio_i = optim_window[window_index - 1]->vio_R_w_i;
    // Eigen::Vector3d t_vio_j = optim_window[window_index - 2]->vio_T_w_i;
    // Eigen::Matrix3d R_vio_j = optim_window[window_index - 2]->vio_R_w_i;

    // Eigen::Matrix3d R_vlj_vli = R_w_vlj.transpose() * R_w_vli;
    // Eigen::Vector3d t_vlj_vli = R_w_vlj.transpose() * (t_w_vli - t_w_vlj);

    // Eigen::Matrix3d R_oj_oi = R_vio_j.transpose() * R_vio_i;
    // Eigen::Vector3d t_oj_oi = R_vio_j.transpose() * (t_vio_i - t_vio_j);

    // double res_angle = Eigen::AngleAxisd(R_oj_oi.transpose() * R_vlj_vli).angle() * 180. / M_PI;
    // double res_t = (t_oj_oi - t_vlj_vli).norm();

    // ROS_DEBUG("res_angle: %f, res_t: %f", res_angle, res_t);

    // if(res_angle < 15. && res_t < 5)
    if(keyframe->VL_res_succ)
    {
        // keyframe->VL_res_valid = true;
        keyframe->R_optimal = keyframe->R_w_vl;
        keyframe->T_optimal = keyframe->T_w_vl;
        return true;
    }
    else
    {
        // keyframe->VL_res_valid = false;
        keyframe->R_optimal = R_vio2vlw * keyframe->vio_R_w_i;
        keyframe->T_optimal = R_vio2vlw * keyframe->vio_T_w_i + t_vio2vlw;
        return false;
    }
}

void MapLocalization::loadParameters()
{
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        Eigen::Quaterniond tmp_q(optim_window[i]->R_optimal);
        param_window_qs[i][0] = tmp_q.w();
        param_window_qs[i][1] = tmp_q.x();
        param_window_qs[i][2] = tmp_q.y();
        param_window_qs[i][3] = tmp_q.z();
        param_window_ts[i][0] = optim_window[i]->T_optimal.x();
        param_window_ts[i][1] = optim_window[i]->T_optimal.y();
        param_window_ts[i][2] = optim_window[i]->T_optimal.z();
    }
}

void MapLocalization::computePoseLoss()
{
    Eigen::Vector3d r_loss, t_loss;
    r_loss = Eigen::Vector3d::Zero();
    t_loss = Eigen::Vector3d::Zero();
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {

        for (int bias = -1; bias < 2; ++bias)
        {
            int j = i + bias;
            if (j == i || j < 0 || j >= MAP_WINDOW_SIZE)
                continue;
            Eigen::Vector3d t_vio_j, t_vio_i;
            Eigen::Matrix3d R_vio_j, R_vio_i;
            optim_window[i]->getVioPose(t_vio_i, R_vio_i);
            optim_window[j]->getVioPose(t_vio_j, R_vio_j);
            Eigen::Quaterniond relative_q(R_vio_i.transpose() * R_vio_j);
            Eigen::Vector3d relative_t(R_vio_i.transpose() * (t_vio_j - t_vio_i));

            Eigen::Quaterniond relative_q_optimal(optim_window[i]->R_optimal.transpose() * optim_window[j]->R_optimal);
            Eigen::Vector3d relative_t_optimal(optim_window[i]->R_optimal.transpose() * (optim_window[j]->T_optimal - optim_window[i]->T_optimal));

            Eigen::Quaterniond diffq(relative_q.inverse() * relative_q_optimal);
            Eigen::Vector3d difft = relative_t - relative_t_optimal;

            r_loss.x() += diffq.x() * 2;
            r_loss.y() += diffq.y() * 2;
            r_loss.z() += diffq.z() * 2;
            t_loss += difft * 2;
        }
    }
    ROS_DEBUG("r_loss: %f, %f, %f; t_loss: %f, %f, %f", r_loss.x(), r_loss.y(), r_loss.z(), t_loss.x(), t_loss.y(), t_loss.z());
}

void MapLocalization::computePointLoss()
{
    double loss = 0;
    int cnt = 0;
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        Eigen::Matrix3d R_w2i = optim_window[i]->R_optimal.transpose();
        Eigen::Vector3d t_w2i = R_w2i * (-optim_window[i]->T_optimal);
        for (int j = 0; j < optim_window[i]->map_pts3d_id.size(); ++j)
        {
            Eigen::Vector3d pt3d = Eigen::Vector3d(
                optim_window[i]->map_pts3d[j].x, 
                optim_window[i]->map_pts3d[j].y,
                optim_window[i]->map_pts3d[j].z);
            Eigen::Vector2d pt2d_norm = Eigen::Vector2d(
                optim_window[i]->map_pts2d_norm[j].x,
                optim_window[i]->map_pts2d_norm[j].y);
            Eigen::Vector3d reproj_pts = qic.transpose() * (R_w2i * pt3d + t_w2i - tic);
            reproj_pts = reproj_pts / reproj_pts.z();
            Eigen::Vector2d reproj_pts_norm = Eigen::Vector2d(reproj_pts.x(), reproj_pts.y());
            loss += (reproj_pts_norm - pt2d_norm).norm();
            cnt += 1;
        }
    }
    loss = loss / cnt;
    ROS_DEBUG("point loss: %f", loss);
}

void MapLocalization::pointStatistic()
{
    unordered_map<int, int> statistic;
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        for (int j = 0; j < optim_window[i]->map_pts3d_id.size(); ++j)
        {
            if (statistic.find(optim_window[i]->map_pts3d_id[j]) == statistic.end())
            {
                statistic[optim_window[i]->map_pts3d_id[j]] = 1;
            }
            else
            {
                statistic[optim_window[i]->map_pts3d_id[j]] += 1;
            }
        }
    }

    vector<int> frame_cnt(20, 0);
    for (auto it = statistic.begin(); it != statistic.end(); ++it)
    {
        frame_cnt[it->second - 1]++;
    }

    string report_str;
    for (int i = 0; i < 20; ++i)
    {
        report_str += to_string(frame_cnt[i]) + " ";
    }
    ROS_DEBUG_STREAM(report_str);
}

void MapLocalization::tightlyCoupledOptimization()
{
    loadParameters();
    ceres::Problem problem;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // ptions.minimizer_progress_to_stdout = true;
    // options.max_solver_time_in_seconds = 0.08;
    options.max_num_iterations = 8;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(0.1);
    // loss_function = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

    std::unordered_map<int, int> id2idx;
    vector<vector<double>> pts3d;

    // unordered_map<int, int> statistic;

    // ROS_INFO("start pose optimization");
    // computePoseLoss();
    // computePointLoss();
    // pointStatistic();

    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        
        problem.AddParameterBlock(param_window_qs[i], 4, local_parameterization);
        problem.AddParameterBlock(param_window_ts[i], 3);

        // reference edges
        if(optim_window[i]->VL_res_valid)
        {
            ceres::CostFunction *pose_priori = ReferenceError::Create(
                optim_window[i]->R_w_vl, optim_window[i]->T_w_vl, 0.1, 0.01);
            problem.AddResidualBlock(
                pose_priori, loss_function,
                param_window_qs[i], param_window_ts[i]);
        }
        // adjacent edges
        for (int bias = -1; bias < 2; ++bias)
        {
            int j = i + bias;
            if(j == i || j < 0 || j >= MAP_WINDOW_SIZE) continue;
            Eigen::Vector3d t_vio_j, t_vio_i;
            Eigen::Matrix3d R_vio_j, R_vio_i;
            optim_window[i]->getVioPose(t_vio_i, R_vio_i);
            optim_window[j]->getVioPose(t_vio_j, R_vio_j);
            Eigen::Quaterniond relative_q(R_vio_i.transpose() * R_vio_j);
            Eigen::Vector3d relative_t(R_vio_i.transpose() * (t_vio_j - t_vio_i));

            ceres::CostFunction *adjacent_edge = RelativeRTError::Create(
                relative_t.x(), relative_t.y(), relative_t.z(),
                relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                0.1, 0.01);
            problem.AddResidualBlock(
                adjacent_edge, loss_function,
                param_window_qs[i], param_window_ts[i],
                param_window_qs[j], param_window_ts[j]);
        }

        // sort errors
        vector<double> reproj_errors = optim_window[i]->cur_errors;
        vector<size_t> idxs(reproj_errors.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(), [&reproj_errors](size_t i1, size_t i2)
                  { return reproj_errors[i1] < reproj_errors[i2]; });
        for (int j = 0; j < idxs.size(); ++j)
        {
            if(reproj_errors[idxs[j]] > 1 || j >= 10)
            {
                break;
            }
            if(optim_window[i]->depths[idxs[j]] < 3.0 || optim_window[i]->depths[idxs[j]] > 30.0)
            {
                continue;
            }
            if(id2idx.find(optim_window[i]->map_pts3d_id[idxs[j]]) != id2idx.end())
            {
                continue;
            }
            id2idx[optim_window[i]->map_pts3d_id[j]] = int(pts3d.size());
            pts3d.emplace_back(vector<double>{optim_window[i]->map_pts3d[j].x, optim_window[i]->map_pts3d[j].y, optim_window[i]->map_pts3d[j].z});
        }

        // for (int j = 0; j < optim_window[i]->map_pts3d_id.size(); ++j)
        // {
        //     if (optim_window[i]->refine_status[j] == 0 || id2idx.find(optim_window[i]->map_pts3d_id[j]) != id2idx.end())
        //     {
        //         continue;
        //     }
        //     id2idx[optim_window[i]->map_pts3d_id[j]] = int(pts3d.size());
        //     pts3d.emplace_back(vector<double>{optim_window[i]->map_pts3d[j].x, optim_window[i]->map_pts3d[j].y, optim_window[i]->map_pts3d[j].z});

        //     // if(statistic.find(optim_window[i]->map_pts3d_id[j]) == statistic.end())
        //     // {
        //     //     statistic[optim_window[i]->map_pts3d_id[j]] = 1;
        //     // }
        //     // else
        //     // {
        //     //     statistic[optim_window[i]->map_pts3d_id[j]] += 1;
        //     // }
        // }
    }
    // problem.SetParameterBlockConstant(param_window_qs[0]);
    // problem.SetParameterBlockConstant(param_window_ts[0]);
    ROS_DEBUG("num points: %d", pts3d.size());

    pointsCLoud = pts3d;
    int num_points = pts3d.size();
    ROS_DEBUG("num points: %d", num_points);
    double param_pts3d[num_points][3];
    vector<bool> put_num(num_points, false);
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        for (int j = 0; j < optim_window[i]->map_pts3d_id.size(); ++j)
        {
            if (id2idx.find(optim_window[i]->map_pts3d_id[j]) == id2idx.end())
                continue;
            int idx = id2idx[optim_window[i]->map_pts3d_id[j]];
            if (!put_num[idx])
            {
                param_pts3d[idx][0] = pts3d[idx][0];
                param_pts3d[idx][1] = pts3d[idx][1];
                param_pts3d[idx][2] = pts3d[idx][2];
                put_num[idx] = true;
                // problem.AddParameterBlock(param_pts3d[idx], 3);
                // problem.SetParameterBlockConstant(param_pts3d[idx]);
                ceres::CostFunction *point_priori = PointPriori::Create(
                    pts3d[idx][0], pts3d[idx][1], pts3d[idx][2]);
                problem.AddResidualBlock(
                    point_priori, loss_function,
                    param_pts3d[idx]);
            }
            ceres::CostFunction *observation_egde = ReprojectionError::Create(
                optim_window[i]->map_pts2d_norm[j].x, optim_window[i]->map_pts2d_norm[j].y, qic, tic);
            problem.AddResidualBlock(
                observation_egde, loss_function,
                param_window_qs[i], param_window_ts[i],
                param_pts3d[idx]);
        }
    }
    ceres::Solve(options, &problem, &summary);
    ROS_DEBUG("pose optimization time: %f", summary.total_time_in_seconds);
    // post processing
    updateParameters();

    // ROS_INFO("pose optimization finish");
    // computePoseLoss();
    // computePointLoss();
}

void MapLocalization::updateParameters()
{
    vector<Matrix3d> Rs_vio2vlw(MAP_WINDOW_SIZE, Matrix3d::Identity());
    // for all keyframes, use optimized parameters to update the pose
    // re-estimate the vio2vlw transformation
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        Eigen::Quaterniond tmp_q(param_window_qs[i][0], param_window_qs[i][1], param_window_qs[i][2], param_window_qs[i][3]);
        Eigen::Vector3d tmp_t(param_window_ts[i][0], param_window_ts[i][1], param_window_ts[i][2]);
        optim_window[i]->R_optimal = tmp_q.toRotationMatrix();
        optim_window[i]->T_optimal = tmp_t;
        Rs_vio2vlw[i] = optim_window[i]->R_optimal * optim_window[i]->vio_R_w_i.transpose();
    }
    Eigen::Matrix3d R_vio2vlw_avg;
    RotationAvg(Rs_vio2vlw, R_vio2vlw_avg);
    Eigen::AngleAxisd qdiff(R_vio2vlw.transpose() * R_vio2vlw_avg);
    R_vio2vlw = R_vio2vlw_avg;

    Eigen::Vector3d t_vio2vlw_avg = Eigen::Vector3d::Zero();
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        t_vio2vlw_avg += optim_window[i]->T_optimal - R_vio2vlw * optim_window[i]->vio_T_w_i;
    }
    t_vio2vlw_avg /= static_cast<double>(MAP_WINDOW_SIZE);
    Eigen::Vector3d tdiff = t_vio2vlw_avg - t_vio2vlw;
    ROS_DEBUG("update vio2vlw angle, trans: %f, %f", qdiff.angle() * 180. / M_PI, t_vio2vlw_avg.norm());
    t_vio2vlw = t_vio2vlw_avg;
}

void MapLocalization::slideWindow()
{
    for (int i = 0; i < MAP_WINDOW_SIZE - 1; ++i)
    {
        optim_window[i] = optim_window[i + 1];
    }
    optim_window[MAP_WINDOW_SIZE - 1] = nullptr;
}

void MapLocalization::readIntrinsicParameter(const string &calib_file)
{
    camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    ROS_INFO("read camera intrinsic parameter finish\n");
}

void MapLocalization::publish()
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(optim_window[0]->time_stamp);
    publish_path(header);
    publish_point_cloud(header);
    // publish_TransedVIO(header);
    publish_visualLoc(header);
    count_vl();
}

void MapLocalization::count_vl()
{
    int cnt = 0;
    for (int i = 0; i < MAP_WINDOW_SIZE; ++i)
    {
        if(optim_window[i]->VL_res_succ)
        {
            cnt++;
        }
    }
    ROS_DEBUG("vl succ: %d", cnt);
}

// void MapLocalization::read_gt()
// {
//     string filename = "/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-04-07_10-20-32/gt_tum_ns.txt";
//     std::ifstream file(filename);

//     if (!file.is_open())
//     {
//         std::cerr << "Failed to open file: " << filename << std::endl;
//         exit(1);
//     }

//     std::string line;
//     while (std::getline(file, line))
//     {
//         std::istringstream iss(line);
//         long key;
//         double tx, ty, tz, qx, qy, qz, qw;

//         if (!(iss >> key >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
//         {
//             std::cerr << "Error parsing line: " << line << std::endl;
//             continue;
//         }

//         gt_data[key] = {tx, ty, tz, qx, qy, qz, qw};
//     }

//     file.close();
// }

// void MapLocalization::publish_validLoc(const std_msgs::Header &header)
// {
    
// }

void MapLocalization::publish_visualLoc(const std_msgs::Header &header)
{
    if (optim_window[0]->VL_res_succ)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        Eigen::Vector3d t_w_vl = optim_window[0]->T_w_vl;
        Eigen::Quaterniond Q_w_vl = Eigen::Quaterniond(optim_window[0]->R_w_vl);
        pose_stamped.pose.position.x = t_w_vl.x();
        pose_stamped.pose.position.y = t_w_vl.y();
        pose_stamped.pose.position.z = t_w_vl.z();
        pose_stamped.pose.orientation.x = Q_w_vl.x();
        pose_stamped.pose.orientation.y = Q_w_vl.y();
        pose_stamped.pose.orientation.z = Q_w_vl.z();
        pose_stamped.pose.orientation.w = Q_w_vl.w();
        visualLocPath.header = header;
        visualLocPath.header.frame_id = "world";
        visualLocPath.poses.push_back(pose_stamped);
        pub_visual_loc.publish(visualLocPath);

        // ofstream visualLoc_path_file(VISUAL_LOC_PATH, ios::app);
        // visualLoc_path_file.setf(ios::fixed, ios::floatfield);
        // visualLoc_path_file.precision(9);
        // visualLoc_path_file << header.stamp.toSec() << " ";
        // visualLoc_path_file.precision(6);
        // visualLoc_path_file << t_w_vl.x() << " "
        //                << t_w_vl.y() << " "
        //                << t_w_vl.z() << " "
        //                << Q_w_vl.x() << " "
        //                << Q_w_vl.y() << " "
        //                << Q_w_vl.z() << " "
        //                << Q_w_vl.w() << endl;
        // visualLoc_path_file.close();

        // if(optim_window[MAP_WINDOW_SIZE - 1]->VL_res_valid)
        // {
        //     ofstream visualLoc_valid_path_file(VALID_LOC_PATH, ios::app);
        //     visualLoc_valid_path_file.setf(ios::fixed, ios::floatfield);
        //     visualLoc_valid_path_file.precision(9);
        //     visualLoc_valid_path_file << header.stamp.toSec() << " ";
        //     visualLoc_valid_path_file.precision(6);
        //     visualLoc_valid_path_file << t_w_vl.x() << " "
        //                    << t_w_vl.y() << " "
        //                    << t_w_vl.z() << " "
        //                    << Q_w_vl.x() << " "
        //                    << Q_w_vl.y() << " "
        //                    << Q_w_vl.z() << " "
        //                    << Q_w_vl.w() << endl;
        //     visualLoc_valid_path_file.close();
        // }
    }
    // geometry_msgs::PoseStamped pose_stamped;
    // pose_stamped.header = header;
    // pose_stamped.header.frame_id = "world";
    // vector<double> pose = gt_data[stol(optim_window[MAP_WINDOW_SIZE - 1]->nanosecond)];
    // Eigen::Vector3d t_w_vl(pose[0], pose[1], pose[2]);
    // Eigen::Quaterniond Q_w_vl = Eigen::Quaterniond(pose[6], pose[3], pose[4], pose[5]);
    // pose_stamped.pose.position.x = t_w_vl.x();
    // pose_stamped.pose.position.y = t_w_vl.y();
    // pose_stamped.pose.position.z = t_w_vl.z();
    // pose_stamped.pose.orientation.x = Q_w_vl.x();
    // pose_stamped.pose.orientation.y = Q_w_vl.y();
    // pose_stamped.pose.orientation.z = Q_w_vl.z();
    // pose_stamped.pose.orientation.w = Q_w_vl.w();
    // gtPath.header = header;
    // gtPath.header.frame_id = "world";
    // gtPath.poses.push_back(pose_stamped);
    // pub_gt.publish(gtPath);
}

void MapLocalization::publish_TransedVIO(const std_msgs::Header &header)
{
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    Eigen::Quaterniond Q_w_vl = Eigen::Quaterniond(R_vio2vlw * optim_window[0]->vio_R_w_i);
    Eigen::Vector3d t_w_vl = R_vio2vlw * optim_window[0]->vio_T_w_i + t_vio2vlw;
    pose_stamped.pose.position.x = t_w_vl.x();
    pose_stamped.pose.position.y = t_w_vl.y();
    pose_stamped.pose.position.z = t_w_vl.z();
    pose_stamped.pose.orientation.x = Q_w_vl.x();
    pose_stamped.pose.orientation.y = Q_w_vl.y();
    pose_stamped.pose.orientation.z = Q_w_vl.z();
    pose_stamped.pose.orientation.w = Q_w_vl.w();
    transedPath.header = header;
    transedPath.header.frame_id = "world";
    transedPath.poses.push_back(pose_stamped);
    pub_transed_vio.publish(transedPath);
}

void MapLocalization::publish_path(const std_msgs::Header &header)
{

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    Quaterniond tmp_Q;
    tmp_Q = Quaterniond(optim_window[0]->R_optimal);
    odometry.pose.pose.position.x = optim_window[0]->T_optimal.x();
    odometry.pose.pose.position.y = optim_window[0]->T_optimal.y();
    odometry.pose.pose.position.z = optim_window[0]->T_optimal.z();
    odometry.pose.pose.orientation.x = tmp_Q.x();
    odometry.pose.pose.orientation.y = tmp_Q.y();
    odometry.pose.pose.orientation.z = tmp_Q.z();
    odometry.pose.pose.orientation.w = tmp_Q.w();

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    path.header = header;
    path.header.frame_id = "world";
    path.poses.push_back(pose_stamped);
    pub_path.publish(path);

    ofstream map_localization_file(MAP_LOCALIZATION_PATH, ios::app);
    map_localization_file.setf(ios::fixed, ios::floatfield);
    map_localization_file.precision(9);
    map_localization_file << header.stamp.toSec() << " ";
    map_localization_file.precision(6);
    map_localization_file << optim_window[0]->T_optimal.x() << " "
                          << optim_window[0]->T_optimal.y() << " "
                          << optim_window[0]->T_optimal.z() << " "
                          << tmp_Q.x() << " "
                          << tmp_Q.y() << " "
                          << tmp_Q.z() << " "
                          << tmp_Q.w() << endl;
    map_localization_file.close();

    // visualization_msgs::Marker arrow_marker;
    // Eigen::Matrix3d R_c2w = optim_window[0]->R_optimal * qic;
    // Eigen::Vector3d direct = R_c2w * Eigen::Vector3d(0, 0, 1);
    // direct = direct / direct.norm();
    // arrow_marker.header = header;
    // arrow_marker.header.frame_id = "world";
    // arrow_marker.ns = "camera_direction";
    // arrow_marker.id = path.poses.size(); // 使用路径点的数量作为箭头的 ID
    // arrow_marker.type = visualization_msgs::Marker::ARROW;
    // arrow_marker.action = visualization_msgs::Marker::ADD;
    // arrow_marker.pose.position.x = 0;
    // arrow_marker.pose.position.y = 0;
    // arrow_marker.pose.position.z = 0;
    // arrow_marker.pose.orientation.x = 0;
    // arrow_marker.pose.orientation.y = 0;
    // arrow_marker.pose.orientation.z = 0;
    // arrow_marker.pose.orientation.w = 1;
    // arrow_marker.scale.x = 0.1;  // 箭头的长度
    // arrow_marker.scale.y = 0.02; // 箭头的宽度
    // arrow_marker.scale.z = 0.02; // 箭头的高度
    // arrow_marker.color.a = 1.0;  // 箭头的透明度
    // arrow_marker.color.r = 1.0;  // 箭头的颜色（红色）
    // arrow_marker.color.g = 0.0;
    // arrow_marker.color.b = 0.0;

    // // 设置箭头的方向
    // geometry_msgs::Point start_point, end_point;
    // start_point.x = optim_window[0]->T_optimal.x();
    // start_point.y = optim_window[0]->T_optimal.y();
    // start_point.z = optim_window[0]->T_optimal.z();
    // end_point.x = optim_window[0]->T_optimal.x() + direct.x();
    // end_point.y = optim_window[0]->T_optimal.y() + direct.y();
    // end_point.z = optim_window[0]->T_optimal.z() + direct.z();
    // arrow_marker.points.push_back(start_point);
    // arrow_marker.points.push_back(end_point);
    // markerArray_msg.markers.push_back(arrow_marker);
    // pub_camera_direct.publish(markerArray_msg);
}

void MapLocalization::publish_point_cloud(const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;

    for (int i = 0; i < pointsCLoud.size(); ++i)
    {
        geometry_msgs::Point32 p;
        p.x = pointsCLoud[i][0];
        p.y = pointsCLoud[i][1];
        p.z = pointsCLoud[i][2];
        point_cloud.points.push_back(p);
    }
    
    pub_point_cloud.publish(point_cloud);
}