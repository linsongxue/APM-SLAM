# APM-SLAM
This repository contains the code for our research paper titled "APM-SLAM: Visual Localization for Fixed Routes with Tightly Coupled A Priori Map".

This work is based on
1. [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
2. [hloc - the hierarchical localization toolbox](https://github.com/cvg/Hierarchical-Localization)
3. [COLMAP with GPS Position Prior](https://github.com/Vincentqyw/colmap-gps)

Thanks to these open-source codes and their authers.

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 20.04.
ROS Noetic. [ROS Installation](https://wiki.ros.org/noetic/Installation)

### 1.2 **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).

### 1.3 **COLMAP**
Follow [COLMAP Installation](https://colmap.github.io/install.html)

### 1.4 **hloc**
Follow [hloc Installation](https://github.com/cvg/Hierarchical-Localization)


## 2. Build MAP-SLAM
Clone the repository, modify CMakeLists.txt and catkin_make:
```
    cd ~/catkin_ws/src
    git clone https://github.com/linsongxue/APM-SLAM.git
    cd ../
    catkin_make
    source ~/catkin_ws/devel/setup.bash
```

You need to modify the CMakeLists.txt of every package to reset **"Ceres_DIR"**, **"colmap_DIR"** and so on.

## 3. 4Seasons Example

### 3.0 Prepare data
Download [Formatted 4Seasons map metadata, rosbag (extraction code: m58y)](https://pan.baidu.com/s/14P4JZnir0PIWzPho37wveQ?pwd=m58y) to YOUR_DATASET_FOLDER. Download the [undistorted images](https://cvg.cit.tum.de/data/datasets/4seasons-dataset/download) to YOUR_DATASET_FOLDER. The dataset format should look like
```
YOUR_DATASET_FOLDER
├── recording_2020-03-24_17-45-31
|   ├── recording_2020-03-24_17-45-31.bag 
|   └── undistorted_images
└── map_from_gnss
    ├── feats-superpoint-n4096-rmax1600.h5
    ├── feats-superpoint-xx-xx.h5
    ├── global-feats-netvlad.h5
    ├── pairs-db-dist20-retrieval.txt
    └── sfm_superpoint+superglue
        ├── cameras.bin
        ├── database.db
        ├── images.bin
        ├── points3D.bin
        └── project.ini

```

You need firstly modify some config settings for
load map, take *config/4seasons/4seansons_mono_imu_config.yaml*, you need set
**sfm_map_path** to **YOUR_DATASET_FOLDER/map_from_gnss/sfm_superpoint+superglue**, where store COLMAP **.bin** files.

### 3.1 Monocualr camera + IMU

```
    roslaunch vins vins_rviz.launch
    rosrun loop_fusion server.py --dataset YOUR_DATASET_FOLDER/recording_2020-03-24_17-45-31 --outputs YOUR_DATASET_FOLDER/map_from_gnss
    rosrun loop_fusion loop_fusion_node ./src/VINS-Fusion/config/4seasons/4seansons_mono_imu_config.yaml
    rosrun vins vins_node ./src/VINS-Fusion/config/4seasons/4seansons_mono_imu_config.yaml 
    rosbag play YOUR_DATASET_FOLDER/recording_2020-03-24_17-45-31/recording_2020-03-24_17-45-31.bag
```

### 3.2 Stereo cameras + IMU

```
    roslaunch vins vins_rviz.launch
    rosrun loop_fusion server.py --dataset YOUR_DATASET_FOLDER/recording_2020-03-24_17-45-31 --outputs YOUR_DATASET_FOLDER/map_from_gnss
    rosrun loop_fusion loop_fusion_node ./src/VINS-Fusion/config/4seasons/4seansons_stereo_imu_config.yaml
    rosrun vins vins_node ./src/VINS-Fusion/config/4seasons/4seansons_stereo_imu_config.yaml 
    rosbag play YOUR_DATASET_FOLDER/recording_2020-03-24_17-45-31/recording_2020-03-24_17-45-31.bag
```

### 3.3 Stereo cameras

```
    roslaunch vins vins_rviz.launch
    rosrun loop_fusion server.py --dataset YOUR_DATASET_FOLDER/recording_2020-03-24_17-45-31 --outputs YOUR_DATASET_FOLDER/map_from_gnss
    rosrun loop_fusion loop_fusion_node ./src/VINS-Fusion/config/4seasons/4seansons_stereo_config.yaml
    rosrun vins vins_node ./src/VINS-Fusion/config/4seasons/4seansons_stereo_config.yaml 
    rosbag play YOUR_DATASET_FOLDER/recording_2020-03-24_17-45-31/recording_2020-03-24_17-45-31.bag
```

## 4. Data format
We will release the mapping code later...

## 5. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
