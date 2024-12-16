import rospy
from loop_fusion.srv import QueryToMatch, QueryToMatchResponse
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R
import h5py
import torch
import os
from hloc.utils.base_model import dynamic_load
from hloc import matchers, match_features, extract_features
from pathlib import Path
import argparse
from time import time
from typing import List, Dict
import cv2

def get_descriptors(names, path, name2idx=None, key="global_descriptor"):
    if name2idx is None:
        with h5py.File(str(path), "r", libver="latest") as fd:
            desc = [fd[n][key].__array__() for n in names]
    else:
        desc = []
        for n in names:
            with h5py.File(str(path[name2idx[n]]), "r", libver="latest") as fd:
                desc.append(fd[n][key].__array__())
    return torch.from_numpy(np.stack(desc, 0)).float()

def collate_fn(datas:List[Dict[str, torch.Tensor]], device:torch.device):
    batched = {}
    for k in datas[0].keys():
        data = []
        for data_item in datas:
            if k == "keypoints1":
                padded = torch.zeros((1, 2000, 2), dtype=torch.float32)
                padded[:, :data_item[k].shape[1]] = data_item[k]
            elif k == "scores1":
                padded = torch.zeros((1, 2000), dtype=torch.float32)
                padded[:, :data_item[k].shape[1]] = data_item[k]
            elif k == "descriptors1":
                padded = torch.zeros((1, 256, 2000), dtype=torch.float32)
                padded[:, :, :data_item[k].shape[2]] = data_item[k]
            else:
                padded = data_item[k]
            data.append(padded)
        batched[k] = torch.cat(data, 0) if k.startswith("image") else torch.cat(data, 0).to(device, non_blocking=True)
    return batched

class MapServer(object):
    def __init__(self, output_dir:Path, sequence_path:Path):
        
        rospy.logdebug(f"Output directory: {output_dir}")
        rospy.logdebug(f"Sequence path: {sequence_path}")
        self.output_dir = output_dir
        self.sequence_path = sequence_path
        self.match_config = match_features.confs["superglue"]
        self.coarse_config = extract_features.confs["netvlad"]
        self.fine_config = extract_features.confs["superpoint_max"]
        coarse_fpath = Path(output_dir, self.coarse_config["output"] + ".h5")
        fine_fpath = Path(output_dir, self.fine_config["output"] + ".h5")
        colmap_path = output_dir / "sfm_superpoint+superglue"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # prepare colmap data
        self.map_frame_names2id, self.map_frame_names, self.map_frame_positions = self.prepare_from_colmap(colmap_path)
        # prepare h5 data
        query_images = sorted(os.listdir(os.path.join(str(sequence_path), "undistorted_images", "cam0")))
        query_names = [f"cam0/{image}" for image in query_images]
        self.names2scores = self.prepare_coarse_feature(coarse_fpath, query_names, self.map_frame_names)
        # prepare model
        self.model = self.prepare_model()
        
        self.feature_points = str(fine_fpath)
        
        self.service = rospy.Service('/match_info', QueryToMatch, self.handle_query_to_match)
        rospy.loginfo("Ready to receive query")
    
    def prepare_from_colmap(self, colmap_path):
        rospy.loginfo(f"Loading colmap data from {colmap_path}")
        reconstruction = pycolmap.Reconstruction(colmap_path)
        self.reconstruction = reconstruction
        names = []
        positions = []
        names2id = {}
        rospy.loginfo(f"Loading colmap data from with {len(reconstruction.images)} images")
        for i in range(len(reconstruction.images)):
            image = reconstruction.images[i]  # gnss changed
            t_c2w = -R.from_quat(image.cam_from_world.rotation.quat).inv().apply(image.cam_from_world.translation)
            names2id[image.name] = i
            names.append(image.name)
            positions.append(t_c2w)
        positions = np.array(positions)
        return names2id, names, positions
    
    def prepare_coarse_feature(self, feature_file, queries, refences):
        rospy.loginfo(f"Loading Coarse features from {feature_file}")
        rospy.logdebug(f"Query example: {queries[0]}")
        rospy.logdebug(f"Reference example: {refences[0]}")
        ref_desc = get_descriptors(refences, feature_file).to(self.device)
        query_desc = get_descriptors(queries, feature_file).to(self.device)
        scores = torch.einsum("id,jd->ij", query_desc, ref_desc).cpu()
        names2scores = {queries[i]: scores[i] for i in range(len(queries))}
        return names2scores
    
    def prepare_model(self):
        rospy.loginfo(f"Loading model {self.match_config['model']['name']}")
        Model = dynamic_load(matchers, self.match_config["model"]["name"])
        model = Model(self.match_config["model"]).eval().to(self.device)
        return model
    
    def get_pairs_point_features(self, image0, image1):
        data = {}
        with h5py.File(self.feature_points, "r") as f:
            for k, v in f[image0].items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float().unsqueeze(0)
            data["image0"] = torch.empty((1, 1,) + tuple(f[image0]["image_size"])[::-1])
            
            for k, v in f[image1].items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float().unsqueeze(0)
            data["image1"] = torch.empty((1, 1,) + tuple(f[image1]["image_size"])[::-1])
        return data
    
    def draw_matches(self, image0, image1, points0, points1, matches, scores=None, dist = 0):
        time0 = os.path.basename(image0).split(".")[0]
        # time1 = image1.split('/')[0] + "_" + os.path.basename(image1).split(".")[0]
        query_image = cv2.imread(str(self.sequence_path / "undistorted_images" / image0), cv2.IMREAD_GRAYSCALE)
        ref_image = cv2.imread(os.path.join(
            "/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-03-24_17-36-22/sfm_map",
            image1), cv2.IMREAD_GRAYSCALE)
        # print(query_image.shape, ref_image.shape)
        height = max(ref_image.shape[0], query_image.shape[0])
        width = ref_image.shape[1] + query_image.shape[1]
        output_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        output_image[:ref_image.shape[0], :query_image.shape[1]] = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
        output_image[:query_image.shape[0], query_image.shape[1]:] = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
        
        # 将点的坐标转换为整数
        points0 = np.round(points0).astype(int)
        points1 = np.round(points1).astype(int)

        # 绘制匹配点和连线
        cnt = 0
        ref_id = self.map_frame_names2id[image1]
        for i, match_idx in enumerate(matches):
            if match_idx == -1:
                continue
            if scores is not None and scores[i] < 0.5:
                continue
            if not self.reconstruction.images[ref_id].points2D[match_idx].has_point3D():
                continue
            
            # points1 在 query_image 中
            pt1 = (points0[i][0], points0[i][1])  # 偏移参考图像的宽度
            # points2 在 ref_image 中
            pt2 = (points1[match_idx][0] + query_image.shape[1], points1[match_idx][1])
            
            # 绘制点
            cv2.circle(output_image, pt1, 5, (0, 255, 0), -1)  # query_image 点为绿色
            cv2.circle(output_image, pt2, 5, (255, 0, 0), -1)  # ref_image 点为蓝色
            
            # 绘制连线
            cv2.line(output_image, pt1, pt2, (0, 255, 255), 1)  # 黄色线连接
            cnt += 1
        cv2.putText(output_image, f"Num matches: {cnt}, dist: {dist}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(f"/home/setsu/workspace/catkin_ws/addition/debug/server/{time0}.png", output_image)
    
    @torch.no_grad()
    def handle_query_to_match(self, req:QueryToMatch):
        rospy.loginfo(f"Received query {req.query_name}")
        start_time = time()
        query_name = req.query_name
        candidate_position = np.array(req.candidate_pos).reshape((1, 3))
        query_scores = self.names2scores[query_name]
        dist = np.linalg.norm(self.map_frame_positions - candidate_position, axis=1)
        invalid_mask = torch.from_numpy(dist > 50.)
        rospy.logdebug(f"Num candidates: {len(invalid_mask) - torch.sum(invalid_mask)}")
        query_scores.masked_fill_(invalid_mask, float("-inf"))
        top_indices = torch.argmax(query_scores).numpy()
        rospy.logdebug(f"Top indices: {top_indices}")
        rospy.logdebug(f"Top scores: {query_scores[top_indices].numpy()}")
        incremental_num = 0
        start_idx = max(0, top_indices - incremental_num)
        end_idx = min(len(self.map_frame_names) - 1, top_indices + incremental_num)
        output_scores = []
        output_matches = []
        output_ids = []
        output_pts2d = None
        # datas = []
        for i, ref_id in enumerate(range(start_idx, end_idx + 1)):
            ref_name = self.map_frame_names[ref_id]
            data = self.get_pairs_point_features(query_name, ref_name)
            # kpts0 = data["keypoints0"].cpu().squeeze().float().numpy()
            # kpts1 = data["keypoints1"].cpu().squeeze().float().numpy()
            if output_pts2d is None:
                output_pts2d = data["keypoints0"].cpu().squeeze().float().numpy().reshape(-1)
        
            data = {
                k: v if k.startswith("image") else v.to(self.device, non_blocking=True)
                for k, v in data.items()
            }
            pred = self.model(data)
            matches = pred["matches0"].cpu().squeeze().int().numpy()
            scores = pred["matching_scores0"].cpu().squeeze().float().numpy()
            # print(query_name, ref_name)
            # if i == 0:
            #     ref_pos = self.map_frame_positions[ref_id]
            #     pair_dist = np.linalg.norm(candidate_position - ref_pos)
            #     self.draw_matches(query_name, ref_name, kpts0, kpts1, matches, scores, pair_dist)
            output_ids.append(ref_id)  # gnss changed
            output_matches.append(matches)
            output_scores.append(scores)
        # batched = collate_fn(datas, self.device)
        # pred = self.model(batched)
        # matches = pred["matches0"]
        # for i, ref_id in enumerate(range(start_idx, end_idx)):
        #     output_ids[i] = ref_id
        #     output_matches.append(matches[i].cpu().squeeze().short().numpy())
        
        rospy.logdebug(f"Num matches: {len(output_matches)}")
        rospy.logdebug(f"Num points: {len(output_matches[0])}")
            
        end_time = time()
        rospy.logdebug(f"Time taken: {(end_time - start_time) * 1000:.3f} ms")
        
        num_m = len(output_ids)
        num_p = len(output_matches[0])
        output_matches = np.concatenate(output_matches)
        output_scores = np.concatenate(output_scores)
        
        return QueryToMatchResponse(
            num_pairs = num_m,
            num_points = num_p,
            pts2d = output_pts2d,
            match0_ids = output_ids,
            match0 = output_matches,
            score0 = output_scores
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-03-24_17-45-31",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="/home/setsu/workspace/ORB_SLAM3/data/4Seasons/recording_2020-03-24_17-36-22/sfm_map/metadata",
        help="Path to the output directory, default: %(default)s",
    )
    args = parser.parse_args()
    seq_dir = args.dataset
    assert seq_dir.exists(), f"{seq_dir} does not exist"
    
    output_dir = args.outputs
    output_dir.mkdir(exist_ok=True, parents=True)
    
    rospy.init_node('match_server', log_level=rospy.DEBUG)
    server = MapServer(output_dir, seq_dir)
    rospy.spin()
