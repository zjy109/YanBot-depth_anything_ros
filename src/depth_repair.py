import rospy
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
import os
from scipy.optimize import curve_fit
import shutil

from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image

from depth_anything_v2.dpt import DepthAnythingV2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.use('Agg')  # 设定非GUI后端，避免 Tkinter 问题

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

class DepthRestoration:
    def __init__(self, model_encoder, model_path, input_size):
        # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # torch.cuda.empty_cache()
        if torch.cuda.is_available():    
            DEVICE = 'cuda'
        else:
            print("No GPU available")
            exit()

        self.input_size = input_size

        self.cv_bridge = CvBridge()

        # 订阅rgb图像
        image_sub = Subscriber("color_to_process", Image, queue_size=1)
        # 订阅原始深度图像
        depth_sub = Subscriber("depth_to_process", Image, queue_size=1)
        # 同步回调
        ts = ApproximateTimeSynchronizer(
            [image_sub, depth_sub], queue_size=10, slop=0.05
        )
        ts.registerCallback(self.sync_sub_callback)

        # 修复掩码 发布者
        self.depth_repaired_pub = rospy.Publisher("depth_repaired", Image, queue_size=1)

        rospy.loginfo("Loading models...")
        
        depth_anything = DepthAnythingV2(**model_configs[model_encoder])
        depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.depth_anything = depth_anything.to(DEVICE).eval()

        rospy.loginfo("Models are loaded")

    def sync_sub_callback(self, image_msg, depth_msg):
        start_time = rospy.Time.now()

        # 将参数服务器里的depth_repair_processing设置成True，防止定时器在前一张图像的深度修复还没完成时发布新图像
        rospy.set_param("depth_repair_processing", True)

        # 提取图像时间戳，将图像消息转化为np数组形式
        time_stamp = image_msg.header.stamp
        image = self.cv_bridge.imgmsg_to_cv2(image_msg)
        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        # 计算原始深度图像的0值占比，如果0值太多则该深度图无效，直接返回
        zero_ratio = np.count_nonzero(raw_depth == 0) / raw_depth.size
        if zero_ratio > 0.5:
            rospy.loginfo("Invalid raw depth")
            return

        # 进行深度估计，得到相对深度
        estimated_depth = self.depth_anything.infer_image(image, self.input_size)

        end_time = rospy.Time.now()
        depth_repair_time = (end_time - start_time).to_sec()*1000
        rospy.loginfo(f"depth estimate time: {depth_repair_time:.1f} ms")

        print(f"image_timestamp:{time_stamp.to_sec()}")

        start_time = rospy.Time.now()

        # # Depth Anything V2 输出的是 逆深度，需要取倒数再得到与d435一致的深度值模式——距离相机较近的物体会有较小的值，较远的物体会有较大的值
        # estimated_depth = 1 / (estimated_depth)  # 实测还是直接用逆深度做 最小二乘 效果更好
        
        # 与原始深度进行大小对比，得到绝对深度
        absolute_depth = self.inverse_depth_to_absolute(estimated_depth.copy(), raw_depth.copy())

        absolute_depth_msg = self.cv_bridge.cv2_to_imgmsg(absolute_depth, encoding="16UC1")
        absolute_depth_msg.header.stamp = time_stamp

        # 测量深度修复的时间
        end_time = rospy.Time.now()
        depth_repair_time = (end_time - start_time).to_sec()*1000
        rospy.loginfo(f"convert to absolute depth time: {depth_repair_time:.1f} ms")
        print(" ")

        # ## 可视化估计深度和测量深度的差异
        # # 计算深度差异
        # depth_difference = absolute_depth.astype(np.int32) - raw_depth.astype(np.int32)
        # # # 计算数据的最大绝对值
        # # vmax = np.max(np.abs(depth_difference))
        # # 设置最大和最小值为 -1000 和 1000
        # vmin = -1000
        # vmax = 1000
        # # 限制 depth_difference 在 [-1000, 1000] 之间
        # depth_difference = np.clip(depth_difference, vmin, vmax)
        # # 使用 CenteredNorm 确保 0 映射到白色
        # norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # # 绘制图像
        # plt.clf()  # 清空之前的图像，防止colorbar叠加
        # plt.imshow(depth_difference, cmap='bwr', norm=norm, interpolation='nearest')
        # plt.colorbar()

        # # 将修复的深度做成可视化图片，调试用
        # vis_raw_depth = cv2.convertScaleAbs(raw_depth.copy(), alpha=0.06)
        # vis_absolute_depth = cv2.convertScaleAbs(absolute_depth.copy(), alpha=0.06)

        # outdir1 = '/home/zjy/vis_depth/raw'
        # outdir2 = '/home/zjy/vis_depth/absolute_estimate'
        # outdir3 = '/home/zjy/vis_depth/difference'

        # filename = f"{time_stamp.to_sec():.9f}.png"

        # # 如果目录已存在，先删除
        # if os.path.exists(outdir1):
        #     shutil.rmtree(outdir1)  # 递归删除整个目录及其内容

        # # 导出d435测量的原始深度
        # os.makedirs(outdir1, exist_ok=True)
        # vis_filename = os.path.join(outdir1, filename)
        # cv2.imwrite(vis_filename, vis_raw_depth)
        # # 导出估计并缩放后的绝对深度
        # os.makedirs(outdir2, exist_ok=True)
        # vis_filename = os.path.join(outdir2, filename)
        # cv2.imwrite(vis_filename, vis_absolute_depth)
        # # 导出深度差异（plt绘制结果，保存为为图片）
        # os.makedirs(outdir3, exist_ok=True)
        # plt_file_name = os.path.join(outdir3, f"{time_stamp.to_sec():.9f}.png")
        # plt.savefig(plt_file_name, dpi=300, bbox_inches='tight')

        self.depth_repaired_pub.publish(absolute_depth_msg)

        # 将参数服务器里的depth_repair_processing设置成False，允许定时器发布新图像
        rospy.set_param("depth_repair_processing", False)
    
    def inverse_depth_to_absolute(self, estimated_depth, measured_depth):
        """
        通过非线性拟合，将估计的逆深度转换为绝对深度
        :param estimated_depth: np.ndarray, 估计的逆深度 (H, W)
        :param measured_depth: np.ndarray, 测量得到的绝对深度 (H, W)
        :return: absolute_depth: np.ndarray, 变换后的绝对深度 (H, W)
        """
        valid_mask = (measured_depth > 100) & (estimated_depth > 1e-6)
        
        # 仅取有效区域
        est_valid = estimated_depth[valid_mask].flatten().astype(np.float32)
        meas_valid = measured_depth[valid_mask].flatten().astype(np.float32)
        # Depth Anything V2 输出的是 逆深度，需要取倒数再得到与d435一致的深度值模式——距离相机较近的物体会有较小的值，较远的物体会有较大的值
        meas_valid = 1.0 /meas_valid

        def inverse_depth_model(x, a, b):
            return a * x + b

        # 拟合 a 和 b
        # bounds=([min_a, min_b], [max_a, max_b]))    0.0001< a <0.0003      0.0001< b <0.0011
        # (a_opt, b_opt), _ = curve_fit(inverse_depth_model, est_valid, meas_valid, p0=[0.0002, 0.0006], 
        #                               bounds=([0.0001, 0.0005], [0.0003, 0.0009]), maxfev=50, xtol=1e-7, method='trf')
        # 实测发现不加范围限制运行速度快很多
        (a_opt, b_opt), _ = curve_fit(inverse_depth_model, est_valid, meas_valid, p0=[0.0002, 0.0006])
        
        # print(f"a: {a_opt}, b: {b_opt}")

        # 进行转换
        absolute_depth = (1.0 / (a_opt * estimated_depth + b_opt))

        # # 计算估计深度和测量深度的差值的平均值
        # diff_mean = np.mean(np.abs((absolute_depth - measured_depth)[valid_mask]))
        # print(f"Mean difference: {diff_mean}")

        absolute_depth = absolute_depth.astype(np.uint16)

        return absolute_depth # 确保深度非负


if __name__ == '__main__':
    rospy.init_node('depth_anything_ros')

    model_encoder = rospy.get_param('~model_encoder', 'vits')
    model_path = rospy.get_param('~model_path')
    input_size = rospy.get_param('~input_size', 518)

    DepthRestoration(model_encoder, model_path, input_size)
    rospy.spin()
