import rospy
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2
import os

from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image

from depth_anything_v2.dpt import DepthAnythingV2

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

        # 提取图像时间戳，将图像消息转化为np数组形式
        time_stamp = image_msg.header.stamp
        image = self.cv_bridge.imgmsg_to_cv2(image_msg)
        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        # 进行深度估计，得到相对深度
        estimated_depth = self.depth_anything.infer_image(image, self.input_size)
        # 与原始深度进行大小对比，得到绝对深度
        absolute_depth = self.convert_to_absolute_depth(estimated_depth, raw_depth)

        absolute_depth_msg = self.cv_bridge.cv2_to_imgmsg(absolute_depth, encoding="16UC1")
        absolute_depth_msg.header.stamp = time_stamp

        vis_depth = cv2.convertScaleAbs(absolute_depth, alpha=0.1)

        # 测量深度修复的时间
        end_time = rospy.Time.now()
        depth_repair_time = (end_time - start_time).to_sec()*1000
        rospy.loginfo(f"depth repair time: {depth_repair_time:.1f} ms")

        # # 将修复的深度做成可视化图片，调试用
        # outdir = '/home/zjy/vis_depth'
        # filename = f"{time_stamp.to_sec():.9f}.png"

        # os.makedirs(outdir, exist_ok=True)
        # vis_filename = os.path.join(outdir, filename)
        # cv2.imwrite(vis_filename, vis_depth)

        # self.depth_repaired_pub.publish(absolute_depth_msg)

    def convert_to_absolute_depth(self, estimated_depth, measured_depth):
        D = measured_depth.astype(np.float32)
        X = estimated_depth.astype(np.float32)

        valid_mask = D > 0
        D_valid = D[valid_mask]
        X_valid = X[valid_mask]

        X_stack = np.vstack([X_valid, np.ones_like(X_valid)]).T
        params, residuals, rank, s = np.linalg.lstsq(X_stack, D_valid, rcond=None)
        A, b = params

        absolute_depth = (A * estimated_depth + b).astype(np.uint16)
        return absolute_depth


if __name__ == '__main__':
    rospy.init_node('depth_anything_ros')

    model_encoder = rospy.get_param('~model_encoder', 'vits')
    model_path = rospy.get_param('~model_path')
    input_size = rospy.get_param('~input_size', 518)

    DepthRestoration(model_encoder, model_path, input_size)
    rospy.spin()
