#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import InferenceResult, Yolov11Inference

class YoloCupDetector(Node):
    def __init__(self) -> None:
        super().__init__('yolo_cup_detector')
        self.declare_parameter('camera_topic', '/camera_depth_sensor/image_raw')  # or /wrist_rgbd_depth_sensor/image_raw
        self.declare_parameter('model_path', '/home/user/ros2_ws/src/starbot_coffee_dispenser/object_detection/data/yolo11n.pt')
        self.declare_parameter('conf', 0.40)
        self.declare_parameter('iou', 0.50)

        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        model_path   = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf    = float(self.get_parameter('conf').value)
        self.iou     = float(self.get_parameter('iou').value)

        # Load model
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        # Resolve 'cup' class id (safer than hardcoding 41)
        self.cup_id = next((int(k) for k, v in self.class_names.items() if v == 'cup'), None)
        if self.cup_id is None:
            raise RuntimeError("Class 'cup' not found in model names.")

        # QoS suitable for camera streams
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, camera_topic, self.cb, sensor_qos)
        self.pub_det = self.create_publisher(Yolov11Inference, '/barista_1/cup_detections', 10)
        self.pub_img = self.create_publisher(Image, '/barista_1/cup_image', 10)

        self.get_logger().info(
            f"Subscribed to {camera_topic}; filtering class 'cup' (id {self.cup_id}); "
            f"conf>={self.conf}, iou={self.iou}"
        )

    def cb(self, msg: Image) -> None:
        # Convert to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Predict only cups
        results = self.model.predict(
            frame, conf=self.conf, iou=self.iou, classes=[self.cup_id], verbose=False
        )

        # Prepare per-frame message
        det_msg = Yolov11Inference()
        det_msg.header = msg.header  # keep original frame_id & timestamp from camera

        if results:
            r0 = results[0]
            if r0.boxes is not None:
                for box in r0.boxes:
                    b = box.xyxy[0].to('cpu').detach().numpy().copy()  # x1,y1,x2,y2
                    cls_id = int(box.cls.item())
                    score = float(box.conf.item()) if box.conf is not None else 0.0

                    inf = InferenceResult()
                    # remove these two lines if your msg doesn't have those fields
                    if hasattr(inf, 'class_id'): inf.class_id = cls_id
                    if hasattr(inf, 'score'):    inf.score = score
                    inf.class_name = self.class_names[cls_id]

                    inf.left, inf.top, inf.right, inf.bottom = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    inf.box_width  = inf.right - inf.left
                    inf.box_height = inf.bottom - inf.top
                    inf.x = inf.left + inf.box_width / 2.0
                    inf.y = inf.top  + inf.box_height / 2.0
                    det_msg.yolov11_inference.append(inf)

            # Publish annotated image
            annotated = r0.plot()
            out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out.header = msg.header
            self.pub_img.publish(out)

        # Publish detections (even if empty, useful downstream)
        self.pub_det.publish(det_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloCupDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
