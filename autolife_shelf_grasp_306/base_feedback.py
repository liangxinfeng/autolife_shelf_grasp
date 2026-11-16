"""只读底盘采样器：独立回调队列，不被关节/UI回调积压挡住。"""
import math
import time
import uuid


class BaseOdometryReader:
    def __init__(self, owner, topic):
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from nav_msgs.msg import Odometry
        self.sample = None
        self.stamp = None
        self.source_age = None
        self.node = Node('grasp_base_feedback_' + uuid.uuid4().hex[:8],
                         context=owner.context, use_global_arguments=False,
                         enable_rosout=False, start_parameter_services=False)
        self.executor = SingleThreadedExecutor(context=owner.context)
        try:
            # 只需要最新实测值，不能在主队列中依次消费旧底盘帧。
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.sub = self.node.create_subscription(Odometry, topic, self.receive, qos)
            self.executor.add_node(self.node)
        except BaseException:
            self.executor.shutdown(); self.node.destroy_node()
            raise

    def receive(self, msg):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.stamp:
            return  # 重发旧帧不能伪装成新反馈。
        self.stamp = stamp
        now = time.monotonic()
        source_ns = stamp[0] * 1000000000 + stamp[1]
        self.source_age = (self.node.get_clock().now().nanoseconds - source_ns) / 1e9
        p, r, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
        self.sample = (now, p.x, p.y,
                       math.atan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y * r.y + r.z * r.z)),
                       math.hypot(v.linear.x, v.linear.y), abs(v.angular.z))

    def latest(self):
        # 不加固定等待，不跑控制节点的定时器；只消费独立队列的最新里程计。
        self.executor.spin_once(timeout_sec=0.)
        return self.sample

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
