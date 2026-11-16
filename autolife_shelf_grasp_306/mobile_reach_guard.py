"""短距离补偿时只读监测当前身体、相机深度和里程计；不输出关节指令。"""
import json
import time
import numpy as np
from .mobile_reach import map_point,sweep_clear,corridor_remaining


class ApproachWatch:
    def __init__(self,node,request,run):
        from std_msgs.msg import String
        from rclpy.qos import qos_profile_sensor_data
        from .camera import Camera
        self.request=request;self.q=None;self.stamp=0.;self.checked=0.;self.image_id=None
        self.camera=Camera(request['camera_cfg'],run/'mobile_depth_preview.jpg')
        node.create_subscription(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_345',self.joints,qos_profile_sensor_data)

    def joints(self,m):
        from autolife_handeye_calibration_306.core import ARMS,BODY,NECK
        try:
            data=json.loads(m.data);q={}
            for key,names in [('leg_waist_joint_state',BODY),('neck_joint_state',NECK),
                              ('left_arm_joint_state',ARMS['left']),('right_arm_joint_state',ARMS['right'])]:
                v=data[key]
                if len(v['position'])!=len(names) or any(v.get('communication_lost',[])):return
                q.update(zip(names,v['position']))
            if not np.isfinite(list(q.values())).all():return
            self.q=q;self.stamp=time.monotonic()
        except (ValueError,KeyError,TypeError):return

    def check(self,odom,starting=False):
        from .bottle import object_points
        now=time.monotonic();r=self.request
        if self.q is None or now-self.stamp>.5:raise ValueError('补偿时等待当前身体保持反馈')
        if any(abs(v-r['source_q'][n])>.5 for n,v in self.q.items()):raise ValueError('身体姿态改变，需更新底盘扫掠与抓放路径')
        # 使用实测里程计，不把导航命令当成已经移动的距离。
        here=map_point([0,0,0],odom,r['base_anchor'])
        yaw=np.arctan2(np.sin(odom[2]-r['base_anchor'][2]),np.cos(odom[2]-r['base_anchor'][2]))
        remaining=corridor_remaining(r,here,yaw,starting)
        if now-self.checked<.15:return
        frame,error=self.camera.get()
        if frame is None or 'depth' not in frame or now-frame['received']>.3:raise ValueError(error or '补偿等待当前场景深度')
        if frame['frame_id']==self.image_id:
            if now-self.checked>.3:raise ValueError('补偿场景深度未更新')
            return
        T=np.asarray(r['camera_to_base'])
        points,_,_=object_points(frame,frame['depth']>0,T,r['camera_cfg'])
        # 平移期间身体不动，名义相机外参不变；只裁掉本次运动不可能碰到的远点。
        points=points[(points[:,0]>-.4)&(points[:,0]<1.5)&(np.abs(points[:,1])<1.)&(points[:,2]>.1)]
        if len(points):points=points[np.unique(np.floor(points/.008).astype(np.int64),axis=0,return_index=True)[1]]
        if not sweep_clear(points,r['body_boxes'],remaining,r.get('held_box')):raise ValueError('补偿路径有物体进入机身/手臂/持物扫掠范围')
        self.checked=now;self.image_id=frame['frame_id']

    def close(self):self.camera.close()
