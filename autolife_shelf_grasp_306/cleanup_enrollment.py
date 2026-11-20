"""A point, its measured support and its saved sample must belong to the same frame."""
import time
import uuid
import cv2
import numpy as np
from .cleanup_surface import overlaps_table
from .cleanup_catalog import enroll
from .point_selection import segment
from .cleanup_identity import STATE_PROMPT,visual_profile
from .table_cleanup import same_instance
from .segmentation_display import display_contours


class CleanupEnrollment:
    def __init__(self):
        self.token=None
        self.item=None

    def handle(self,cmd,root,run,perception,store,snapshot):
        action=cmd['action']
        if action=='cleanup_frame':
            self.token=None
            self.item=None
            self.frame=snapshot()
            self.tables=perception.segmenter.proposals(self.frame,'all',prompt='table. tabletop.')
            if not self.tables:raise ValueError('未识别到桌面，请重新取景')
            if not cv2.imwrite(str(run/'cleanup_frame.png'),self.frame['image']):raise OSError('桌面图像保存失败')
            self.token=uuid.uuid4().hex
            self.created=time.monotonic()
            return dict(token=self.token,shape=list(self.frame['image'].shape))
        if not self.token or cmd.get('token')!=self.token or time.monotonic()-self.created>120:
            raise ValueError('录入画面已过期，请重新取景')
        if action=='cleanup_point':
            self.item=None
            item=segment(perception,self.frame,store,cmd['x'],cmd['y'])
            if not overlaps_table(item,self.tables):raise ValueError('点选物体与桌面没有图像重叠，可重新点选')
            self.item=item
            return dict(id=item['id'],name=item['suggested_name'],box=item['box'],mask_contours=display_contours(item))
        if action!='cleanup_save' or self.item is None or cmd.get('id')!=self.item['id']:
            raise ValueError('请先点选并确认当前物体的轮廓')
        if 'crop' in self.item and 'views' not in self.item:
            self.item['views']=perception.embed([np.rot90(self.item['crop'],k) for k in (1,2,3)])
            states=perception.segmenter.proposals(self.frame,'all',prompt=STATE_PROMPT)
            self.item['profile']=visual_profile(perception,self.item,[v['semantic_label'] for v in states
                                                if same_instance(self.item['mask'],v['mask'])])
        count=enroll(root,cmd['name'],self.item)
        self.token=None
        self.frame=self.item=self.tables=None
        return dict(name=cmd['name'],count=count)
