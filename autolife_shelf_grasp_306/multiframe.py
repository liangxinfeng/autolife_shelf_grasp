"""Bounded majority fusion: tolerate isolated depth tails, not missing geometry."""
import math
import numpy as np


def consensus_indices(centers, dimensions=None):
    points=np.asarray(centers,float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points):
        raise ValueError('有效深度帧不足，请重试')
    if not np.isfinite(points).all():raise ValueError('物体中心含无效深度')
    if len(points)==2 and np.linalg.norm(points[1]-points[0])>.010:
        raise ValueError('两次观测的抓点互相矛盾，请重新取景')
    keep=np.linalg.norm(points-np.median(points,axis=0),axis=1)<=.010
    if dimensions is not None:
        sizes=np.asarray(dimensions,float)
        if sizes.shape!=points.shape or not np.isfinite(sizes).all() or np.any(sizes<=0):
            raise ValueError('物体尺寸数据无效')
        median=np.median(sizes,axis=0)
        # Compare each observation to the consensus, not two noisy extremes.
        # Width/depth have already been projected into the SAME physical axes.
        # Extra 1mm covers quantization rather than interpreting it as shape.
        tolerance=np.maximum(.008,median*.15)+.001
        if len(sizes)==2 and np.any(np.abs(sizes[1]-sizes[0])>tolerance):
            raise ValueError('两次观测的尺寸互相矛盾，请重新取景')
        keep&=np.all(np.abs(sizes-median)<=tolerance,axis=1)
    minimum=math.ceil(len(points)*.70)
    if keep.sum()<minimum:
        raise ValueError('当前观测的尺寸或中心互相矛盾，请重新取景')
    center=np.median(points[keep],axis=0)
    keep&=np.linalg.norm(points-center,axis=1)<=.010
    if keep.sum()<minimum:raise ValueError('多帧物体中心仍在变化，请保持物体和机器人不动后重试')
    return np.flatnonzero(keep).tolist()
