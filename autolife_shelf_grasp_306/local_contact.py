"""Front contact using a measured whole-object hull, never a smaller held object.

Offsets are represented by a virtual object-centre TCP during IK/placement;
the calibrated physical TCP remains explicit for finger-clearance checks.
"""
import numpy as np
from scipy.spatial import ConvexHull
from .bottle import horizontal_basis
from .object_extent import bbox

MAX_CONTACT_OFFSET_M=.060


def merged_footprint(products,centers,center,up):
    points=np.concatenate([np.asarray(p['footprint_vertices_torso'])+c-center for p,c in zip(products,centers)])
    u,v,n=horizontal_basis(up);xy=points@np.column_stack([u,v])
    return points[ConvexHull(xy).vertices].tolist()


def prism(product):
    vertices=np.asarray(product['measured_footprint_torso'],float)
    axes,extent=bbox(product);up=axes[:,2]
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or len(vertices)>256 or not np.isfinite(vertices).all():
        raise ValueError('局部夹取缺少可靠物体轮廓')
    # Re-centre the hull vertically, retaining conservative whole-object height.
    vertices=vertices-(vertices@up)[:,None]*up
    p=np.r_[vertices-up*extent[2]/2,vertices+up*extent[2]/2]
    n=len(vertices);i=np.arange(n);j=(i+1)%n
    edges=np.r_[np.c_[i,j],np.c_[i+n,j+n],np.c_[i,i+n]]
    return p,edges


def swept_points(product,contact,forward,reach=.018):
    """Whole-height prism portion entered by fingertips, including near side.

    robot_v3_1 fingers protrude 12.2 mm past calibrated TCP; 18 mm includes
    reserve. Farther material is not deleted from the held-object envelope.
    """
    p,edges=prism(product);d=(p-contact)@forward-reach
    a,b=edges.T;cross=(d[a]*d[b])<0
    a=a[cross];b=b[cross]
    cuts=p[a]+(p[b]-p[a])*(d[a]/(d[a]-d[b]))[:,None]
    result=np.r_[p[d<=0],cuts]
    return result


def required_opening(product,contact,rotation,preopening_margin=.010):
    p=swept_points(product,contact,rotation[:,0])
    if not len(p):return 0.
    # 两指对称：偏心物体必须计入偏心量。保留4mm几何测量余量。
    # 默认优先选有余量的局部夹点，避免只夹到方盒尖角；验证已选路径时
    # 传0，只核对真实必需开口，预张开余量由执行器按最大开口调整。
    return float(2*np.max(np.abs((p-contact)@rotation[:,1]))+.004+preopening_margin)


def candidates(product,cfg):
    if 'measured_footprint_torso' not in product:return []
    vertices=np.asarray(product['measured_footprint_torso']);result=[];near_limit=[]
    for yaw in [v for a in range(0,91,5) for v in ([0.] if a==0 else [-float(a),float(a)])]:
        forward,side,up=horizontal_basis(cfg['up_torso'],np.deg2rad(cfg['approach_yaw_deg']+yaw))
        rotation=np.column_stack([forward,side,up]);near=float(np.min(vertices@forward))
        close_fit=None
        # Closest-to-centre valid contact first; require >=10mm engagement.
        for x in np.arange(-.005,max(near+.010,-MAX_CONTACT_OFFSET_M)-1e-8,-.005):
            contact=forward*x
            p=swept_points(product,contact,forward);lateral=p@side
            contact+=side*(lateral.min()+lateral.max())/2
            width=required_opening(product,contact,rotation,preopening_margin=0.)
            if width<=cfg['gripper_max_width_m']:
                candidate=dict(yaw=yaw,contact=contact,required_opening_m=width)
                if width+.010<=cfg['gripper_max_width_m']:
                    result.append(candidate)
                    close_fit=None
                    break
                # At the aperture limit, accept a verified whole-width span.
                # A diagonal sliver at the corner of an oversized box is not
                # an opposed contact pair.
                if close_fit is None and np.ptp(vertices@side)+.004<=cfg['gripper_max_width_m']:
                    close_fit=candidate
        if close_fit is not None:near_limit.append(close_fit)
        if len(result)>=5:break
    return (result+near_limit)[:5]


def held_tcp(grasp,physical_tcp):
    """Use the saved centre offset, not today's arm orientation or new vision."""
    if 'physical_tcp' not in grasp:return np.asarray(physical_tcp)
    physical=np.asarray(grasp['physical_tcp']);virtual=np.asarray(grasp['tcp'])
    offset=np.asarray(grasp['product']['object_center_offset_wrist'])
    if (physical.shape!=(3,) or virtual.shape!=(3,) or offset.shape!=(3,) or
        not np.isfinite(virtual).all() or not np.allclose(physical,physical_tcp,atol=1e-7) or
        not np.allclose(virtual,physical+offset,atol=1e-7) or np.linalg.norm(offset)>.08):
        raise ValueError('局部夹取中心偏移记录不一致，请勿继续放置')
    return virtual
