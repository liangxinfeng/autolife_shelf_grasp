"""Load one portable application configuration; never import a hardware driver."""
import os
from pathlib import Path
import sys
import yaml


PATH_FIELDS=('urdf_path','collision_urdf_path','calibration','left_calibration',
             'right_calibration','gripper_width_calibration','vendor_python_path',
             'vendor_control_config','python','vendor_python')


def load_config(root):
    """Resolve geometry/calibration paths beside the config root, not the cwd.

    Detector model paths keep their existing Perception(root) interpretation.
    A data-only root may use the installed package's configuration and assets.
    """
    root=Path(root).expanduser().resolve()
    path=root/'config'/'grasp.yaml'
    if not path.is_file():
        from ament_index_python.packages import get_package_share_directory
        root=Path(get_package_share_directory('autolife_shelf_grasp_306'))
        path=root/'config'/'grasp.yaml'
    cfg=yaml.safe_load(path.read_text('utf-8'))
    if not isinstance(cfg,dict):raise ValueError('抓取配置必须是键值表：'+str(path))
    for key in PATH_FIELDS:
        if cfg.get(key):
            resource=Path(cfg[key]).expanduser()
            cfg[key]=str(resource if resource.is_absolute() else root/resource)
    cfg['python']=cfg.get('python') or sys.executable
    if os.environ.get('ROBOT_ID') is not None:cfg['robot_id']=int(os.environ['ROBOT_ID'])
    return cfg
