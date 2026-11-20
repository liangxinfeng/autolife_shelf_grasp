"""Resolve geometry beside the configured URDF, including installed ROS packages."""
from pathlib import Path
import importlib.util
import sys


def collision_model_path(cfg):
    """An explicit collision model wins; otherwise use the active URDF's sibling."""
    return Path(cfg.get('collision_urdf_path') or
                Path(cfg['urdf_path']).with_name('robot_v3_1_simplified.urdf')).expanduser()


def add_vendor_import_paths(cfg):
    """The factory package may already be installed in the running interpreter."""
    for path in (cfg.get('vendor_python_path'),'/usr/lib/python3/dist-packages'):
        if path and path not in sys.path:sys.path.append(path)


def vendor_control_path(cfg):
    if cfg.get('vendor_control_config'):return Path(cfg['vendor_control_config'])
    add_vendor_import_paths(cfg)
    spec=importlib.util.find_spec('autolife_robot_arm')
    if spec is None or not spec.submodule_search_locations:
        raise ValueError('缺少原厂autolife_robot_arm依赖；请安装到当前Python，或配置vendor_python_path和vendor_control_config')
    package=Path(next(iter(spec.submodule_search_locations)))
    for relative in ('configs/autolife_s1/robot_v3_1.json','configs/robot_v3_1.json'):
        path=package/relative
        if path.is_file():return path
    raise ValueError('原厂包中未找到robot_v3_1.json；请明确配置vendor_control_config')
