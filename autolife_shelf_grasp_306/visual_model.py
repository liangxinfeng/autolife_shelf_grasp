"""Validate the coordinate chain used to teach a visual reference."""
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def coordinate_chain(raw):
    robot=ET.fromstring(raw)
    # ModelGeometry.frames uses joint transforms and link topology.
    # tool_transform derives the TCP from finger and wrist link frames.
    links=sorted(link.attrib['name'] for link in robot.findall('link'))
    joints=[ET.canonicalize(ET.tostring(j,encoding='unicode'),strip_text=True)
            for j in robot.findall('joint')]
    return links,joints


def check_reference_model(meta,geometry):
    current=Path(geometry.path).read_bytes()
    source=meta['model_sha256']
    if meta.get('frame')!=geometry.root:
        raise ValueError('参照坐标系已改变，请重新记录')
    if hashlib.sha256(current).hexdigest()==source:return
    if not isinstance(source,str) or not re.fullmatch('[a-f0-9]{64}',source):
        raise ValueError('参照模型摘要无效')
    archive=Path(geometry.path).parent/'reference_models'/(source+'.urdf')
    if archive.is_file():
        old=archive.read_bytes()
        if hashlib.sha256(old).hexdigest()==source and coordinate_chain(old)==coordinate_chain(current):
            return
    raise ValueError('本机模型的参照坐标链无法验证，请重新记录参照')
