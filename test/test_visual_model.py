import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from autolife_shelf_grasp_306.visual_model import check_reference_model
from autolife_shelf_grasp_306.visual_action_worker import VisualActions

MODELS=Path(__file__).resolve().parents[1]/'urdf'


def pair():
    old=next((MODELS/'reference_models').glob('*.urdf'))
    meta={'model_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'frame':'Link_Zero_Point'}
    return old,meta,NS(root='Link_Zero_Point',path=MODELS/'robot_v3_1.urdf')


def test_authenticated_legacy_chain():
    old,meta,current=pair()
    assert meta['model_sha256']!=hashlib.sha256(current.path.read_bytes()).hexdigest()
    check_reference_model(meta,current)


@pytest.mark.parametrize('changed',['knee','finger','camera','frame','archive','mount'])
def test_change_to_reference_geometry_is_rejected(tmp_path,changed):
    old,meta,current=pair()
    folder=tmp_path/'reference_models';folder.mkdir()
    (folder/old.name).write_bytes(old.read_bytes() if changed!='archive' else b'changed')
    tree=ET.parse(current.path)
    if changed=='knee':tree.find(".//joint[@name='Joint_Knee']/axis").set('xyz','0 1 0')
    if changed in ('finger','camera'):
        keyword='Finger' if changed=='finger' else 'Camera'
        joint=next(j for j in tree.findall('joint') if keyword in j.find('child').get('link'))
        joint.find('origin').set('xyz','1 2 3')
    if changed=='frame':meta['frame']='other_frame'
    current.path=tmp_path/'new.urdf';tree.write(current.path)
    with pytest.raises(ValueError):
        check_reference_model(meta,current)
        if changed=='mount':
            actual=np.eye(4);actual[0,3]=.01
            VisualActions.check_mount(NS(mount=lambda q,T:actual),{'camera_mount':np.eye(4)},None,None)
