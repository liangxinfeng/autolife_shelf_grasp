"""Vendor dependency discovery and interpreter selection, without a child process."""
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from autolife_shelf_grasp_306 import resources
from autolife_shelf_grasp_306 import vendor_ik_process as ik


def test_vendor_config_can_come_from_installed_package_without_fixed_site_path(tmp_path,monkeypatch):
    path=tmp_path/'configs'/'autolife_s1'/'robot_v3_1.json';path.parent.mkdir(parents=True);path.write_text('{}')
    calls=[]
    def spec(name):
        calls.append(name);return SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    monkeypatch.setattr(resources.importlib.util,'find_spec',spec)
    assert resources.vendor_control_path({})==path
    assert calls==['autolife_robot_arm']
    explicit=tmp_path/'another.json'
    assert resources.vendor_control_path({'vendor_control_config':str(explicit)})==explicit
    assert len(calls)==1


def test_missing_vendor_dependency_has_actionable_error(monkeypatch):
    monkeypatch.setattr(resources.importlib.util,'find_spec',lambda name:None)
    with pytest.raises(ValueError,match='缺少原厂autolife_robot_arm依赖'):
        resources.vendor_control_path({})


@pytest.mark.parametrize('explicit',[False,True])
def test_ik_interpreter_is_explicit_or_current_never_guessed_from_site_path(tmp_path,monkeypatch,explicit):
    cfg={'vendor_python_path':str(tmp_path/'unrelated_package_location')}
    expected=sys.executable
    if explicit:
        executable=tmp_path/'chosen_python';executable.write_text('not executed')
        cfg['vendor_python']=expected=str(executable)
    launches=[]
    def spawn(command,**kwargs):
        launches.append((command,kwargs))
        return SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO(),poll=lambda:0)
    monkeypatch.setattr(ik.subprocess,'Popen',spawn)
    monkeypatch.setattr(ik.VendorIKProcess,'read',lambda self,timeout:{'ready':True})
    solver=ik.VendorIKProcess(cfg,'model.urdf','base','tip',{})
    solver.close()
    command,options=launches[0]
    assert command[0]==str(Path(expected))
    assert cfg['vendor_python_path'] in options['env']['PYTHONPATH'].split(os.pathsep)


def test_ik_current_interpreter_needs_no_vendor_python_path(monkeypatch):
    launches=[]
    monkeypatch.setattr(ik.subprocess,'Popen',lambda command,**kw:
        launches.append(command) or SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO(),poll=lambda:0))
    monkeypatch.setattr(ik.VendorIKProcess,'read',lambda self,timeout:{'ready':True})
    solver=ik.VendorIKProcess({},'model.urdf','base','tip',{});solver.close()
    assert launches[0][0]==str(Path(sys.executable))


@pytest.mark.parametrize('error,closed',[(ik.NoIKSolution('no pose'),False),
                                         (ValueError('timeout awaiting reply'),True),
                                         (BrokenPipeError('lost worker'),True)])
def test_solver_request_reuses_only_a_complete_reply(monkeypatch,error,closed):
    process=SimpleNamespace(pid=55,stdin=io.StringIO(),stdout=io.StringIO(),poll=lambda:-11)
    solver=object.__new__(ik.VendorIKProcess)
    solver.process=process;solver._closed=False
    def read(timeout):raise error
    monkeypatch.setattr(solver,'read',read)
    expected=ik.VendorSolverExited if isinstance(error,BrokenPipeError) else type(error)
    with pytest.raises(expected) as caught:solver.request('ik',seed=[0.])
    if expected is ik.VendorSolverExited:
        assert caught.value.details==dict(pid=55,returncode=-11,kind='ik',inputs={'seed':[0.]})
    assert solver._closed is closed
    assert process.stdin.closed is closed


def test_kdl_worker_reuses_only_identical_geometry_and_lock_angles(tmp_path,monkeypatch,capsys):
    from types import ModuleType
    model=tmp_path/'robot.urdf'
    xml='<robot name="a"><joint name="ankle" type="revolute"><axis xyz="0 1 0"/></joint></robot>'
    model.write_text(xml);stamp=model.stat()
    builds=[];fk_inputs=[];ik_inputs=[]
    class Solver:
        def __init__(self,path,bases,tips):
            builds.append(Path(path).read_bytes())
        def fk(self,bases,tips,angles):
            fk_inputs.append(angles[0].tolist());return angles
        def ik(self,bases,tips,poses,seeds,**kwargs):
            assert kwargs['ik_pos_solver_type']=='nr_jl'
            if poses[0]==[-1]:return []
            ik_inputs.append((poses[0],seeds[0].tolist(),kwargs['kwargs_ik_pos']['q_max'].tolist()))
            return seeds
    module=ModuleType('autolife_robot_sdk.utils.kdl_ik_solver');module.KdlSolver=Solver
    monkeypatch.setitem(sys.modules,module.__name__,module)
    def requests():
        for value in (1.,2.):
            yield json.dumps(dict(kind='configure',locked={'ankle':0.}))
            yield json.dumps(dict(kind='fk',q=[value]))
            yield json.dumps(dict(kind='ik',pose=[value],seed=[value],lower=[0],upper=[value+2]))
        yield json.dumps(dict(kind='configure',locked={'ankle':.00001}))
        yield json.dumps(dict(kind='configure',locked={'ankle':.00001}))
        model.write_text(xml.replace('name="a"','name="b"'))
        os.utime(model,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        yield json.dumps(dict(kind='configure',locked={'ankle':.00001}))
        yield json.dumps(dict(kind='ik',pose=[-1],seed=[0],lower=[-1],upper=[1]))
        yield json.dumps(dict(kind='configure',locked={'other':0.}))
    monkeypatch.setattr(sys,'stdin',requests())
    monkeypatch.setattr(sys,'argv',['worker',str(model),'base','tip','{"ankle":0}'])
    ik.worker()
    replies=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert replies[0]=={'ready':True} and all('result' in r for r in replies[1:-2])
    assert replies[-2]['no_solution'] is True
    assert 'error' in replies[-1] and len(builds)==3 and len(set(builds))==3
    assert fk_inputs==[[1.],[2.]]
    assert ik_inputs==[([1.],[1.],[3.]),([2.],[2.],[4.])]
