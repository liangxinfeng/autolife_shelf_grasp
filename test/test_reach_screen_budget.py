"""Cold preparation and actual reach solving share only the global deadline."""
from types import SimpleNamespace as NS

import pytest

from autolife_shelf_grasp_306 import reach_precheck as reach
from autolife_shelf_grasp_306.planning_control import PlanningControl,PlanningTimeout


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def advance(self,seconds):self.now+=seconds


def harness(monkeypatch,prepare=(0.,0.,0.,0.)):
    clock=Clock();children=[];prepared=[]
    monkeypatch.setattr(reach.time,'monotonic',clock)
    def control(*args,**kwargs):
        child=PlanningControl(*args,clock=clock,**kwargs)
        children.append(child);return child
    monkeypatch.setattr(reach,'PlanningControl',control)
    parent=PlanningControl(timeout_sec=10.,clock=clock)
    def arm_config(side,cfg):
        prepared.append(side);clock.advance(prepare[0])
        return {'arm':side},[0.,0.,0.]
    def broad_margin(*args):
        clock.advance(prepare[1]);return .1
    def native_planner(cfg,context):
        clock.advance(prepare[2])
        return NS(configure_source=lambda source:clock.advance(prepare[3]))
    pool=NS(arm_config=arm_config,native_planner=native_planner)
    monkeypatch.setattr(reach,'broad_margin',broad_margin)
    def run(order=('left','right')):
        return reach.screen(pool,{},[.4,0.,0.],{}, {},order,parent,lambda text:None,None)
    return clock,parent,children,prepared,run


def test_cold_preparation_does_not_consume_shared_native_screen_budget(monkeypatch):
    # Configuration+envelope alone take the complete old 1.5-second screen
    # budget on each arm. Both arms must still receive actual solving time.
    clock,parent,children,prepared,run=harness(monkeypatch,(.8,.7,.5,.1))
    calls=[]
    def probe(planner,*args):
        calls.append(planner.context.timeout_sec)
        clock.advance(.4 if len(calls)==1 else .5)
        planner.context.checkpoint()
        return {'status':'unresolved' if len(calls)==1 else 'key_poses_passed'}
    monkeypatch.setattr(reach,'native_key_poses',probe)
    rows=run()
    assert prepared==['left','right'] and len(rows)==2
    assert calls==pytest.approx([reach.SCREEN_SECONDS,reach.SCREEN_SECONDS-.4])
    assert rows[-1]['status']=='key_poses_passed'
    assert clock()==pytest.approx(5.1) and parent.deadline==10.
    assert children[-1].deadline<parent.deadline


def test_actual_native_time_is_shared_and_cannot_restart_per_arm(monkeypatch):
    clock,parent,children,prepared,run=harness(monkeypatch)
    calls=[]
    def probe(planner,*args):
        calls.append(planner.context.timeout_sec)
        clock.advance(.9 if len(calls)==1 else .61)
        planner.context.checkpoint()
        return {'status':'unresolved'}
    monkeypatch.setattr(reach,'native_key_poses',probe)
    rows=run(('left','right','left'))
    assert calls==pytest.approx([1.5,.6])
    assert prepared==['left','right'] and len(rows)==2
    assert all(row['status']=='unresolved' for row in rows)
    assert parent.terminal is None and parent.deadline==10.


def test_cold_preparation_still_exhausts_global_ten_second_budget(monkeypatch):
    clock,parent,children,prepared,run=harness(monkeypatch,(6.,4.1,0.,0.))
    monkeypatch.setattr(reach,'native_key_poses',lambda *args:pytest.fail('global budget already exhausted'))
    with pytest.raises(PlanningTimeout,match='10秒'):
        run()
    assert clock()==pytest.approx(10.1) and parent.deadline==10.
    assert parent.terminal[0] is PlanningTimeout
    assert prepared==['left'] and not children


def test_native_budget_is_capped_by_global_time_remaining(monkeypatch):
    clock,parent,children,prepared,run=harness(monkeypatch,(5.,4.5,0.,0.))
    calls=[]
    def probe(planner,*args):
        calls.append(planner.context.timeout_sec)
        clock.advance(.6)
        planner.context.checkpoint()
        pytest.fail('native checkpoint must propagate parent timeout')
    monkeypatch.setattr(reach,'native_key_poses',probe)
    with pytest.raises(PlanningTimeout,match='10秒'):
        run()
    assert calls==pytest.approx([.5]) and len(children)==1
    assert children[0].deadline==parent.deadline==10.
    assert prepared==['left'] and parent.terminal[0] is PlanningTimeout
