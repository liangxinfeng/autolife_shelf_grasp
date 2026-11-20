import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
from PyQt5 import QtCore as C
from autolife_shelf_grasp_306.navigation_ui import NavigationPanel


def panel():
    proc=Mock()
    proc.state.return_value=C.QProcess.Running
    proc.write.side_effect=len
    owner=Mock()
    owner.mobile_reach.navigation_event.return_value=False
    return NS(proc=proc,owner=owner,info=Mock(),buffer=b'',goal_key=None,inflight=False,
              data={},last_chain='',show_notice=Mock())


def test_late_goal_cannot_unlock_current_leg_or_cancel_workflow():
    p=panel()
    NavigationPanel.command(p,'navigate',run_id='run',leg='run:2')
    packets=[dict(event='nav_done',run_id='run',leg='run:1',success=True),
             dict(event='nav_error',run_id='run',leg='run:1',text='旧请求超时')]
    p.proc.readAllStandardOutput.return_value=''.join(json.dumps(x)+'\n' for x in packets).encode()
    NavigationPanel.read(p)
    assert p.inflight and p.goal_key==('run','run:2')
    p.owner.navigation_done.assert_not_called()
    p.owner.workflow.fail.assert_not_called()
    done=dict(event='nav_done',run_id='run',leg='run:2',success=True,name='前台')
    p.proc.readAllStandardOutput.return_value=((json.dumps(done)+'\n')*2).encode()
    NavigationPanel.read(p)
    assert not p.inflight and p.goal_key is None
    p.owner.navigation_done.assert_called_once_with(done)


def test_idle_navigation_bridge_exit_keeps_unrelated_action():
    p=panel();p.owner.workflow.active=True
    NavigationPanel.finished(p)
    p.owner.workflow.fail.assert_not_called()
    assert not p.data['ready']


def test_stop_reaches_navigation_even_if_studio_fails(window):
    w=window
    def failing_stop():
        assert (w.run/'stop').is_file()
        raise RuntimeError('模拟示教停止失败')
    w.studio.stop=failing_stop
    w.stop_motion()
    w.navigation.command.assert_called_with('stop')
    receipt=json.loads((w.run/'stop_reason.json').read_text())
    assert any('模拟示教停止失败' in msg for msg in receipt['stop_errors'])


from test_cuda_optional import window,qt
