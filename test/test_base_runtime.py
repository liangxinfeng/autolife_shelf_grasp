import sys
from autolife_shelf_grasp_306.base_execution import helper_command,helper_exit_reason

def test_chassis_uses_the_current_deployment_interpreter(monkeypatch):
    monkeypatch.setattr(sys,'executable','/vision/venv/bin/python')
    assert helper_command()==['/vision/venv/bin/python','-m','autolife_shelf_grasp_306.chassis','--fixed-sequence']

def test_exit_message_exposes_dependency_failure(tmp_path):
    p=tmp_path/'error.log';p.write_text("Traceback:\nModuleNotFoundError: No module named 'redis'\n")
    assert 'redis' in helper_exit_reason(1,p)

def test_no_error_log_still_reports_exit_code(tmp_path):
    assert '2' in helper_exit_reason(2,tmp_path/'absent')
