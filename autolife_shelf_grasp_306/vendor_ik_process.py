"""原厂IK使用独立计算进程；解释器可明确配置，默认沿用当前Python。

协议只接受FK/IK数值计算，完全不导入或创建机器人控制器。
"""
import contextlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time


class NoIKSolution(ValueError):
    """Vendor solver finished its configured numerical search without a bounded pose."""


class VendorSolverExited(ValueError):
    """The computation process disappeared before returning an FK/IK result."""
    def __init__(self,pid,returncode,kind,values):
        self.details=dict(pid=pid,returncode=returncode,kind=kind,inputs=values)
        super().__init__(f'原厂计算进程退出（请求 {kind}，PID {pid}，退出码 {returncode}），本次求解结果未知')


class VendorIKProcess:
    def __init__(self, cfg, model, base, tip, locked, checkpoint=lambda:None):
        self.checkpoint=checkpoint;self._closed=False
        python = Path(cfg.get('vendor_python') or sys.executable)
        if not python.is_file():
            raise ValueError('原厂IK Python不存在：'+str(python))
        env = dict(os.environ)
        env['PYTHONFAULTHANDLER']='1'
        # Preserve explicitly supplied package search paths in the child. Do
        # not infer an interpreter from the site-packages directory layout.
        paths=[cfg.get('vendor_python_path')]+list(sys.path)
        env['PYTHONPATH']=os.pathsep.join(dict.fromkeys(str(p) for p in paths if p))
        env['LD_LIBRARY_PATH'] = ':'.join(v for v in env.get('LD_LIBRARY_PATH','').split(':')
            if v and '/s3_perception/' not in v)
        self.process = subprocess.Popen([str(python), '-u', str(Path(__file__).resolve()),
            str(model), base, tip, json.dumps(locked)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env)
        try:
            ready = self.read(15.)
            if ready != {'ready': True}:
                raise ValueError('原厂IK启动协议错误')
        except EOFError as exc:
            error=self.exit_error('startup',dict(locked=locked))
            self.close();raise error from exc
        except Exception:
            self.close();raise

    def read(self, timeout):
        deadline=time.monotonic()+timeout
        while True:
            self.checkpoint()
            if select.select([self.process.stdout],[],[],min(.05,max(0.,deadline-time.monotonic())))[0]:break
            if time.monotonic()>=deadline:raise ValueError('原厂IK计算超时')
        line = self.process.stdout.readline()
        if not line:raise EOFError('原厂计算进程没有返回结果')
        response = json.loads(line)
        if response.get('no_solution'):raise NoIKSolution(response['error'])
        if 'error' in response:raise ValueError(response['error'])
        return response

    def request(self, kind, **values):
        try:
            self.process.stdin.write(json.dumps(dict(kind=kind, **values))+'\n')
            self.process.stdin.flush()
            return self.read(5.)['result']
        except NoIKSolution:
            raise  # A complete response; the next request can reuse this solver.
        except (EOFError,BrokenPipeError,ConnectionResetError) as exc:
            error=self.exit_error(kind,values)
            self.close();raise error from exc
        except BaseException:
            # The protocol is ordered. A late reply to a cancelled/timed-out
            # request cannot be consumed as the answer to the next FK or IK.
            self.close()
            raise

    def exit_error(self,kind,values):
        code=self.process.poll()
        if code is None:
            # Capture the child's exit status before cleanup can send SIGTERM.
            with contextlib.suppress(subprocess.TimeoutExpired):code=self.process.wait(timeout=.05)
        return VendorSolverExited(self.process.pid,code,kind,values)

    def close(self):
        if self._closed:return
        self._closed=True
        p = self.process
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=2.)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=2.)
        with contextlib.suppress(OSError):p.stdin.close()
        p.stdout.close()


def worker():
    # 原厂库导入会打印提示，全部导向stderr，stdout仅传输JSON。
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from autolife_robot_sdk.utils.kdl_ik_solver import KdlSolver
        import tempfile
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        import math
        model, base, tip, initial_locked = sys.argv[1:]
        initial_locked=json.loads(initial_locked)
        allowed=set(initial_locked)
        solver=None;configuration=None
        def configure(locked):
            nonlocal solver,configuration
            angles={name:float(value) for name,value in locked.items()}
            if set(angles)!=allowed or any(not math.isfinite(v) or abs(v)>360 for v in angles.values()):
                raise ValueError('固定关节配置无效')
            # Reuse only the compiled chain. The caller still verifies each new
            # source pose by FK, and supplies the seed, bounds and target on every IK.
            current=(Path(model).read_bytes(),angles)
            if current==configuration:return
            tree=ET.ElementTree(ET.fromstring(current[0]))
            # 固定关节仍保留实测转角对应的完整安装变换，避免给IK多余自由度。
            for name, angle in angles.items():
                joint=tree.getroot().find(f"joint[@name='{name}']")
                origin=joint.find('origin')
                if origin is None:origin=ET.SubElement(joint,'origin',xyz='0 0 0',rpy='0 0 0')
                rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                rotation=Rotation.from_euler('xyz',rpy)*Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.deg2rad(angle))
                origin.set('rpy',' '.join(format(v,'.17g') for v in rotation.as_euler('xyz')))
                joint.set('type','fixed')
            with tempfile.TemporaryDirectory(prefix='vendor_ik_fixed_') as tmp:
                path=Path(tmp)/'fixed.urdf';tree.write(path)
                solver=KdlSolver(str(path), [base], [tip])
            configuration=current
        configure(initial_locked)
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request['kind']=='configure':
                    configure(request['locked']);result=[]
                elif request['kind']=='fk':
                    result = solver.fk([base],[tip],[np.asarray(request['q'])])[0]
                elif request['kind']=='ik':
                    seed=np.asarray(request['seed']);lower=np.asarray(request['lower']);upper=np.asarray(request['upper'])
                    solutions = solver.ik([base],[tip],[request['pose']],[seed],
                        ik_pos_solver_type='nr_jl',kwargs_ik_pos={
                            'q_min':lower,'q_max':upper,'maxiter':250,'eps':1e-5})
                    # The caller can recover a bounded seed and ask NR_JL again.
                    # The SDK's unconstrained LMA fallback crashed on a field target.
                    if not solutions:raise NoIKSolution('原厂有界逆解未收敛')
                    result=solutions[0]
                else:raise ValueError('仅支持无动作FK/IK')
            response = {'result':list(map(float,result))}
        except NoIKSolution as exc:
            response={'no_solution':True,'error':str(exc)}
        except Exception as exc:
            response = {'error':str(exc)}
        print(json.dumps(response,allow_nan=False),flush=True)


if __name__=='__main__':
    try:worker()
    except ImportError as exc:
        print(json.dumps({'error':'原厂IK依赖不可用：'+str(exc)},ensure_ascii=False),flush=True)
