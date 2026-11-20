"""Workflow editor and Qt adapter over existing navigation/grasp/place APIs."""
import json
import time
from copy import deepcopy
from PyQt5 import QtCore as C, QtWidgets as Q
from .task_flow import Cursor, KINDS, TargetNotVisible, validate, select_candidates
from .core import atomic_json
from .task_pause import defer_call
from .task_capabilities import require_perception, worker_required_now
from .task_runtime import TaskStore
from .held_receipt import has_held


class FlowDialog(Q.QDialog):
    def __init__(self, owner):
        super().__init__(owner); self.owner = owner; self.resize(1280, 640)
        self.setWindowTitle('任务流程 · 导航 / 抓取 / 放置 / 示教动作 / 底盘定量 / 语音')
        self.path = owner.root / 'data/task_flows.json'
        self.library = json.loads(self.path.read_text('utf-8')) if self.path.exists() else {}
        layout = Q.QVBoxLayout(self); row = Q.QHBoxLayout(); layout.addLayout(row)
        self.names = Q.QComboBox(); self.names.addItem('新建流程'); self.names.addItems(sorted(self.library)); row.addWidget(self.names)
        self.name = Q.QLineEdit(); self.name.setPlaceholderText('流程名称'); row.addWidget(self.name, 1)
        nav_row=Q.QHBoxLayout();layout.addLayout(nav_row)
        self.nav_note=Q.QLabel();self.nav_note.setWordWrap(True);nav_row.addWidget(self.nav_note,1)
        if not owner.ready:
            capability=Q.QLabel('当前视觉不可用：动作、导航和底盘步骤可照常运行；流程到达抓取/放置步骤时保留进度，等待视觉恢复。')
            capability.setWordWrap(True);layout.addWidget(capability)
        refresh=Q.QPushButton('刷新航点');refresh.clicked.connect(self.refresh_waypoints);nav_row.addWidget(refresh)
        types_button=Q.QPushButton('垃圾种类清单');types_button.clicked.connect(self.edit_cleanup_types);nav_row.addWidget(types_button)
        self.enroll_trash=Q.QPushButton('点选录入垃圾');self.enroll_trash.clicked.connect(self.pick_cleanup_sample);nav_row.addWidget(self.enroll_trash)
        self.navigation_button=Q.QPushButton('导航地图 / 定位');self.navigation_button.clicked.connect(self.navigation_map);nav_row.addWidget(self.navigation_button)
        self.table = Q.QTableWidget(0, 7); self.table.setHorizontalHeaderLabels(['步骤类型', '航点 / 物品 / 动作名称 / 播报文字','身体高度方式','下降高度(cm)','头部俯仰(°)','执行手臂','动作速度 / 播报策略'])
        for col,width in ((0,105),(2,150),(3,125),(4,155)):self.table.setColumnWidth(col,width)
        self.table.horizontalHeader().setSectionResizeMode(1, Q.QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        row = Q.QHBoxLayout(); layout.addLayout(row)
        for kind, label in KINDS.items():
            button = Q.QPushButton('＋' + label); row.addWidget(button)
            button.clicked.connect(lambda _, k=kind: self.add(k, ''))
        for label, fn in [('删除步骤', self.remove), ('上移', lambda: self.move(-1)), ('下移', lambda: self.move(1))]:
            button = Q.QPushButton(label); row.addWidget(button); button.clicked.connect(fn)
        note = Q.QLabel('高度为从最高站姿下降：正向0～56cm，反向0～53cm。头部0°正前方，正数俯视，负数仰视。\n导航途中调姿，二者完成后继续；抓取复位后调姿再识别；放置/示教动作开始前调姿。示教动作默认沿用记录的身体高度及其全过程变化，不额外插入单独升降。')
        note.setWordWrap(True); layout.addWidget(note)
        row = Q.QHBoxLayout(); layout.addLayout(row)
        for label, fn in [('保存流程', self.save), ('删除流程', self.delete), ('试听选中语音',self.preview_voice), ('开始流程', self.start)]:
            button = Q.QPushButton(label); row.addWidget(button); button.clicked.connect(fn)
        self.continue_button=Q.QPushButton('手动继续');row.addWidget(self.continue_button)
        self.continue_button.clicked.connect(self.continue_task)
        self.progress=Q.QLabel();self.progress.setWordWrap(True);layout.addWidget(self.progress)
        self.names.currentTextChanged.connect(self.load)
        self.waypoint_timer=C.QTimer(self);self.waypoint_timer.setInterval(750)
        self.waypoint_timer.timeout.connect(self.refresh_waypoints)
        self.refresh_waypoints()

    def showEvent(self,event):
        super().showEvent(event);self.refresh_waypoints();self.waypoint_timer.start()

    def navigation_map(self):
        from .navigation_ui import NavigationDialog
        panel=self.owner.navigation
        if panel.dialog:panel.dialog.close()
        dialog=NavigationDialog(panel,parent=self);panel.dialog=dialog
        try:dialog.exec_()
        finally:panel.dialog=None;dialog.deleteLater()

    def done(self,result):
        self.waypoint_timer.stop()
        if hasattr(self,'voice_preview'):self.voice_preview.stop()
        super().done(result)

    def refresh_waypoints(self):
        self.owner.controls()
        flow=getattr(self.owner,'workflow',None)
        self.enroll_trash.setEnabled(bool(self.owner.ready and flow and not flow.pending and flow.channels_idle()))
        self.continue_button.setEnabled(self.owner.flow_retry.isEnabled())
        self.continue_button.setToolTip(self.owner.flow_retry.toolTip())
        self.progress.setText(self.owner.flow_label.text())
        names=self.choices('navigate');data=self.owner.navigation.data
        from pathlib import Path
        label=Path(data.get('active_map') or '').stem or '当前地图'
        unavailable=False
        for i in range(self.table.rowCount()):
            if self.table.item(i,0).data(C.Qt.UserRole)!='navigate':continue
            box=self.table.cellWidget(i,1);selected=box.currentText()
            if box.property('waypoint_choices')!=names:
                cursor=box.lineEdit().cursorPosition() if box.isEditable() else 0
                box.blockSignals(True);box.clear();box.addItems(names)
                if selected:box.setCurrentText(selected);box.lineEdit().setCursorPosition(cursor)
                box.setProperty('waypoint_choices',names);box.blockSignals(False)
            invalid=bool(selected and selected not in names);unavailable|=invalid
            box.setToolTip('该航点不在当前列表，请重新选择' if invalid else '点击右侧下拉箭头选择控制中心航点')
        suffix='导航已连接' if data.get('ready') else '可先编排；执行前需启动导航并完成定位'
        if unavailable:suffix+='；部分步骤航点已失效，请重新选择'
        self.nav_note.setText(f'{label} · {len(names)}个航点 · {suffix}' if names else
            '尚未读取到航点，请在控制中心选择有航点的地图后刷新')

    def continue_task(self):
        self.owner.workflow.retry_current()
        if self.owner.workflow.active:self.accept()
        else:self.refresh_waypoints()

    def edit_cleanup_types(self):
        from .cleanup_catalog import library,save_types
        try:
            catalog=library(self.owner.root)
        except (OSError,ValueError) as exc:
            Q.QMessageBox.warning(self,'读取垃圾种类失败',str(exc))
            return
        dialog=Q.QDialog(self);dialog.setWindowTitle('垃圾种类清单')
        layout=Q.QVBoxLayout(dialog)
        layout.addWidget(Q.QLabel('每行一种。自定义识别词可写“烟头 = cigarette butt”。'))
        editor=Q.QPlainTextEdit('\n'.join(catalog['types']));layout.addWidget(editor)
        general=Q.QCheckBox('同时识别清单以外有明显废弃特征的物品')
        general.setChecked(catalog.get('open_waste',bool(catalog['types'])));layout.addWidget(general)
        buttons=Q.QDialogButtonBox(Q.QDialogButtonBox.Ok|Q.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        while dialog.exec_()==Q.QDialog.Accepted:
            try:
                save_types(self.owner.root,editor.toPlainText().splitlines(),general.isChecked())
                return
            except (OSError,ValueError) as exc:Q.QMessageBox.warning(self,'未保存',str(exc))

    def pick_cleanup_sample(self):
        from .cleanup_enrollment_ui import SampleDialog
        dialog=SampleDialog(self);self.owner.cleanup_dialog=dialog
        try:
            dialog.freeze()
            dialog.exec_()
        finally:
            self.owner.cleanup_dialog=None
            dialog.deleteLater()

    def choices(self, kind):
        if kind == 'navigate':
            from .navigation_logic import waypoint_names
            return waypoint_names(self.owner.navigation.data)
        if kind == 'action': return sorted(self.owner.studio.records())
        if kind == 'grasp':
            path = self.owner.root/'data/products.json'
            return [p['name'] for p in json.loads(path.read_text()).get('products', [])] if path.exists() else []
        if kind=='tts':return []
        return ['当前桌面']

    def add(self, kind, target, posture=None, base=None, execution_arm="original", action_speed_percent=None, required=False):
        row = self.table.rowCount(); self.table.insertRow(row)
        item = Q.QTableWidgetItem(KINDS[kind]); item.setData(C.Qt.UserRole, kind)
        item.setFlags(item.flags() & ~C.Qt.ItemIsEditable); self.table.setItem(row, 0, item)
        box = Q.QComboBox(); box.setEditable(kind not in ('place','cleanup')); box.addItems(self.choices(kind))
        if kind=='navigate':
            box.setProperty('waypoint_choices',self.choices(kind))
            box.setMinimumContentsLength(12);box.setMaxVisibleItems(16)
            box.lineEdit().setPlaceholderText('点击右侧 ▼ 选择航点')
        if target: box.setCurrentText(target)
        if kind=='base':
            from .base_sequence_ui import BaseCell
            box.deleteLater();box=BaseCell(self.table,base)
        self.table.setCellWidget(row, 1, box); self.table.selectRow(row)
        p=posture or {};direction=Q.QComboBox()
        if kind=='action':direction.addItem('沿用动作高度','recorded')
        for label,value in [('保持当前','keep'),('正向下蹲','forward'),('反向下蹲','reverse')]:direction.addItem(label,value)
        direction.setCurrentIndex(max(0,direction.findData(p.get('body_mode','recorded' if kind=='action' and posture is None else 'keep'))))
        height=Q.QDoubleSpinBox();height.setDecimals(1);height.setSuffix(' cm')
        def sync_height():
            height.setMaximum(53. if direction.currentData()=='reverse' else 56.)
            recorded=direction.currentData()=='recorded'
            height.setEnabled(not recorded)
            if recorded:
                from .recorded_posture import summary
                text,value=summary(self.owner.studio.records().get(box.currentText()))
                height.setToolTip(text);direction.setToolTip(text)
                blocked=height.blockSignals(True);height.setValue(value);height.blockSignals(blocked)
            else:height.setToolTip('执行步骤前的高度设置；示教动作随后仍按记录的全身关节回放')
        direction.currentIndexChanged.connect(sync_height);height.setValue(p.get('lowering_cm',0));sync_height()
        if kind=='action':box.currentTextChanged.connect(sync_height)
        head=Q.QWidget();head_row=Q.QHBoxLayout(head);head_row.setContentsMargins(2,0,2,0)
        from .step_posture import head_pitch_limits
        enable=Q.QCheckBox('设置');angle=Q.QDoubleSpinBox();angle.setDecimals(1);angle.setRange(*head_pitch_limits());angle.setSuffix('°')
        enable.setChecked(p.get('head_pitch_deg') is not None);angle.setValue(p.get('head_pitch_deg') or 0)
        # Editing is always available; a user edit opts this field into the step.
        if direction.currentData()!='recorded':height.setToolTip('直接输入高度会启用正向下蹲；沿用动作记录时，完整回放记录中的高度变化。')
        height.valueChanged.connect(lambda _:direction.setCurrentIndex(direction.findData('forward')) if direction.currentData()=='keep' else None)
        angle.setToolTip('直接修改角度会自动勾选“设置”；取消勾选则保持当前头部角度。')
        angle.valueChanged.connect(lambda _:enable.setChecked(True))
        head_row.addWidget(enable);head_row.addWidget(angle);head.enable=enable;head.angle=angle
        self.table.setCellWidget(row,2,direction);self.table.setCellWidget(row,3,height);self.table.setCellWidget(row,4,head)
        arm=Q.QComboBox();arm.addItem('原动作','original');arm.addItem('左手（镜像右手）','left');arm.addItem('右手（镜像左手）','right')
        arm.setCurrentIndex(max(0,arm.findData(execution_arm)));arm.setEnabled(kind=='action')
        arm.setToolTip('镜像指定侧的手臂与夹爪，另一手保持当前姿态；腰腿/头部沿用原动作。关联物体的姿态继续按现场物体纠偏。')
        self.table.setCellWidget(row,5,arm);self.table.setColumnWidth(5,160)
        from .action_speed_ui import ActionSpeedCell
        speed=ActionSpeedCell(self.table,action_speed_percent) if kind=='action' else Q.QLabel('—')
        if kind=='tts':
            speed=Q.QCheckBox('播报必须成功');speed.setChecked(required)
            speed.setToolTip('未勾选：播报失败记录提示并继续；勾选：失败后保留任务等待恢复。')
        self.table.setCellWidget(row,6,speed);self.table.setColumnWidth(6,165)
        if kind=='tts':
            box.setStyleSheet('QComboBox::drop-down {width:0px;} QComboBox::down-arrow {image:none;}')
            box.lineEdit().setPlaceholderText('输入要播报的中文文字，最多500字')
        for col in ([2,3,4,5] if kind=='tts' else [5] if kind!='action' else []):
            blank=Q.QLabel('—');blank.setAlignment(C.Qt.AlignCenter);blank.setEnabled(False)
            self.table.setCellWidget(row,col,blank)
        self.table.setRowHeight(row,76)

    def record(self):
        rows=[]
        for i in range(self.table.rowCount()):
            kind=self.table.item(i,0).data(C.Qt.UserRole)
            target='底盘定量动作' if kind=='base' else self.table.cellWidget(i,1).currentText()
            row=dict(kind=kind,target=target)
            if kind=='tts':row['required']=self.table.cellWidget(i,6).isChecked()
            if kind!='tts':
                row['posture']=dict(body_mode=self.table.cellWidget(i,2).currentData(),
                    lowering_cm=self.table.cellWidget(i,3).value(),
                    head_pitch_deg=self.table.cellWidget(i,4).angle.value() if self.table.cellWidget(i,4).enable.isChecked() else None)
            if kind=='base':row['base']=self.table.cellWidget(i,1).value
            if kind=='action':
                row['execution_arm']=self.table.cellWidget(i,5).currentData()
                speed=self.table.cellWidget(i,6).value()
                if speed is not None:row['action_speed_percent']=speed
            rows.append(row)
        return validate(dict(steps=rows))

    def load(self, name):
        self.table.setRowCount(0); self.name.setText('' if name == '新建流程' else name)
        for row in self.library.get(name, {}).get('steps', []): self.add(row['kind'], row['target'],row.get('posture'),row.get('base'),row.get('execution_arm','original'),row.get('action_speed_percent'),row.get('required',False))

    def remove(self):
        if self.table.currentRow() >= 0: self.table.removeRow(self.table.currentRow())

    def move(self, delta):
        i = self.table.currentRow(); j = i + delta
        if not 0 <= j < self.table.rowCount(): return
        try: rows = self.record()['steps']
        except ValueError: return
        rows[i], rows[j] = rows[j], rows[i]; self.table.setRowCount(0)
        for row in rows: self.add(row['kind'], row['target'],row.get('posture'),row.get('base'),row.get('execution_arm','original'),row.get('action_speed_percent'),row.get('required',False))
        self.table.selectRow(j)

    def save(self):
        try:
            name = self.name.text().strip()
            if not name or len(name)>80: raise ValueError('请填写简短的流程名称')
            record = self.record()
            if name in self.library and Q.QMessageBox.question(self, '覆盖流程', '覆盖同名流程？', defaultButton=Q.QMessageBox.No) != Q.QMessageBox.Yes: return
            self.library[name] = record; atomic_json(self.path, self.library)
            self.names.blockSignals(True); self.names.clear(); self.names.addItem('新建流程'); self.names.addItems(sorted(self.library)); self.names.setCurrentText(name); self.names.blockSignals(False)
        except (ValueError, OSError) as exc: Q.QMessageBox.warning(self, '保存失败', str(exc))

    def delete(self):
        name = self.names.currentText()
        if name in self.library and Q.QMessageBox.question(self, '删除流程', '删除“'+name+'”？', defaultButton=Q.QMessageBox.No) == Q.QMessageBox.Yes:
            del self.library[name]; atomic_json(self.path, self.library); self.names.removeItem(self.names.currentIndex())

    def preview_voice(self):
        row=self.table.currentRow()
        if row<0 or self.table.item(row,0).data(C.Qt.UserRole)!='tts':
            Q.QMessageBox.information(self,'试听语音','先选中一条语音播报步骤，再点击试听。');return
        if self.owner.workflow.active:return
        from .task_speech import TaskSpeech
        if not hasattr(self,'voice_preview'):
            self.voice_preview=TaskSpeech(self.owner)
            self.voice_preview.completed.connect(lambda token,ok,text:self.progress.setText(text))
        try:
            self.voice_preview.start('preview',self.table.cellWidget(row,1).currentText())
            self.progress.setText('正在合成并试听，重复文字使用缓存音频')
        except Exception as exc:Q.QMessageBox.warning(self,'语音试听',str(exc))

    def start(self):
        try: self.owner.workflow.start(self.record())
        except Exception as exc: Q.QMessageBox.warning(self, '不能开始', str(exc)); return
        if self.owner.workflow.active: self.accept()


class Workflow(C.QObject):
    started=C.pyqtSignal(str)
    completed=C.pyqtSignal(str)
    def __init__(self, owner):
        super().__init__(owner); self.owner = owner; self.cursor = None; self.phase = ''; self.plan_id = None
        self.actions = {}; self.map = None
        self.failed_phase=''
        self.candidates=[];self.candidate_failures=[];self.candidate_index=0
        self.stationary_only=False;self.dock_candidates=[]
        self.posture_resume=None;self.nav_arrived=False;self.nav_posture_done=True;self.nav_posture_started=False
        owner.studio.completed.connect(self.action_done)
        from .base_sequence_ui import BaseRunner
        self.base=BaseRunner(owner);self.base.completed.connect(self.base_done)
        from .task_speech import TaskSpeech
        self.speech=TaskSpeech(owner);self.speech.completed.connect(self.speech_done)
        self.store=TaskStore(owner.root/'data/task_runtime.json')
        self.worker_request=None
        self.recovery_timer=C.QTimer(self);self.recovery_timer.setInterval(250)
        self.recovery_timer.timeout.connect(self.recover_tick);self.recovery_timer.start()
        C.QTimer.singleShot(0,self.restore_pending)

    @property
    def active(self): return bool(self.cursor and self.cursor.state in ('running','paused'))

    @property
    def pending(self): return bool(self.cursor and self.cursor.pending)

    @property
    def waiting(self): return bool(self.cursor and self.cursor.state=='waiting')

    def channels_idle(self):
        o=self.owner;mobile=getattr(o,'mobile_reach',None);visual=getattr(o.studio,'visual',None)
        return not (o.busy or o.motion or o.cancel_pending or o.demo_active or o.placement_active
                    or o.navigation.inflight or o.studio.moving() or self.base.moving()
                    or (mobile and (mobile.active or mobile.proc)) or (visual and visual.busy)
                    or getattr(o,'parallel_prepare',None) is not None)

    def note_worker_request(self,request_id):
        if self.active:self.worker_request=request_id

    def mark_dispatch(self,phase=None):
        if self.active:
            self.cursor.metadata['action_hold_grippers']=getattr(self,'action_hold_grippers',False)
            self.cursor.dispatch(phase or self.phase)

    def write_context(self):
        self.cursor.phase=self.phase
        self.cursor.metadata.update(map=self.map,actions=self.actions,failed_phase=self.failed_phase,
                                    action_hold_grippers=getattr(self,'action_hold_grippers',False))
        self.cursor.persist()
        atomic_json(self.owner.run/'task_flow.json',dict(run_id=self.cursor.run_id,
            task_id=self.cursor.task_id,cycle_index=self.cursor.cycle_index,
            attempt_id=self.cursor.attempt_id,map=self.map,record=self.cursor.record,
            actions=self.actions,retry_index=self.cursor.index))

    def restore_pending(self):
        if self.cursor is not None:return
        try:
            cursor=Cursor.restore(self.store)
            if cursor is None:return
            cursor.record=validate(cursor.record)
            self.cursor=cursor;self.map=cursor.metadata.get('map');self.actions=cursor.metadata.get('actions',{})
            self.failed_phase=cursor.phase or cursor.metadata.get('failed_phase') or 'recovered'
            self.action_hold_grippers=cursor.metadata.get('action_hold_grippers',False)
            saved_session=cursor.metadata.get('session_name')
            if saved_session and saved_session!=self.owner.run.name:
                cursor.reason+='；原会话资料缺失，需恢复持物来源后才能继续放置，或取消后重新抓取'
                cursor.persist()
            self.phase='waiting';self.worker_request=None
            self.owner.flow_label.setText('任务已保留 · '+cursor.reason)
            self.owner.controls()
        except (ValueError,KeyError,TypeError,OSError) as exc:
            self.owner.flow_label.setText('任务恢复记录不可用，未自动执行：'+str(exc))

    def recover_tick(self):
        if not self.waiting or self.cursor.resume!='automatic':return
        auto=getattr(self.owner,'automatic',None)
        if auto and auto.paused:return
        if time.monotonic()<self.cursor.retry_at or not self.channels_idle():return
        if not self.owner.ready:return
        self.retry_current(automatic=True)

    def cancel(self,reason='用户取消任务'):
        if not self.pending:return
        try:self.cursor.cancel(reason)
        except OSError as exc:self.owner.flow_label.setText('任务已取消；状态保存失败：'+str(exc))
        finally:
            self.phase='';self.worker_request=None;self.speech.stop()

    @property
    def retry_available(self):
        return bool(self.waiting and self.failed_phase and
                    0<=self.cursor.index<len(self.cursor.record['steps']))

    def stopped(self,reason):
        if not self.pending:return
        if hasattr(self,'speech'):self.speech.stop()
        if self.phase!='waiting':self.failed_phase=self.phase
        self.cursor.wait(reason);self.phase='waiting';self.write_context()

    def retry_current(self,*,automatic=False):
        """Explicit retry of this step only. Never resume a stale motor path."""
        o=self.owner
        if getattr(self,'confirming_retry',False):return
        try:
            if not self.retry_available:raise ValueError('没有已停止的步骤可重试')
            if not self.channels_idle():
                raise ValueError('请等待前一次操作完全停止、后台就绪后再重试')
            row=self.cursor.step;kind=row['kind'];scan_only=False;place_only=False
            require_perception(o, {'steps':[row]})
            if kind=='navigate':
                data=o.navigation.data
                if (time.time()-data.get('snapshot_time',0)>2 or not data.get('ready') or not data.get('pose')
                        or data.get('map_fault') or data.get('active_map')!=self.map or row['target'] not in data.get('names',[])):
                    raise ValueError('请先恢复原地图的导航与定位，再重试该航点')
            if kind in ('grasp','cleanup'):
                # 手动取走物品后允许明确重试，不让上次抓取文件永久锁住步骤。
                scan_only=self.failed_phase in ('scan','plan','confirm') and self.cursor.effect=='none'
                detail='重新识别并规划；抓取使用的夹爪应已空出。'
            elif kind=='place':
                from .placement_request import held_grasp
                held_grasp(o.run)
                place_only=self.failed_phase=='place' and self.cursor.effect=='none'
                detail='重新识别桌面并规划放置；请确认物品仍夹稳、未在上次操作中放下。'
            elif kind=='action':detail='从该示教动作的第一帧重新执行，不重复之前的导航或抓取。请确认当前姿态和现场适合重做。'
            elif kind=='base' and self.cursor.effect!='none':
                detail='底盘可能已部分移动。请现场核对本步骤全部位移和转向是否已经完成。仅在停稳并确认实际到达本步最终目标时选“是”，记录本步完成并继续；否则选“否”保留任务，或取消后重新编排剩余位移。'
            elif kind=='base':detail='尚未下发底盘动作，将执行本步骤。'
            else:detail='从当前位置重新规划到该航点，不重复之前已完成的步骤。'
            reconcile=kind=='base' and self.cursor.effect!='none'
            if not automatic:
                token=self.cursor.token
                self.confirming_retry=True
                try:
                    answer=Q.QMessageBox.question(o,'核对底盘终点' if reconcile else '重试当前步骤',
                        f'第{self.cursor.index+1}步 · {KINDS[kind]} · {row["target"]}\n{detail}',
                        defaultButton=Q.QMessageBox.No)
                finally:self.confirming_retry=False
                if (answer!=Q.QMessageBox.Yes or not self.retry_available or
                        self.cursor.token!=token or not self.channels_idle()):return
            if reconcile:
                if automatic:return
                self.cursor.warn('现场已确认底盘完成本步全部位移与转向')
            o.studio.release();o.invalidate_plan();self.plan_id=None;self.posture_resume=None
            if reconcile:
                # Keep the durable unknown-effect checkpoint until complete()
                # atomically advances the index. Never persist a fresh move.
                self.cursor.state='running'
            else:
                self.cursor.retry(confirmed=not automatic)
                self.write_context()
            self.failed_phase='';self.worker_request=None
            auto=getattr(o,'automatic',None)
            if auto and not auto.active and self.cursor.metadata.get('automatic'):
                saved=self.cursor.metadata['automatic']
                auto.active=True;auto.phase='running';auto.paused=False;auto.task_id=self.cursor.task_id
                auto.record=deepcopy(self.cursor.record);auto.actions=self.snapshot_actions(auto.record,self.actions);auto.map=self.map
                for key in ('task_name','mode','limit','interval_s','completed'):setattr(auto,key,saved[key])
            if auto and auto.active:
                auto.run_id=self.cursor.run_id
                if not automatic:
                    auto.paused=False;auto.deferred.clear();auto.deferred_event=None
                    (o.run/'task_pause').unlink(missing_ok=True)
            (o.run/'stop').unlink(missing_ok=True)
            if reconcile:self.advance()
            elif scan_only:
                o.flow_label.setText(f'重试第{self.cursor.index+1}步 · 重新识别与规划')
                self.start_scan();o.controls()
            elif place_only:
                self.start_place();o.controls()
            else:self.next()
        except Exception as exc:
            if self.active:self.fail(str(exc))
            elif automatic:
                self.cursor.wait(str(exc));o.flow_label.setText('任务保留 · '+str(exc));o.controls()
            else:Q.QMessageBox.warning(o,'暂不能重试',str(exc))

    def snapshot_actions(self,record,source=None):
        """Freeze each referenced action once. Navigation and grasp need no library I/O."""
        names=dict.fromkeys(row['target'] for row in record['steps'] if row['kind']=='action')
        if not names:return {}
        actions=self.owner.studio.records(names) if source is None else source
        from .action_studio import full_record
        for name in names:
            if name not in actions:raise ValueError('动作库中没有：'+name)
        return {name:full_record(actions[name]) for name in names}

    def start(self, record, *, confirmed=False, actions_snapshot=None, expected_map=None):
        o = self.owner
        auto=getattr(o,'automatic',None)
        if auto and auto.active and not (confirmed and auto.launching): raise ValueError('自动任务正在运行')
        if o.cancel_pending: raise ValueError('正在停止，请等待完成')
        if getattr(o,'parallel_prepare',None) is not None:raise ValueError('上一准备进程正在退出')
        if self.pending or not self.channels_idle(): raise ValueError('当前任务尚未结束，请先恢复或取消')
        record = validate(record)
        # The automatic session freezes its selected actions before the first cycle.
        self.actions=auto.actions if auto and auto.active else self.snapshot_actions(record,actions_snapshot)
        data = o.navigation.data
        if any(r['kind']=='navigate' for r in record['steps']):
            if expected_map is not None and data.get('active_map')!=expected_map: raise ValueError('地图已变化，请重新选择并启动任务')
            if time.time()-data.get('snapshot_time', 0)>2 or not data.get('ready') or not data.get('pose') or data.get('map_fault'):
                raise ValueError('请点击抓取页“启动导航”，并在“放大地图 / 定位”中完成定位')
            names = data.get('names', [])
            for row in record['steps']:
                if row['kind']=='navigate' and row['target'] not in names: raise ValueError('当前地图没有航点：'+row['target'])
        text = '\n'.join(f'{i+1}. {KINDS[r["kind"]]} · {r["target"]}'+(' · 含身体/头部调姿' if r.get('posture') else '') for i,r in enumerate(record['steps']))
        from .base_sequence import summary
        text+=''.join('\n底盘：'+summary(r['base']) for r in record['steps'] if r['kind']=='base')
        grasp_mode='抓取前请确保所用夹爪可用。放置使用最近完成的抓取记录。'
        if not confirmed and Q.QMessageBox.question(o, '确认任务流程', text+'\n\n将自动执行导航、放置和示教动作（包括腰腿与夹爪）。'+grasp_mode+'\n示教动作没有门阻力检测，请确认现场与示教时一致、全程看护并准备实体急停。', defaultButton=Q.QMessageBox.No) != Q.QMessageBox.Yes: return
        metadata=dict(map=data.get('active_map'),actions=self.actions,session_name=o.run.name)
        if auto and auto.active:
            metadata['automatic']=dict(task_name=auto.task_name,mode=auto.mode,limit=auto.limit,
                interval_s=auto.interval_s,completed=auto.completed)
        o.studio.release(); o.demo.setChecked(False)
        self.cursor=Cursor(record,store=self.store,task_id=getattr(auto,'task_id',None) if auto and auto.active else None,
            cycle_index=auto.completed if auto and auto.active else 0,metadata=metadata)
        self.map=data.get('active_map');self.failed_phase='';self.worker_request=None
        (o.run/'stop').unlink(missing_ok=True)
        self.write_context()
        self.started.emit(self.cursor.run_id)
        self.next()

    def next(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.next):return
        if not self.active: return
        o = self.owner; row = self.cursor.step; self.phase = row['kind']; self.plan_id = None
        self.posture_resume=None;self.nav_arrived=False;self.nav_posture_done=not bool(row.get('posture'));self.nav_posture_started=False
        o.flow_label.setText(f'任务 {self.cursor.index+1}/{len(self.cursor.record["steps"])} · {KINDS[row["kind"]]} · {row["target"]}'+({'left':' · 左手镜像执行','right':' · 右手镜像执行'}.get(row.get('execution_arm'),'')))
        try:
            require_perception(o, {'steps':[row]})
            if row['kind'] in ('base','grasp','cleanup','place'):o.studio.release()
            if row['kind']=='tts':
                try:self.speech.start(self.cursor.token,row['target'])
                except Exception as exc:self.speech_done(self.cursor.token,False,str(exc))
            elif row['kind']=='base':
                if not row['base']['enabled']:self.advance();return
                self.adjust_posture(self.start_base)
            elif row['kind']=='navigate':
                if not o.navigation.proc or o.navigation.proc.state()!=C.QProcess.Running: raise ValueError('导航连接未启动')
                if row.get('posture'):
                    # Confirm hold/SYNC while the base is stationary. Do not
                    # spend the whole navigation leg enabling a new controller.
                    self.phase='prepare_navigation'
                    o.flow_label.setText('出发准备：保持当前关节姿态；出发后同步调姿')
                    self.mark_dispatch('prepare_navigation');o.studio.prepare_posture(self.cursor.token+':nav_prepare')
                else:self.send_navigation()
            elif row['kind']=='action':
                held=o.run/'held_object.json'
                if self.cursor.metadata.get('action_entry_step')==self.cursor.step_id:
                    self.action_hold_grippers=self.cursor.metadata.get('action_hold_grippers',False)
                else:
                    receipt=json.loads(held.read_text('utf-8')) if held.exists() else {}
                    self.action_hold_grippers=receipt.get('status')=='completed' and bool(receipt.get('plan_id'))
                    self.cursor.metadata.update(action_entry_step=self.cursor.step_id,
                                                action_hold_grippers=self.action_hold_grippers)
                    self.cursor.persist()  # Preserve grip policy BEFORE consuming its receipt.
                # A user-authored all-joint action may open the gripper; stale
                # grasp provenance must not be reused for automatic placement.
                (o.run/'held_object.json').unlink(missing_ok=True)
                (o.run/'prepare_result.json').unlink(missing_ok=True)
                self.adjust_posture(self.start_fixed_action)
            elif row['kind']=='cleanup':
                self.adjust_posture(self.start_scan)
            elif row['kind']=='grasp':
                from .grasp_start import requires_reset
                if requires_reset(o.root) and not has_held(o.run):
                    self.phase='prepare';self.mark_dispatch('prepare');o.send('auto_prepare' if getattr(o,'automatic',None) and o.automatic.active else 'prepare')
                else:self.adjust_posture(self.start_scan)
            else:
                last=self.cursor.metadata.get('cleanup_result',{})
                if last.get('index')==self.cursor.index-1 and last.get('picked') is None:
                    self.cursor.warn('本轮清理没有抓到物品，跳过相邻放置步骤')
                    self.advance()
                else:self.adjust_posture(self.start_place)
            o.controls()
        except Exception as exc: self.fail(str(exc))

    def send_navigation(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.send_navigation):return
        if not self.active:return
        self.phase='navigate'
        self.mark_dispatch('navigate')
        self.owner.navigation.command('navigate',name=self.cursor.step['target'],map=self.map,
            run_id=self.cursor.run_id,leg=self.cursor.token,workflow=True)
        self.owner.flow_label.setText('正在出发至 '+self.cursor.step['target'])
        self.owner.controls()

    def adjust_posture(self,resume):
        if defer_call(self.owner,self.adjust_posture,resume):return
        from .step_posture import targets
        if not targets(self.cursor.step.get('posture')):resume();return
        self.posture_resume=resume
        if self.phase!='navigate':self.phase='posture'
        self.owner.flow_label.setText('任务调姿中 · 保持手臂与夹爪；导航步骤会同时行驶')
        self.mark_dispatch('posture')
        self.owner.studio.adjust_posture(self.cursor.token+':posture',self.cursor.step['posture'],
            keep_session=self.cursor.step['kind'] in ('navigate','action'))
        self.owner.controls()

    def start_fixed_action(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_fixed_action):return
        from .action_speed import apply
        record=apply(self.actions[self.cursor.step['target']],self.cursor.step.get('action_speed_percent'))
        if self.cursor.step.get('execution_arm','original')!='original':record['retarget_arm']=self.cursor.step['execution_arm']
        self.phase='action';self.mark_dispatch('action');self.owner.studio.execute(self.cursor.token,record,
            preserve_grippers_at_entry=self.action_hold_grippers,keep_session=True)

    def speech_done(self,token,success,text):
        if not self.active or self.phase!='tts' or token!=self.cursor.token:return
        if success:self.advance()
        elif not self.cursor.step.get('required',False):
            self.cursor.warn(text);self.owner.status.setText('播报未完成，继续任务：'+text);self.advance()
        else:self.fail(text)

    def start_base(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_base):return
        if self.owner.navigation.inflight:raise ValueError('导航尚未停止，未开始定量移动')
        self.phase='base';self.owner.studio.release();self.owner.invalidate_plan();self.owner.live()
        (self.owner.run/'prepare_result.json').unlink(missing_ok=True)
        self.mark_dispatch('base')
        self.base.start(self.cursor.token)

    def base_done(self,token,success,text):
        if not self.active or self.phase!='base' or token!=self.cursor.token:return
        if success:self.advance()
        else:self.fail(text)

    def start_place(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_place):return
        from .placement_request import new_standalone_request
        from .demo_config import load
        o=self.owner;self.phase='place';o.demo_context=new_standalone_request(o.run,load(o.root))
        # Perception/planning are read-only. Hardware is acquired only by the
        # executor after an executable plan exists, so retries cannot reopen jaws.
        self.cursor.effect='none';self.write_context()
        o.placement_active=True;o.demo_phase='planning';o.send('standalone_place')

    def start_scan(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.start_scan):return
        if self.cursor.step['kind']=='cleanup':
            from .cleanup_catalog import library
            self.cleanup_items=[];self.cleanup_failures={}
            catalog=library(self.owner.root);names=catalog['types']
            if not names and not catalog.get('open_waste',False):
                self.finish_cleanup(None,'垃圾种类清单为空，未扫描桌面。')
                return
            self.phase='scan';self.cursor.effect='none';self.write_context()
            self.owner.send('scan_task',cleanup=True,types=names)
            return
        o=self.owner;path=o.root/'data/products.json'
        products=json.loads(path.read_text()).get('products',[]) if path.exists() else []
        from .task_recognition import task_detection_request
        target=self.cursor.step['target'];request=task_detection_request(target,products)
        self.semantic_target=request['semantic']
        self.phase='scan';self.cursor.effect='none';self.write_context();o.send('scan_task',target=target)

    def plan_next_candidate(self):
        if not self.active:return
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.plan_next_candidate):return
        o=self.owner
        rows={item['id']:i for i,item in enumerate(o.items)}
        cleanup=self.cursor.step['kind']=='cleanup'
        if cleanup:
            while self.candidate_index<len(self.candidates) and self.candidates[self.candidate_index] not in rows:
                self.cleanup_failures[self.candidates[self.candidate_index]]='所选物品已离开本轮识别列表'
                self.candidate_index+=1
        # Check every matching object here before requesting a different dock.
        # A new plan request reobserves the selected target and robot state.
        if not cleanup and self.stationary_only and self.candidate_index==len(self.candidates) and self.dock_candidates:
            self.candidates=self.dock_candidates
            self.stationary_only=False;self.candidate_index=0
        if self.candidate_index>=len(self.candidates):
            if cleanup:
                self.finish_cleanup(None)
                return
            reason='；'.join(self.candidate_failures) or '没有可用识别目标'
            atomic_json(self.owner.run/'task_candidate_selection.json',dict(token=self.cursor.token,
                candidates=self.candidates,selected=None,failures=self.candidate_failures))
            raise ValueError('当前物品无法抓取：'+reason)
        instance=self.candidates[self.candidate_index];self.candidate_index+=1
        o.list.setCurrentRow(rows[instance])
        self.phase='plan';o.send('plan',id=instance,stationary_only=self.stationary_only)
        o.flow_label.setText(f'自动选择 {self.cursor.step["target"]} · 候选 {self.candidate_index}/{len(self.candidates)} · 正在规划')
        atomic_json(o.run/'task_candidate_selection.json',dict(token=self.cursor.token,
            candidates=self.candidates,selected=instance,failures=self.candidate_failures))

    def finish_cleanup(self,picked,note=''):
        from .table_cleanup import report,skip_reason
        result=report(self.cleanup_items,self.cleanup_failures,picked)
        if note:result.update(text=note,remaining=None)
        result.update(index=self.cursor.index,step_id=self.cursor.step_id)
        if picked is None:
            atomic_json(self.owner.run/'task_candidate_selection.json',dict(token=self.cursor.token,
                candidates=[v['id'] for v in self.cleanup_items if not skip_reason(v)],
                selected=None,items=result['items'],note=note))
        self.cursor.metadata['cleanup_result']=result
        self.owner.status.setText(result['text'])
        self.advance()

    def navigation_started(self,data):
        if not self.active or self.phase!='navigate' or data.get('run_id')!=self.cursor.run_id or data.get('leg')!=self.cursor.token:return
        if self.nav_posture_started:return
        self.nav_posture_started=True
        try:self.adjust_posture(self.navigation_posture_done)
        except Exception as exc:self.fail(str(exc))

    def navigation_posture_done(self):
        self.nav_posture_done=True
        if self.nav_arrived:self.advance()
        else:self.owner.flow_label.setText('任务姿态已到位 · 继续导航至 '+self.cursor.step['target'])

    def advance(self):
        if getattr(self.owner,'automatic',None) and defer_call(self.owner,self.advance):return
        if not self.active: return
        if self.cursor.index==len(self.cursor.record['steps'])-1:
            try:self.owner.studio.release()
            except Exception as exc:self.fail('结束控制交接：'+str(exc));return
        try:self.cursor.complete(self.cursor.token)
        except OSError as exc:
            self.fail('完成回执保存失败，当前步骤未推进：'+str(exc));return
        # Close the completed phase before the next Qt turn. A duplicate grasp
        # receipt carries a plan_id, not a step token, and must not skip a step.
        self.phase='';self.plan_id=None
        if self.active:
            token=self.cursor.token
            C.QTimer.singleShot(0,lambda:self.next() if self.active and self.cursor.token==token else None)
        else:
            summary=self.cursor.metadata.get('cleanup_result',{}).get('text','')
            self.phase=''; self.owner.flow_label.setText('任务流程已完成'+('\n'+summary if summary else '')); self.owner.controls()
            self.completed.emit(self.cursor.run_id)

    def fail(self, reason,*,retryable=False,await_input=False):
        if not self.active: return
        readonly=self.phase in ('scan','plan','confirm') and self.cursor.effect=='none' and self.channels_idle()
        self.failed_phase=self.phase
        try:
            self.cursor.wait(reason,retryable=retryable,await_input=await_input)
            self.phase='waiting';self.worker_request=None;self.write_context()
        except OSError as exc:
            self.cursor.state='waiting';self.cursor.resume='operator'
            reason=str(reason)+'；任务状态保存失败：'+str(exc)
            self.cursor.reason=reason;self.phase='waiting';self.worker_request=None
        finally:
            if readonly:self.owner.invalidate_plan()
            else:self.owner.stop_motion(reason=reason,source='workflow_wait')
        suffix=' · 将重新识别规划' if self.cursor.resume=='automatic' else ' · 等待确认后恢复'
        self.owner.flow_label.setText('任务已保留：'+reason+suffix);self.owner.controls()

    def worker_event(self, data):
        if not self.active: return
        if 'request_id' in data and data['request_id']!=self.worker_request:return
        o=self.owner; kind=data.get('event')
        try:
            if kind in ('error','fatal','vision_unavailable'):
                if not (worker_required_now(o) or (self.phase=='place' and self.cursor.effect=='none')):return
                if kind=='error' and self.phase=='plan' and data.get('candidate_unavailable'):
                    self.candidate_failures.append(f'#{self.candidates[self.candidate_index-1]}: '+data['text'])
                    if self.cursor.step['kind']=='cleanup':
                        self.cleanup_failures[self.candidates[self.candidate_index-1]]=data['text']
                    elif self.stationary_only and data.get('dock_eligible') is True:
                        self.dock_candidates.append(self.candidates[self.candidate_index-1])
                    self.plan_next_candidate();return
                raise ValueError(data['text'])
            if kind=='reset_planned' and data.get('blockers'): raise ValueError('；'.join(data['blockers']))
            if kind=='scene' and self.phase=='scan':
                if self.cursor.step['kind']=='cleanup':
                    from .table_cleanup import candidates
                    if not (data.get('task_recognition') or {}).get('cleanup'):
                        raise ValueError('清理识别回执与当前步骤不一致')
                    if not data['task_recognition'].get('table_confirmed'):
                        self.finish_cleanup(None,'本轮未确认桌面，已跳过清理。桌上剩余项未知。')
                        return
                    self.cleanup_items=deepcopy(o.items)
                    self.candidates=candidates(self.cleanup_items);self.candidate_index=0
                    self.stationary_only=True;self.dock_candidates=[];self.candidate_failures=[]
                    self.plan_next_candidate();return
                path=o.root/'data/products.json'; products=json.loads(path.read_text()).get('products', []) if path.exists() else []
                target=self.cursor.step['target']
                if data.get('task_recognition') is not None:
                    request=data['task_recognition']
                    if request['target']!=target:raise ValueError('识别结果与当前任务物品不一致')
                    self.semantic_target=request['semantic']
                self.candidates=select_candidates(o.items,target,products,semantic=self.semantic_target)
                self.stationary_only=len(self.candidates)>1 and o.mobile_reach_enabled.isChecked()
                self.dock_candidates=[]
                self.candidate_index=0;self.candidate_failures=[];self.plan_id=None
                self.plan_next_candidate()
            elif kind=='planned' and self.phase=='plan':
                if data.get('blockers'): raise ValueError('；'.join(data['blockers']))
                if o.pending_plan is None: raise ValueError('规划目标校验未通过')
                self.plan_id=data['plan_id']; self.phase='confirm'
                o.flow_label.setText('路径已通过 · 准备自动抓取' if o.auto_grasp_gate.intent else
                    '路径已预览 · 请点击“确认抓取”继续任务流程')
            o.controls()
        except Exception as exc:
            self.fail(str(exc),retryable=self.phase in ('scan','plan','place') and kind not in ('fatal','vision_unavailable'),
                      await_input=isinstance(exc,TargetNotVisible))

    def motion_done(self, result):
        if not self.active: return False
        o=self.owner
        if self.phase=='prepare':
            if result.get('status')!='prepared':
                safe=(result.get('reason_code')=='FEEDBACK_UNAVAILABLE' and result.get('safe_to_retry') is True
                      and result.get('effect_certainty')=='not_dispatched')
                if safe:self.cursor.effect='none'
                self.fail(result.get('reason','复位失败'),await_input=safe);return True
            try:self.adjust_posture(self.start_scan)
            except Exception as exc:self.fail(str(exc))
            return True
        if self.phase=='grasp':
            if result.get('plan_id') not in (None,self.plan_id):
                self.fail('运动回执与本次规划不一致，执行结果待核对');return True
            if self.plan_id and result.get('status')=='completed' and result.get('plan_id')==self.plan_id:
                if self.cursor.step['kind']=='cleanup':self.finish_cleanup(self.candidates[self.candidate_index-1])
                else:self.advance()
            else:
                safe=(result.get('plan_id')==self.plan_id and result.get('safe_to_retry') is True
                      and result.get('effect_certainty')=='not_dispatched')
                if safe:self.cursor.effect='none';self.phase='plan'
                if safe and self.cursor.step['kind']=='cleanup' and result.get('candidate_unavailable') is True:
                    self.cleanup_failures[self.candidates[self.candidate_index-1]]=result.get('reason','本次目标无法执行')
                    self.plan_next_candidate()
                    return True
                self.fail(result.get('reason','抓取未成功，未继续后续任务'),retryable=safe,
                          await_input=safe and result.get('reason_code')=='FEEDBACK_UNAVAILABLE')
            return True
        return False

    def navigation_done(self, data):
        if not self.active or self.phase!='navigate' or data.get('run_id')!=self.cursor.run_id or data.get('leg')!=self.cursor.token: return
        if data.get('success') and data.get('map')==self.map:
            self.nav_arrived=True
            if self.nav_posture_done:self.advance()
            else:self.owner.flow_label.setText('航点已到达 · 等待身体和头部调姿完成')
        else: self.fail(data.get('text','导航失败或地图已变化'))

    def action_done(self, token, success, text):
        if self.active and self.phase=='prepare_navigation' and token==self.cursor.token+':nav_prepare':
            if success:self.send_navigation()
            else:self.fail(text)
            return
        if self.active and token==self.cursor.token+':posture' and self.posture_resume:
            resume=self.posture_resume;self.posture_resume=None
            if not success:self.fail(text);return
            try:resume();self.owner.controls()
            except Exception as exc:self.fail(str(exc))
            return
        if not self.active or self.phase!='action' or token!=self.cursor.token: return
        if success: self.advance()
        else: self.fail(text)
