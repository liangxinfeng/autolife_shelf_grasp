"""任务步骤速度：默认沿用，用户明确修改才覆盖该步骤。"""
from PyQt5 import QtWidgets as Q
from .action_speed import percent


class ActionSpeedCell(Q.QWidget):
    def __init__(self,parent=None,value=None):
        super().__init__(parent)
        layout=Q.QVBoxLayout(self);layout.setContentsMargins(2,0,2,0);layout.setSpacing(1)
        self.follow=Q.QCheckBox('沿用动作');self.speed=Q.QDoubleSpinBox()
        self.speed.setRange(25,150);self.speed.setDecimals(0);self.speed.setSingleStep(5)
        self.speed.setSuffix(' %');self.speed.setKeyboardTracking(False)
        self.speed.setValue(100 if value is None else percent(value))
        self.follow.setChecked(value is None);self.speed.setEnabled(value is not None)
        self.follow.toggled.connect(lambda checked:self.speed.setEnabled(not checked))
        layout.addWidget(self.follow);layout.addWidget(self.speed)
        self.setToolTip('与“编动作”整体速度相同，覆盖仅影响此步骤；100%为标准速度，最高150%；连续录制100%为原时序。停顿时间保持不变。')

    def value(self):return None if self.follow.isChecked() else percent(self.speed.value())
