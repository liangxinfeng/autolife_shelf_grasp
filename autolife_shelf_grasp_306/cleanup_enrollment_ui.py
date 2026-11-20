from PyQt5 import QtGui as G, QtWidgets as Q
from .cleanup_catalog import load_types


class SampleDialog(Q.QDialog):
    def __init__(self,parent):
        super().__init__(parent)
        from .app import ImageView
        self.owner=parent.owner
        self.token=self.selected=None
        self.setWindowTitle('点选录入桌面垃圾')
        self.resize(780,610)
        layout=Q.QVBoxLayout(self)
        self.image=ImageView();layout.addWidget(self.image,1)
        self.image.clicked.connect(self.pick)
        self.note=Q.QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        row=Q.QHBoxLayout();layout.addLayout(row)
        self.capture=Q.QPushButton('重新取景');row.addWidget(self.capture)
        self.capture.clicked.connect(self.freeze)
        self.name=Q.QComboBox();self.name.setEditable(True);self.name.addItems(load_types(self.owner.root))
        self.name.setCurrentIndex(-1)
        self.name.lineEdit().setPlaceholderText('选择已有种类或填写固定名称')
        row.addWidget(self.name,1)
        self.save=Q.QPushButton('保存为垃圾样本');row.addWidget(self.save)
        self.save.clicked.connect(self.commit);self.save.setEnabled(False)

    def freeze(self):
        self.token=self.selected=None
        self.image.boxes=[];self.image.pix=G.QPixmap();self.image.update()
        self.save.setEnabled(False);self.capture.setEnabled(False)
        self.note.setText('正在确认桌面并定格画面…')
        self.owner.send('cleanup_frame')

    def pick(self,x,y):
        if not self.token or self.owner.busy:return
        self.selected=None;self.save.setEnabled(False);self.capture.setEnabled(False)
        self.image.boxes=[];self.image.update()
        self.note.setText('正在分割物体并核对图像位置…')
        self.owner.send('cleanup_point',token=self.token,x=x,y=y)

    def commit(self):
        if self.selected is None or self.owner.busy:return
        name=self.name.currentText().strip()
        if not name:self.note.setText('请填写垃圾种类名称');return
        self.save.setEnabled(False);self.capture.setEnabled(False)
        self.owner.send('cleanup_save',id=self.selected,token=self.token,name=name)

    def receive(self,data):
        self.capture.setEnabled(True)
        kind=data['event']
        if kind=='error':
            self.note.setText(data['text']);self.save.setEnabled(self.selected is not None)
        elif kind=='cleanup_frame':
            self.token=data['token'];self.image.shape=data['shape']
            self.image.pix=G.QPixmap(str(self.owner.run/'cleanup_frame.png'))
            self.image.update();self.note.setText('点击桌面上要录入的垃圾。')
        elif kind=='cleanup_point':
            self.selected=data['id'];self.image.boxes=[data];self.image.selected=self.selected;self.image.update()
            self.save.setEnabled(True)
            self.note.setText(f"本次画面描述为「{data['name']}」。请确认轮廓及要保存的固定种类名称。")
        else:
            self.note.setText(f"已保存「{data['name']}」，该种类现有 {data['count']} 个样本。")
            self.selected=self.token=None
            if self.name.findText(data['name'])<0:self.name.addItem(data['name'])
