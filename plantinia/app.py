from __future__ import annotations
import sys, tempfile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QTextEdit,QFileDialog,QTabWidget,QGroupBox,QFormLayout,QDoubleSpinBox,QComboBox,QMessageBox)
from PySide6.QtGui import QPixmap
from PIL import Image
import cv2
from .vision import PlantVision
from .llm import PlantBrain
from .robot import ArmController
from .sensors import SensorHub

class PlantiniaWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Plantinia Offline"); self.resize(1100,720)
        self.vision=PlantVision(); self.brain=PlantBrain(); self.robot=ArmController(dry_run=True); self.sensors=SensorHub(); self.last_vision=[]
        tabs=QTabWidget(); tabs.addTab(self.diagnose_tab(),"Plant Vision"); tabs.addTab(self.brain_tab(),"Plant Brain"); tabs.addTab(self.soil_tab(),"Soil & Sensors"); tabs.addTab(self.robot_tab(),"5-DOF Robot"); self.setCentralWidget(tabs)

    def diagnose_tab(self):
        w=QWidget(); l=QVBoxLayout(w); self.image=QLabel("Load a plant image or take a camera snapshot"); self.image.setMinimumHeight(360); self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row=QHBoxLayout(); b=QPushButton("Open image"); b.clicked.connect(self.diagnose); c=QPushButton("Camera snapshot"); c.clicked.connect(self.camera_snapshot); row.addWidget(b); row.addWidget(c)
        self.vision_out=QTextEdit(); self.vision_out.setReadOnly(True); l.addWidget(self.image); l.addLayout(row); l.addWidget(self.vision_out); return w

    def _show_result(self, path):
        self.image.setPixmap(QPixmap(path).scaled(650,360,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        self.last_vision=self.vision.predict(Image.open(path)); self.vision_out.setPlainText("\n".join(f"{x['label']}: {x['confidence']:.1%}" for x in self.last_vision))

    def diagnose(self):
        path,_=QFileDialog.getOpenFileName(self,"Plant image",filter="Images (*.jpg *.jpeg *.png *.bmp)")
        if path: self._show_result(path)

    def camera_snapshot(self):
        cap=cv2.VideoCapture(0)
        if not cap.isOpened(): QMessageBox.warning(self,"Camera","Could not open camera."); return
        ok,frame=cap.read(); cap.release()
        if not ok: QMessageBox.warning(self,"Camera","Could not capture a frame."); return
        path=str(Path(tempfile.gettempdir())/"plantinia_camera.jpg"); cv2.imwrite(path,frame); self._show_result(path)

    def brain_tab(self):
        w=QWidget(); l=QVBoxLayout(w); self.question=QTextEdit(); self.question.setPlaceholderText("Ask Plantinia about a plant, disease, watering, nutrition, or the robot..."); self.answer=QTextEdit(); self.answer.setReadOnly(True); b=QPushButton("Ask offline Plant Brain"); b.clicked.connect(lambda:self.answer.setPlainText(self.brain.ask(self.question.toPlainText(),self.last_vision,self.sensors.as_dict()))); l.addWidget(self.question); l.addWidget(b); l.addWidget(self.answer); return w

    def soil_tab(self):
        w=QWidget(); l=QVBoxLayout(w); form=QFormLayout(); self.fields={}
        for name in ["moisture","temperature_c","ph","nitrogen","phosphorus","potassium","light_lux"]:
            s=QDoubleSpinBox(); s.setRange(-100000,100000); s.setValue(0); self.fields[name]=s; form.addRow(name.replace('_',' ').title(),s)
        b=QPushButton("Update sensors"); out=QTextEdit(); out.setReadOnly(True)
        def update(): self.sensors.update(**{k:v.value() for k,v in self.fields.items()}); out.setPlainText("\n".join(self.sensors.recommended_action()))
        b.clicked.connect(update); l.addLayout(form); l.addWidget(b); l.addWidget(out); return w

    def robot_tab(self):
        w=QWidget(); l=QVBoxLayout(w); top=QHBoxLayout(); self.tool=QComboBox(); self.tool.addItems(["gripper","mineral","fertiliser","shears"]); top.addWidget(QLabel("Tool")); top.addWidget(self.tool); select=QPushButton("Select tool"); select.clicked.connect(lambda:self.robot.select_tool(self.tool.currentText())); top.addWidget(select); stop=QPushButton("EMERGENCY STOP"); stop.clicked.connect(self.robot.stop); top.addWidget(stop); l.addLayout(top)
        box=QGroupBox("Target position (mm)"); f=QFormLayout(box); self.xyz=[]
        for n in "XYZ":
            s=QDoubleSpinBox(); s.setRange(-300,300); s.setValue(0 if n!='Z' else 80); self.xyz.append(s); f.addRow(n,s)
        move=QPushButton("Move in dry-run / arm mode"); out=QTextEdit(); out.setReadOnly(True)
        def go():
            try: out.setPlainText(str(self.robot.move_xyz(*(s.value() for s in self.xyz))))
            except Exception as e: QMessageBox.warning(self,"Arm limit",str(e))
        move.clicked.connect(go); l.addWidget(box); l.addWidget(move); l.addWidget(out); l.addWidget(QLabel("Hardware mode is disabled by default. Add a physical emergency stop and controller-side joint limits before connecting motors. Shears require a hardware interlock.")); return w

def run():
    app=QApplication(sys.argv); win=PlantiniaWindow(); win.show(); sys.exit(app.exec())
