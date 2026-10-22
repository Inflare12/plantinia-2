from __future__ import annotations
import json, math, time
from dataclasses import dataclass, field
from typing import Optional

try:
    import serial
except Exception:
    serial = None


@dataclass
class ArmLimits:
    base: tuple[float,float] = (-160,160)
    shoulder: tuple[float,float] = (-20,140)
    elbow: tuple[float,float] = (-150,150)
    wrist_pitch: tuple[float,float] = (-120,120)
    wrist_roll: tuple[float,float] = (-180,180)

@dataclass
class ArmGeometry:
    upper: float = 120.0
    forearm: float = 120.0
    tool: float = 70.0

@dataclass
class ArmController:
    port: Optional[str] = None
    baud: int = 115200
    dry_run: bool = True
    limits: ArmLimits = field(default_factory=ArmLimits)
    geometry: ArmGeometry = field(default_factory=ArmGeometry)
    tool: str = "gripper"

    def __post_init__(self):
        self.ser = None
        self.joints = {"base":0.0,"shoulder":45.0,"elbow":60.0,"wrist_pitch":-15.0,"wrist_roll":0.0}
        self.estop = False
        if not self.dry_run and serial and self.port:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)

    def _check(self, name, value):
        lo, hi = getattr(self.limits, name)
        if not lo <= value <= hi:
            raise ValueError(f"{name}={value:.1f} outside [{lo},{hi}]")

    def set_joints(self, **angles):
        if self.estop:
            raise RuntimeError("Emergency stop is active")
        for name, value in angles.items():
            self._check(name, float(value))
        self.joints.update({k: float(v) for k,v in angles.items()})
        packet = {"type":"joints","joints":self.joints,"tool":self.tool}
        self._send(packet)
        return packet

    def _send(self, packet):
        if self.dry_run:
            return
        self.ser.write((json.dumps(packet)+"\n").encode())

    def stop(self):
        self.estop = True
        self._send({"type":"stop"})

    def reset_stop(self):
        self.estop = False

    def select_tool(self, tool: str):
        if tool not in {"gripper","fertiliser","mineral","shears"}:
            raise ValueError("Unsupported tool")
        if tool == "shears" and not self.dry_run:
            # The controller must require its own hardware interlock before enabling shears.
            self._send({"type":"tool_interlock_check"})
        self.tool = tool
        self._send({"type":"tool","tool":tool})

    def ik(self, x, y, z, wrist_pitch=0.0):
        """Simple 5-DOF IK for a base-rotating 2-link arm.
        Coordinates are millimetres relative to the shoulder pivot.
        """
        r = math.hypot(x,y)
        base = math.degrees(math.atan2(y,x))
        # Remove tool reach projected into the arm plane.
        pitch = math.radians(wrist_pitch)
        rr = r - self.geometry.tool * math.cos(pitch)
        zz = z - self.geometry.tool * math.sin(pitch)
        a,b = self.geometry.upper,self.geometry.forearm
        c2 = (rr*rr + zz*zz - a*a - b*b)/(2*a*b)
        if not -1 <= c2 <= 1:
            raise ValueError("Target is outside arm workspace")
        elbow = math.acos(c2)
        shoulder = math.atan2(zz,rr) - math.atan2(b*math.sin(elbow), a+b*math.cos(elbow))
        elbow_deg = math.degrees(elbow)
        shoulder_deg = math.degrees(shoulder)
        wrist = wrist_pitch - shoulder_deg - elbow_deg
        angles = {"base":base,"shoulder":shoulder_deg,"elbow":elbow_deg,"wrist_pitch":wrist,"wrist_roll":0.0}
        for n,v in angles.items(): self._check(n,v)
        return angles

    def move_xyz(self, x,y,z,wrist_pitch=0.0):
        return self.set_joints(**self.ik(x,y,z,wrist_pitch))

    def execute_pick_place(self, source, target, tool=None):
        """Generate a conservative pick/place sequence. Hardware motion remains under the arm limits."""
        if tool: self.select_tool(tool)
        sx,sy,sz = source; tx,ty,tz = target
        lift = 45.0
        sequence = [
            (sx,sy,sz+lift), (sx,sy,sz), (sx,sy,sz+lift),
            (tx,ty,tz+lift), (tx,ty,tz), (tx,ty,tz+lift)
        ]
        packets=[]
        for p in sequence:
            packets.append(self.move_xyz(*p))
        return packets
