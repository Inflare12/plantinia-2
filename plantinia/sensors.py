from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class SoilReading:
    moisture: float | None = None
    temperature_c: float | None = None
    ph: float | None = None
    nitrogen: float | None = None
    phosphorus: float | None = None
    potassium: float | None = None
    light_lux: float | None = None

class SensorHub:
    """Sensor abstraction. Replace/update values from Arduino/ESP32 serial later."""
    def __init__(self):
        self.reading = SoilReading()

    def update(self, **values):
        for k,v in values.items():
            if hasattr(self.reading,k): setattr(self.reading,k, float(v) if v is not None else None)

    def as_dict(self):
        return asdict(self.reading)

    def recommended_action(self):
        r=self.reading
        actions=[]
        if r.moisture is not None and r.moisture < 30: actions.append("soil is dry: consider watering")
        if r.moisture is not None and r.moisture > 85: actions.append("soil is very wet: avoid additional watering")
        if r.ph is not None and (r.ph < 5.5 or r.ph > 7.5): actions.append("check pH before adding amendments")
        if r.nitrogen is not None and r.nitrogen < 20: actions.append("nitrogen reading is low; verify with a calibrated test before fertilising")
        if not actions: actions.append("no automatic soil action suggested from current readings")
        return actions
