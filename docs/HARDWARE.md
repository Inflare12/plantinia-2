# Cardboard 5-DOF arm interface

## Suggested joints
1. Base rotation
2. Shoulder
3. Elbow
4. Wrist pitch
5. Wrist roll

## Tool changer
The software names four tool states: `gripper`, `mineral`, `fertiliser`, and `shears`. A future controller can map these states to a servo, solenoid, auger, or other mechanism.

## Serial protocol
One JSON object per line. Example:

```json
{"type":"joints","joints":{"base":10,"shoulder":45,"elbow":60,"wrist_pitch":-15,"wrist_roll":0},"tool":"gripper"}
```

Emergency stop:

```json
{"type":"stop"}
```

Tool selection:

```json
{"type":"tool","tool":"fertiliser"}
```

## Sensors
The `SensorHub` accepts moisture, temperature, pH, N/P/K and light readings. An Arduino/ESP32 bridge can later parse its own sensor protocol and call `SensorHub.update()`.

## Safety requirements
- Keep `dry_run=True` until the arm has been tested without motors.
- Add a physical, normally-closed emergency stop that cuts motor power.
- Put hard mechanical stops on every joint.
- Duplicate software joint limits in the microcontroller.
- Do not enable the shears actuator without a physical guard/interlock.
- Start with very low speed and torque on the cardboard prototype.
