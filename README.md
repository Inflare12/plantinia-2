# Plantinia Offline

Plantinia Offline is a local-first plant intelligence and robotics workstation. It is designed to run without cloud services after the one-time model/dataset setup.

## Core features
- Offline plant vision: classify common plant/disease categories from camera images.
- Offline plant LLM: local GGUF model for plant questions, grounded with the bundled plant knowledge base.
- Plant knowledge base stored locally as JSON.
- One-command model/dataset bootstrap.
- 5-DOF robotic arm controller with inverse kinematics, serial protocol, tool changer and dry-run simulation.
- Mineral/fertiliser dispensing and cutting-shear tool selection.
- Sensor abstraction for soil moisture, temperature, pH, NPK and light sensors.
- Windows desktop application.
- PyInstaller build producing a Windows `.exe`.

## Offline design
The application does not require an account, subscription, API key, ads, or a web backend. Internet is used only by `scripts/bootstrap_models.ps1` to download the selected model weights and training dataset. Once those files exist in `models/` and `data/datasets/`, the app runs locally.

## Hardware
The software supports a 5-DOF arm through a simple serial JSON protocol. The cardboard arm can be replaced later by a real servo/stepper controller without changing the plant/AI layer.

See `docs/HARDWARE.md` and `docs/ARCHITECTURE.md`.

## Build
On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_models.ps1
.\scripts\build_exe.ps1
```

The resulting executable is placed in `dist/PlantiniaOffline/PlantiniaOffline.exe`.
