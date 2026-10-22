# Architecture

```text
Camera/Image ──> MobileNetV3 vision model ──┐
                                             ├──> Plant Brain (local GGUF LLM) ──> advice
Plant KB JSON ───────────────────────────────┘
Soil sensors ────────────────────────────────┘

Plant Brain / rules ──> Robot planner ──> 5-DOF IK ──> serial controller
                                             │
                                             ├── gripper
                                             ├── mineral dispenser
                                             ├── fertiliser dispenser
                                             └── shears (hardware interlock)
```

The AI layers do not directly toggle motors. The robot layer enforces joint limits, has a dry-run mode, and exposes an emergency-stop state. A future microcontroller should enforce the same limits independently.

The vision model is trained locally from PlantVillage data and stored as `models/plantinia_vision.pt`. The language model is a local GGUF file. The knowledge base is a plain JSON file and can be expanded without retraining the LLM.
