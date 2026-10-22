# Dataset and model attribution

Plantinia downloads the PlantVillage dataset during bootstrap rather than committing the images to this repository. The upstream dataset contains 54,306 healthy/diseased leaf images across 14 crops and 26 diseases. The upstream/Hugging Face documentation identifies the dataset as CC BY-SA 3.0 and provides the Mohanty et al. (2016) citation.

Plantinia also downloads Qwen2.5-0.5B-Instruct in GGUF format for the local language model. The Qwen model is Apache-2.0 licensed.

Because the vision model is trained from PlantVillage, keep the dataset attribution and applicable ShareAlike terms with any redistributed trained model. Check the current upstream dataset/model licenses before commercial redistribution.
