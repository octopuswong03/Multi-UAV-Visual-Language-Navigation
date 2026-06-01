
## AirVLN-E Dataset

The annotations of the enriched AirVLN-E dataset can be downloaded [here](https://drive.google.com/drive/folders/1gfnC64NlrFxotAq3Z5Q_-a_UOIOQ-OCD?usp=sharing). 
The simulator can be downloaded from [AirVLN](https://github.com/AirVLN/AirVLN/tree/main)

______________________________________________________________________

## Getting Started

### Prerequisites
- Python 3.8
- Conda Environment

### Installation
```bash
  # clone the repo
  git clone https://github.com/WeichenZh/CityNavAgent.git
  cd CityNavAgent-main

  # Create a virtual environment
  conda create -n citynavagent python=3.8
  conda activate citynavagent

  # install dependencies with pip
  pip install -r requirements.txt
```
The project directory structure is similar to AirVLN, which should be like this:
```
Project_dir/
├── Nav-main/
├── DATA/
│   ├── data
│   │   ├── aerialvln-s
│   │   ├── aerialvln-e
├── ENVs/
│   ├── env_1
│   ├── ...
```

## Setup
______________________________________________________________________
### Dependencies Installation
#### GroundingSAM
Install GroundingSAM following the official instructions at [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything). Then, place the GroundingSAM project under ```./external ``` directory.
Download SAM and GroundingDino weights at [sam_vit_h]([https://github.com/facebookresearch/segment-anything#model-checkpoints](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)) and [swint_ogc](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth).

The ```./external ``` directory structure is like this:
```
external/
├── Grounded_Sam_Lite/
│   ├── grondingdino
│   ├── segment_anything
|   ├── weights
```

#### LM-Nav
Install LM-Nav requirements following the official instructions at [lm_nav](https://github.com/blazejosinski/lm_nav).

The ```./external ``` directory structure is like this:
```
external/
├── Grounded_Sam_Lite/
│   ├── grondingdino
│   ├── segment_anything
|   ├── weights
|   ├── grounded_sam_api.py
├── lm_nav/
│   ├── landmark_extraction.py
│   ├── navigation_graph.py
|   ├── optimal_route.py
|   ├── pipeline.py
|   ├── utils_lm.py
```

#### OpenAI API KEY
Set your OpenAI API KEY in SimRun.py

### Data Preparation
Download memory graphs and AirVLN data at [here](https://drive.google.com/drive/folders/1k8PPf83JEsisCAKXFWRc0xLROloQ8290?usp=drive_link). And put the data under the ```./data``` directory.

The directory should be like this:
```
CityNavAgent/
├── data/
│   ├── gt_by_env
│   ├── mem_graphs
|   ├── mem_graphs_pruned
```

## Inference
Run ```SimRun.py``` :
```
python SimRun.py --Image_Width_RGB 512 --Image_Height_RGB 512 --Image_Width_DEPTH 512 --Image_Height_DEPTH 512
```
______________________________________________________________________

## 🙏 Acknowledgement

We have used code snippets from different repositories, especially from: AirVLN, LM_NAV, GroundingDino, and SAM. We would like to acknowledge and thank the authors of these repositories for their excellent work.
