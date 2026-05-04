# metanet-rl-calibration

## Setup

1. Set up a Python virtual environment. 
2. Activate virtual environment and run ```pip install -r requirements.txt``` to install requirements and package
3. To do a training run example, you can use the example data from the [Metanet IPOPT repo](https://github.com/woxsao/metanet-calibration/tree/main/examples/example_i24_data). Download/clone if you have not already.
4. In the root of this repository, run  ```./train.sh <path to data>```
5. After training, run ```python eval.py  --base_path <path to data>  --model_dir <path to folder holding model.zip>```