# A framework for training and testing neural network
This is a training and testing framework of neural network. 

**Dataset Code:** You can add your own dataset and metrics code as the example shown in `./datasets/classify_dataset.py`

**Model Code:** You can add your own model code as the example shown in `./model.py`

**Config Code:** All the dataset, model, loss and evaluation metric are configured by configuration files in `./configs`. You can modify them as the examples shown in `./configs`

**Run Command:** The training command is shown in `./run.sh`. If you want to evaluate your model, you can add `--evaluate` in the command which will turn the `train.py` to evaluation mode.

The code is modified from [PVCNN](https://github.com/mit-han-lab/pvcnn)
