# Conditional-Score-Based-Diffusion-Model-for-Cortical-Thickness-Trajectory-Prediction


This repository contains the source code associated with our paper titled "Conditional-Score-Based-Diffusion-Model-for-Cortical-Thickness-Trajectory-Prediction" which has been accepted at MICCAI 2024.

## Requirements

Ensure all the necessary packages listed.

numpy matplotlib scikit-learn scikit-image click requests psutil tqdm imageio imageio-ffmpeg pyspng pillow 



## Running the Training/Experiments

To conduct experiments, please build adn run docker image using the command below. Note that you should adjust the paths and hyperparameters according to your specific requirements:

1. move to the location of source code (where dockerfile is located).
2. Build docker image

```bash
sudo docker build -f ./dockerfile_train_residual -t model_train_residual ./
```

3. Run docker image
```bash
sudo docker run --shm-size=8G --rm --gpus all -v /home/example/:/external/ model_train_residual
```
Note; here "/home/example/" is where source code and dockerfile are located in your GPU server.


4. To perform experimentsn, please build adn run docker image using the command below. 
```bash
sudo docker build -f ./dockerfile_generate_residual -t generate_residual ./
```
```bash
sudo docker run --shm-size=8G --rm --gpus all -v /home/example/:/external/ generate_residual
```



Pretrained model weights :
https://drive.google.com/drive/folders/1MSyKmPCNtZ0z6cP2lIBEso0CdCXFVEF0?usp=sharing
Please contact to author or leave the issue in github, if you have any question on model weights. 
