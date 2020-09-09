# Set up
Set up a conda environment with conda_env.yml with the following command:  
`conda create --name bsnn --file conda_env.yml`. If this doesn't work email
Bhargava at bsm92@cornell.edu.

# Training models
Models and logs will only be saved if `main.py` is run with `--name <name>`,
where `<name>` is conceptually an experiment name that used to name the log
file and saved model. Log files will be found under `./log`, saved models under
`./checkpoints`. 

Generally, a training run will look like `python main.py --name <name> -d
<dataset> -m <model> --epochs <epochs> --batch-size <batch-size>`, where
`<dataset>` is one of 'mnist', 'cifar10', or 'svhn' and `model` is one of
'lenet5', 'resnet<num>' (check `models/resnet.py` for details) or 'vgg<num>'
(likewise). 

The code saves checkpoints every 50 epochs. Training can be resumed with the
`--resume` flag. 

## MNIST
training for MNIST requires the `--resize-input` flag to be passed, and
`--normalize` if you are training a lenet5 model
