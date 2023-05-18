# Running inference with a MONAI trained-UNet from within MATLAB

This repository contains a simple example of how one could run a UNet (2D for simplicity) to segment an image of the heart within MATLAB using the python API. For this to work, there are a bunch of setup tasks to complete, which are described here.

## Create a new MATLAB conda environment

% conda create --name matlab-env python=3.9   

Make sure the version of python matches one that is supported by your MATLAB version here: https://ch.mathworks.com/support/requirements/python-compatibility.html 

Then activate the environment and find out where the python executable is:

% conda activate Matlab-env
% which python

In my case, for example, it is here:
/Users/amithkamath/opt/anaconda3/envs/matlab-env/bin/python

## Install python dependencies

Then in the terminal, install tensor flow (or whatever dependencies your python code while running in MATLAB has) using:

% conda install <package-name>

To run examples from PyTorch for segmentation, we need both PyTorch and torch vision installed. 

% conda install pytorch torchvision 

If required, also consider installing monai using 

%pip install monai

(If pip isn’t available, you need to set it up, using %conda install pip)

## Get MATLAB to recognize this environment + libraries

Now restart MATLAB and run pyversion with this path (from step 1 above):

>> pyversion('/Users/amithkamath/opt/anaconda3/envs/matlab-env/bin/python')

And check that pe = pyenv points to the right locations and all the fields make sense. 

Then, if you have written a custom python file with functions that are called from MATLAB, you need to add the module to the ‘path’, this is done using:

>> py.importlib.import_module(‘<filename>’)

filename must be in the current folder, or specify the full path to the module.

For example, in this case, it should return something like this:

>> py.importlib.import_module('monaiInference')

ans = 

  Python module with properties:

    createModel: [1×1 py.function]

    <module ... >