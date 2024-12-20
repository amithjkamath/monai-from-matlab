# Running inference with a MONAI-trained UNet from within MATLAB

This repository contains a simple example of how one could run inference on a MONAI-trained UNet to segment an image of the heart within MATLAB using the python API. This file describes the setup required to run this code. 

## Create a new conda environment

This code requires the MONAI library to be installed. The easiest way to do this is to create a new conda environment and install the required libraries.

% conda create --name matlab-monai python=3.9   

Make sure the version of python matches one that is supported by your MATLAB version here: https://ch.mathworks.com/support/requirements/python-compatibility.html 

Then activate the environment and find out where the python executable is:

% conda activate matlab-monai
% which python

In my case, for example, it is here:
/Users/uname/opt/anaconda3/envs/matlab-monai/bin/python

## Install python dependencies

Then in the terminal, navigate to the root folder of this repository and install the dependencies of your python code using:

% pip install -r requirements.txt

## Get MATLAB to recognize this environment + libraries

Now start MATLAB and run pyversion with this path (from step 1 above):

>> pyversion('/Users/uname/opt/anaconda3/envs/matlab-monai/bin/python')

And check that pe = pyenv points to the right locations and all the fields make sense. 

Then, we add the custom python file with functions that are called from MATLAB, by adding the module to the ‘path’, this is done using:

>> py.importlib.import_module(‘matlab_monai_bridge’)

this file must be in the current folder, otherwise, you must specify the full path to the module.

In this case, it should return something like this:

ans = 

  Python module with properties:

    ans = 
        Python module with properties:
                     Spacing: [1×1 py.abc.ABCMeta]
              ScaleIntensity: [1×1 py.abc.ABCMeta]
                       torch: [1×1 py.module]
                   LoadImage: [1×1 py.abc.ABCMeta]
                        Norm: [1×1 py.monai.networks.layers.factories.LayerFactory]
                  AsDiscrete: [1×1 py.abc.ABCMeta]
                   save_data: [1×1 py.function]
    sliding_window_inference: [1×1 py.function]
                 Orientation: [1×1 py.abc.ABCMeta]
                create_model: [1×1 py.function]
                  save_model: [1×1 py.function]
               get_transform: [1×1 py.function]
          EnsureChannelFirst: [1×1 py.abc.ABCMeta]
        ...

## Run the code

Now we are ready to run the code from the .mlx file in this repository: "test_MATLAB_MONAI_bridge.mlx". Follow the instructions from there next!