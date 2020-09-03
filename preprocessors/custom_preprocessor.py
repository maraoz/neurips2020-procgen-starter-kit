#!/usr/bin/env python

from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.utils.framework import try_import_tf

import numpy as np


tf = try_import_tf()

class MyPreprocessorClass(Preprocessor):
    """Custom preprocessing for observations

    Adopted from https://docs.ray.io/en/master/rllib-models.html#custom-preprocessors
    """

    def _init_shape(self, obs_space, options):
        print('custom_preprocesor')
        print(obs_space, obs_space.shape, type(obs_space.shape))
        os = obs_space.shape
        new_shape = (os[0],os[1], 1)
        print('new_shape', new_shape)
        return new_shape

    def transform(self, obs):
        #print(type(obs), obs.shape)
        gs = np.dot(obs, [0.2989, 0.5870, 0.1140])
        gs = gs.reshape(64, 64, 1)
        #print(gs, gs.shape, type(gs))
        return gs
