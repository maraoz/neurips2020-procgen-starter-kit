#!/usr/bin/env python

from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.utils.framework import try_import_tf

tf = try_import_tf()

class MyPreprocessorClass(Preprocessor):
    """Custom preprocessing for observations

    Adopted from https://docs.ray.io/en/master/rllib-models.html#custom-preprocessors
    """

    last_obs = None

    N_FRAMES = 5

    def _init_shape(self, obs_space, options):
        print('custom_preprocesor')
        print(obs_space, obs_space.shape, type(obs_space.shape))
        new_shape = (self.N_FRAMES,)+ obs_space.shape  # New shape after preprocessing
        print(new_shape)
        print(1/0)
        return new_shape

    def transform(self, obs):
        if self.last_obs is None:
            self.last_obs = obs
            return obs
        delta = (obs - self.last_obs+255)/2
        # import numpy as np
        # print(delta.shape, np.linalg.norm(delta))
        # print(delta)
        self.last_obs = obs
        return delta
