#!/usr/bin/env python

from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.utils.framework import try_import_tf
tf = try_import_tf()

class MyPreprocessorClass(Preprocessor):
    """Custom preprocessing for observations

    Adopted from https://docs.ray.io/en/master/rllib-models.html#custom-preprocessors
    """

    last_obs = None

    def _init_shape(self, obs_space, options):
        print('custom_preprocesor')
        print(obs_space, obs_space.shape)
        return obs_space.shape  # New shape after preprocessing

    def transform(self, obs):
        if self.last_obs is None:
            self.last_obs = obs
            return obs
        delta = (obs - self.last_obs+255)/2
        self.last_obs = obs
        return delta
