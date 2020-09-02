from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
import os


tf = try_import_tf()


home = os.getenv('PROJECT_HOME', '/home/aicrowd')
print('home', home)

def conv_layer(spec, name):
    full_name = name + "-" + str(spec['depth']) + "-" + str(spec['kernel']) + "-" +str(spec['strides'])
    return tf.keras.layers.Conv2D(
        filters=spec['depth'], kernel_size=spec['kernel'], strides=spec['strides'], padding="same", name=full_name
        )


def residual_block(x, spec, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == spec['depth']
    first = spec.copy()
    #first['kernel'] = 1
    #first['strides'] = 1
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=first['depth']**2, activation="relu", name=prefix + "_dense")(x)
    x = conv_layer(spec, name=prefix + "_conv1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x + inputs
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(first, name=prefix + "_conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)


def conv_sequence(x, spec, prefix):
    x = conv_layer(spec, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, spec, prefix=prefix + "_block0")
    x = residual_block(x, spec, prefix=prefix + "_block1")
    return x


def conv_core(x):
    # conv_core preprocess
    x = tf.cast(x, tf.float32) / 255.0

    specs = [
        {"depth": 16, "kernel": 3, "strides": 1},
        {"depth": 16, "kernel": 3, "strides": 1},
        {"depth": 16, "kernel": 3, "strides": 1},
        {"depth": 16, "kernel": 3, "strides": 1},
    ]
    for i, spec in enumerate(specs):
        x = conv_sequence(x, spec, prefix=f"seq{i}")
    return x

def resnet_core(x):
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    resnet = tf.keras.applications.ResNet50V2(
        include_top=True,
        weights='imagenet'
    )
    for layer in resnet.layers:
        layer.trainable = False
    remove_n = 1#105+46
    s = tf.keras.models.Model(resnet.input, resnet.layers[-remove_n].output, name='resnet-core')
    for layer in s.layers:
        print('adding layer',layer.name)
    for layer in s.layers:
        layer.trainable = False

    #s.save('/Users/manu/git/neurips2020-procgen-starter-kit/models/small')
    return s(x), resnet

def resnet18_save(x):
    from classification_models.keras import Classifiers
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    x = preprocess_input(x)
    resnet18 = ResNet18((224, 224, 3), weights='imagenet', include_top=False)
    for layer in resnet18.layers:
        print("Layer '%s' " % layer.name)  
        layer.trainable = False
    resnet18.save('/Users/manu/git/neurips2020-procgen-starter-kit/models/resnet18.h5')
    return resnet18(x)


def prune_and_save(model):
    remove_n=22
    s = tf.keras.models.Model(model.input, model.layers[-remove_n].output)
    for layer in s.layers:
        print('adding layer',layer.name)

    s.save('/Users/manu/git/neurips2020-procgen-starter-kit/models/resnet18-stage3.h5')
    print(1/0)


def presaved_core(name):
    def named_core(x):
        fullpath = os.path.join(home, 'models', name+'.h5')
        model = tf.keras.models.load_model(fullpath)
        for layer in model.layers:
            print(name, layer.name)
            layer.trainable = False

        #prune_and_save(model)
        return model(x)
    return named_core

small_core = presaved_core('small')
resnet18_core = presaved_core('resnet18')
resnet18_stage3_core = presaved_core('resnet18-stage3')

def mobile_core(x):

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    mobile = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
    )
    return mobile(x)

def densenet_core(x):
    densenet = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        pooling=None
    )
    return densenet(x)

class ImpalaCNN(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        full = None

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = inputs
        # conv core
        x = conv_core(x)

        # resnet core
        #x, full = resnet_core(x)

        # mobile core
        # x = mobile_core(x)

        # densenet core
        # x = densenet_core(x)

        # resnet18 core
        # x = resnet18_stage3_core(x)

        # average pooling2d
        #x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # small core
        #x = small_core(x)

        # flatten relu
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)

        # dense
        x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)
        #x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)

        # outputs
        #print('num_outputs',num_outputs)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)

        # build model
        self.base_model = tf.keras.Model(inputs, [logits, value])
        for layer in self.base_model.layers:
            print(layer.name)
        self.register_variables(self.base_model.variables)
        if full is not None:
            self.register_variables(full.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("impala_cnn_tf", ImpalaCNN)
