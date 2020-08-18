from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog


tf = try_import_tf()


def conv_layer(spec, name):
    return tf.keras.layers.Conv2D(
        filters=spec['depth'], kernel_size=spec['kernel'], strides=spec['strides'], padding="same", name=name
    )


def residual_block(x, spec, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == spec['depth']
    x = tf.keras.layers.ReLU()(x)
    first = spec.copy()
    first['kernel'] = 1
    first['strides'] = 1
    x = conv_layer(first, name=prefix + "_conv0")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(spec, name=prefix + "_conv1")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x + inputs


def conv_sequence(x, spec, prefix):
    x = conv_layer(spec, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(x)
    x = residual_block(x, spec, prefix=prefix + "_block0")
    # added ReLU
    x = tf.keras.layers.ReLU()(x)
    x = residual_block(x, spec, prefix=prefix + "_block1")
    # added ReLU
    x = tf.keras.layers.ReLU()(x)
    return x


def conv_core(x):
    from types import SimpleNamespace
    specs = [
        {"depth": 16, "kernel": 3, "strides": 1},
        {"depth": 32, "kernel": 3, "strides": 1},
        {"depth": 32, "kernel": 3, "strides": 1}
    ]
    for i, spec in enumerate(specs):
        x = conv_sequence(x, spec, prefix=f"seq{i}")
    return x

def resnet_core(x):
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    resnet = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling=None
    )
    for layer in resnet.layers:
        layer.trainable = False

    for layer in resnet.layers[-25:]:
        layer.trainable = True
        print("Layer '%s' is trainable" % layer.name)  

    return resnet(x)

def resnet18_core(x):
    from classification_models.keras import Classifiers
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    x = preprocess_input(x)
    resnet18 = ResNet18((224, 224, 3), weights='imagenet', include_top=False)
    for layer in resnet18.layers:
        print("Layer '%s' " % layer.name)  
        layer.trainable = False
    return resnet18(x)


def mobile_core(x):

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    mobile = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        pooling=None
    )

    s = tf.keras.models.Sequential()
    i = 0
    for layer in mobile.layers[:-2]:
        print('adding layer',i, layer)
        i += 1
        s.add(layer)
    for layer in s.layers:
        layer.trainable = False

    return s(x)

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

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        x = tf.cast(inputs, tf.float32) / 255.0

        # conv core
        x = conv_core(x)

        # resnet core
        #x = resnet_core(x)

        # mobile core
        #x = mobile_core(x)

        # densenet core
        # x = densenet_core(x)

        # resnet18 core
        #x = resnet18_core(x)

        # average pooling2d
        #x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # flatten relu
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)

        # dense
        x = tf.keras.layers.Dense(units=512, activation="relu", name="hidden")(x)
        #x = tf.keras.layers.Dense(units=256, activation="relu", name="hidden")(x)

        # outputs
        #print('num_outputs',num_outputs)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)

        # build model
        self.base_model = tf.keras.Model(inputs, [logits, value])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("impala_cnn_tf", ImpalaCNN)
