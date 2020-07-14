from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()


def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )


def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs


def conv_sequence(x, depth, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x

def conv_core(x):
    depths = [16, 32, 64, 128]#, 128, 256]
    for i, depth in enumerate(depths):
        x = conv_sequence(x, depth, prefix=f"seq{i}")
    return x

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

        # manual conv core
        #x = conv_core(x)

        # resnet core
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

        x = resnet(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.TimeDistributed(x)
        # x = tf.keras.layers.LSTM(256)(x)

        # n256 dense 256x256
        n256 = 1
        for i in range(n256):
            x = tf.keras.layers.Dense(
                    units=256, activation="relu", name=f"hidden-{i}")(x)

        # n4 dense 4x4
        n4 = 0
        for i in range(n4):
            x = tf.keras.layers.Dense(units=num_outputs, activation="relu", name=f"pi-{i}")(x)
        logits = tf.keras.layers.Dense(units=num_outputs, name="pi-last")(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)

        self.base_model = tf.keras.Model(inputs, [logits, value])
        # Print model summary
        # print(self.base_model.summary())
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
