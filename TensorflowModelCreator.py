# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import regularizers

# Usuals
import pandas as pd

sns.set()
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.grid'] = False
tf.compat.v1.enable_eager_execution()

print(tf.__version__)
print(pd.__version__)

# Create a Stratey
class Tensorflow_Model_Generator_Class():

    def __init__(self, ):
        
        # ---------------------------------------------------------------------------- #
        #                     Tensorflow lists of hyperparam types                     #
        # ---------------------------------------------------------------------------- #
        self.initializers = [None,
                             tf.keras.initializers.Constant(),
                             tf.keras.initializers.GlorotNormal(),
                             tf.keras.initializers.GlorotUniform(),
                             tf.keras.initializers.HeNormal(),
                             tf.keras.initializers.HeUniform(),
                             tf.keras.initializers.Identity(),
                             tf.keras.initializers.LecunNormal(),
                             tf.keras.initializers.LecunUniform(),
                             tf.keras.initializers.Ones(),
                             tf.keras.initializers.Orthogonal(),
                             tf.keras.initializers.RandomNormal(),
                             tf.keras.initializers.RandomUniform(),
                             tf.keras.initializers.TruncatedNormal(),
                             tf.keras.initializers.VarianceScaling(),
                             tf.keras.initializers.Zeros(),
                             ]

        self.activators = [None,
                           tf.keras.activations.tanh,
                           tf.keras.activations.softmax,
                           tf.keras.activations.elu,
                           tf.keras.activations.softplus,
                           tf.keras.activations.softsign,
                           tf.keras.activations.relu,
                           tf.keras.activations.sigmoid,
                           tf.keras.activations.hard_sigmoid,
                           tf.keras.activations.linear,
                           tf.keras.activations.exponential,
                           tf.keras.activations.selu,
                           tf.keras.activations.swish,
                           ]

        self.regularizers = [None,
                             tf.keras.regularizers.L1(),
                             tf.keras.regularizers.L1L2(),
                             tf.keras.regularizers.L2()]

        self.constraints = [None,
                            tf.keras.constraints.MaxNorm(),
                            tf.keras.constraints.MinMaxNorm(),
                            tf.keras.constraints.NonNeg(),
                            tf.keras.constraints.RadialConstraint(),
                            tf.keras.constraints.UnitNorm(),
                            ]

        self.optimizers = [tf.optimizers.Nadam(),
                           tf.optimizers.Ftrl(),
                           tf.optimizers.Adam(),
                           tf.optimizers.SGD(),
                           tf.optimizers.RMSprop(),
                           tf.optimizers.Adagrad(),
                           tf.optimizers.Adadelta(),
                           tf.optimizers.Adamax(),
                           ]

        self.dropouts = [None, 0.1, 0.25, 0.5]

        # ---------------------------------------------------------------------------- #
        #                      Initial Tensorflow model gene space                     #
        # ---------------------------------------------------------------------------- #
        self.prelim_gene_space = {

            # Universal hyperparams
            # "epochs": [1000],
            # "early_stopping_patience": [0.1],
            "batch_size": [16, 32, 64, 128, 256, 512],
            "n_layers": [1, 2, 4, 8],
            "n_neurons": [5, 3, 2, 1.5, 1, 0.75, 0.5, 0.25], # multipliers
            "dropout": self.dropouts,
            # "losses": ["mae"],
            # "metrics": ["mae"],
            "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
            "optimizers": self.optimizers,

            # Dense: First Layer
            "first_layer_activation": self.activators,
            "first_layer_use_bias": [True, False],
            "first_layer_kernel_initializer": self.initializers,
            "first_layer_bias_initializer": self.initializers,
            "first_layer_kernel_regularizer": self.regularizers,
            "first_layer_bias_regularizer": self.regularizers,
            "first_layer_activity_regularizer": self.regularizers,
            "first_layer_kernel_constraint": self.constraints,
            "first_layer_bias_constraint": self.constraints,

            # Dense: Subsequent Layer
            "sub_layer_activation": self.activators,
            "sub_layer_use_bias": [True, False],
            "sub_layer_kernel_initializer": self.initializers,
            "sub_layer_bias_initializer": self.initializers,
            "sub_layer_kernel_regularizer": self.regularizers,
            "sub_layer_bias_regularizer": self.regularizers,
            "sub_layer_activity_regularizer": self.regularizers,
            "sub_layer_kernel_constraint": self.constraints,
            "sub_layer_bias_constraint": self.constraints,

            # Dense: Final Layer
            "final_layer_activation": self.activators,
            "final_layer_use_bias": [True, False],
            "final_layer_kernel_initializer": self.initializers,
            "final_layer_bias_initializer": self.initializers,
            "final_layer_kernel_regularizer": self.regularizers,
            "final_layer_bias_regularizer": self.regularizers,
            "final_layer_activity_regularizer": self.regularizers,
            "final_layer_kernel_constraint": self.constraints,
            "final_layer_bias_constraint": self.constraints,

            # # LSTM: First Layer (additionals)
            # "first_layer_recurrent_activation":self.activators,
            # "first_layer_recurrent_initializer":self.initializers,
            # "first_layer_unit_forget_bias":[True, False],
            # "first_layer_recurrent_regularizer":self.regularizers,
            # "first_layer_recurrent_constraint":self.constraints,
            # "first_layer_recurrent_dropout":dropouts,
            # "first_layer_return_sequences":[True, False],
            # "first_layer_return_state":[True, False],
            # "first_layer_go_backwards":[True, False],
            # "first_layer_stateful":[True, False],
            # "first_layer_time_major":[True, False],
            # "first_layer_unroll":[True, False],
            # # GRU: First Layer
            # "first_layer_reset_after":[True, False],

            # # LSTM: Subsequent Layer (additionals)
            # "sub_layer_recurrent_activation":self.activators,
            # "sub_layer_recurrent_initializer":self.initializers,
            # "sub_layer_unit_forget_bias":[True, False],
            # "sub_layer_recurrent_regularizer":self.regularizers,
            # "sub_layer_recurrent_constraint":self.constraints,
            # "sub_layer_recurrent_dropout":dropouts,
            # "sub_layer_return_sequences":[True, False],
            # "sub_layer_return_state":[True, False],
            # "sub_layer_go_backwards":[True, False],
            # "sub_layer_stateful":[True, False],
            # "sub_layer_time_major":[True, False],
            # "sub_layer_unroll":[True, False],
            # # GRU: Subsequent Layer
            # "sub_layer_reset_after":[True, False],

        }

        # Convert lists to final gene space
        self.gene_space = []
        for key in self.prelim_gene_space.keys():
            self.gene_space.append({"low": int(0), "high": int(len(self.prelim_gene_space[key])), "step": int(1)})
        
        self.list_of_keys = [key for key in self.prelim_gene_space.keys()]



    def generate_dense_model(self, current_hyperparams, num_of_training_features, model_output_size):
        """

        :param current_hyperparams:
        :return:
        """
        # Make sure current_hyperparams is all int's re: pygad changes them to floats
        current_hyperparams = [int(a) for a in current_hyperparams]

        # Reconstruct the variables we need, makes it easier to ref variables below
        batch_size = self.prelim_gene_space['batch_size'][current_hyperparams[self.list_of_keys.index('batch_size')]]
        n_layers = self.prelim_gene_space['n_layers'][current_hyperparams[self.list_of_keys.index('n_layers')]]
        n_neurons = self.prelim_gene_space['n_neurons'][current_hyperparams[self.list_of_keys.index('n_neurons')]]
        dropout = self.prelim_gene_space['dropout'][current_hyperparams[self.list_of_keys.index('dropout')]]
        optimizer = self.prelim_gene_space['optimizers'][current_hyperparams[self.list_of_keys.index('optimizers')]]
        learning_rate = self.prelim_gene_space['learning_rate'][current_hyperparams[self.list_of_keys.index('learning_rate')]]
        # Dense
        first_layer_activation = self.prelim_gene_space['first_layer_activation'][
            current_hyperparams[self.list_of_keys.index('first_layer_activation')]]
        first_layer_use_bias = self.prelim_gene_space['first_layer_use_bias'][
            current_hyperparams[self.list_of_keys.index('first_layer_use_bias')]]
        first_layer_kernel_initializer = self.prelim_gene_space['first_layer_kernel_initializer'][
            current_hyperparams[self.list_of_keys.index('first_layer_kernel_initializer')]]
        first_layer_bias_initializer = self.prelim_gene_space['first_layer_bias_initializer'][
            current_hyperparams[self.list_of_keys.index('first_layer_bias_initializer')]]
        first_layer_kernel_regularizer = self.prelim_gene_space['first_layer_kernel_regularizer'][
            current_hyperparams[self.list_of_keys.index('first_layer_kernel_regularizer')]]
        first_layer_bias_regularizer = self.prelim_gene_space['first_layer_bias_regularizer'][
            current_hyperparams[self.list_of_keys.index('first_layer_bias_regularizer')]]
        first_layer_activity_regularizer = self.prelim_gene_space['first_layer_activity_regularizer'][
            current_hyperparams[self.list_of_keys.index('first_layer_activity_regularizer')]]
        first_layer_kernel_constraint = self.prelim_gene_space['first_layer_kernel_constraint'][
            current_hyperparams[self.list_of_keys.index('first_layer_kernel_constraint')]]
        first_layer_bias_constraint = self.prelim_gene_space['first_layer_bias_constraint'][
            current_hyperparams[self.list_of_keys.index('first_layer_bias_constraint')]]
        sub_layer_activation = self.prelim_gene_space['sub_layer_activation'][
            current_hyperparams[self.list_of_keys.index('sub_layer_activation')]]
        sub_layer_use_bias = self.prelim_gene_space['sub_layer_use_bias'][
            current_hyperparams[self.list_of_keys.index('sub_layer_use_bias')]]
        sub_layer_kernel_initializer = self.prelim_gene_space['sub_layer_kernel_initializer'][
            current_hyperparams[self.list_of_keys.index('sub_layer_kernel_initializer')]]
        sub_layer_bias_initializer = self.prelim_gene_space['sub_layer_bias_initializer'][
            current_hyperparams[self.list_of_keys.index('sub_layer_bias_initializer')]]
        sub_layer_kernel_regularizer = self.prelim_gene_space['sub_layer_kernel_regularizer'][
            current_hyperparams[self.list_of_keys.index('sub_layer_kernel_regularizer')]]
        sub_layer_bias_regularizer = self.prelim_gene_space['sub_layer_bias_regularizer'][
            current_hyperparams[self.list_of_keys.index('sub_layer_bias_regularizer')]]
        sub_layer_activity_regularizer = self.prelim_gene_space['sub_layer_activity_regularizer'][
            current_hyperparams[self.list_of_keys.index('sub_layer_activity_regularizer')]]
        sub_layer_kernel_constraint = self.prelim_gene_space['sub_layer_kernel_constraint'][
            current_hyperparams[self.list_of_keys.index('sub_layer_kernel_constraint')]]
        sub_layer_bias_constraint = self.prelim_gene_space['sub_layer_bias_constraint'][
            current_hyperparams[self.list_of_keys.index('sub_layer_bias_constraint')]]
        final_layer_activation = self.prelim_gene_space['final_layer_activation'][
            current_hyperparams[self.list_of_keys.index('final_layer_activation')]]
        final_layer_use_bias = self.prelim_gene_space['final_layer_use_bias'][
            current_hyperparams[self.list_of_keys.index('final_layer_use_bias')]]
        final_layer_kernel_initializer = self.prelim_gene_space['final_layer_kernel_initializer'][
            current_hyperparams[self.list_of_keys.index('final_layer_kernel_initializer')]]
        final_layer_bias_initializer = self.prelim_gene_space['final_layer_bias_initializer'][
            current_hyperparams[self.list_of_keys.index('final_layer_bias_initializer')]]
        final_layer_kernel_regularizer = self.prelim_gene_space['final_layer_kernel_regularizer'][
            current_hyperparams[self.list_of_keys.index('final_layer_kernel_regularizer')]]
        final_layer_bias_regularizer = self.prelim_gene_space['final_layer_bias_regularizer'][
            current_hyperparams[self.list_of_keys.index('final_layer_bias_regularizer')]]
        final_layer_activity_regularizer = self.prelim_gene_space['final_layer_activity_regularizer'][
            current_hyperparams[self.list_of_keys.index('final_layer_activity_regularizer')]]
        final_layer_kernel_constraint = self.prelim_gene_space['final_layer_kernel_constraint'][
            current_hyperparams[self.list_of_keys.index('final_layer_kernel_constraint')]]
        final_layer_bias_constraint = self.prelim_gene_space['final_layer_bias_constraint'][
            current_hyperparams[self.list_of_keys.index('final_layer_bias_constraint')]]

        # # Inspect the current_hyperparams' params
        # print("batch_size:", batch_size,
        #     "n_layers:", n_layers,
        #     "n_neurons:", n_neurons,
        #     "dropout:", dropout,
        #     "optimizer:", optimizer,
        #     "learning_rate:", learning_rate, "\n",
        #     "first_layer_activation:", first_layer_activation,
        #     "first_layer_use_bias:", first_layer_use_bias,
        #     "first_layer_kernel_initializer:", first_layer_kernel_initializer,
        #     "first_layer_bias_regularizer:", first_layer_bias_regularizer,
        #     "first_layer_activity_regularizer:", first_layer_activity_regularizer,
        #     "first_layer_kernel_constraint:", first_layer_kernel_constraint,
        #     "first_layer_bias_constraint:", first_layer_bias_constraint,
        #     "sub_layer_activation:", sub_layer_activation,
        #     "sub_layer_use_bias:", sub_layer_use_bias,
        #     "sub_layer_kernel_initializer:", sub_layer_kernel_initializer,
        #     "sub_layer_bias_initializer:", sub_layer_bias_initializer,
        #     "sub_layer_kernel_regularizer:", sub_layer_kernel_regularizer,
        #     "sub_layer_bias_regularizer:", sub_layer_bias_regularizer,
        #     "sub_layer_activity_regularizer:", sub_layer_activity_regularizer,
        #     "sub_layer_kernel_constraint:", sub_layer_kernel_constraint,
        #     "sub_layer_bias_constraint:", sub_layer_bias_constraint,
        #     "final_layer_activation:", final_layer_activation,
        #     "final_layer_use_bias:", final_layer_use_bias,
        #     "final_layer_kernel_initializer:", final_layer_kernel_initializer,
        #     "final_layer_bias_initializer:", final_layer_bias_initializer,
        #     "final_layer_kernel_regularizer:", final_layer_kernel_regularizer,
        #     "final_layer_bias_regularizer:", final_layer_bias_regularizer,
        #     "final_layer_activity_regularizer:", final_layer_activity_regularizer,
        #     "final_layer_kernel_constraint:", final_layer_kernel_constraint,
        #     "final_layer_bias_constraint:", final_layer_bias_constraint)

        # # Make new dataset re: batch size
        # window = WindowGenerator(input_width=past_lookback_window_timesteps,
        #                          label_width=past_lookback_window_timesteps,
        #                          shift=future_prediction_timesteps,
        #                          label_columns=['Close'],
        #                          batch_size=batch_size)

        # print('Input shape:', window.example[0].shape)

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        # This should be val_loss in future
                                                        patience=5,
                                                        mode='min')

        # The n_layers passed to us is simply describing a shape. it does not
        # know the num_cols we are dealing with, it is simply values of 0.5, 1, and 2,
        # which need to be multiplied by the num_cols
        # UPDATE. CHANGE FROM NUM_COLS TO TOTAL DATAPOINTS!!!!! NOT SURE WHAT I PUT THIS?
        neurons_per_layer = []
        for _ in range(int(n_layers)):
            neurons_per_layer.append(max(int(num_of_training_features * n_neurons), 2))

        # Begin model building
        model = tf.keras.Sequential()

        # Add first layer
        model.add(tf.keras.layers.Dense(neurons_per_layer[0],
                                        activation=first_layer_activation,
                                        # input_dim=train_df.shape[1], # Maybe not needed?
                                        use_bias=first_layer_use_bias,
                                        kernel_initializer=first_layer_kernel_initializer,
                                        bias_initializer=first_layer_bias_initializer,
                                        kernel_regularizer=first_layer_kernel_regularizer,
                                        bias_regularizer=first_layer_bias_regularizer,
                                        activity_regularizer=first_layer_activity_regularizer,
                                        kernel_constraint=first_layer_kernel_constraint,
                                        bias_constraint=first_layer_bias_constraint,
                                        # **kwargs
                                        ))  # , kernel_regularizer=regularizers.l2(0.01)

        # First Dropout
        if dropout != None:
            model.add(tf.keras.layers.Dropout(dropout))

        # Second layer
        if len(neurons_per_layer) <= 2:
            model.add(tf.keras.layers.Dense(neurons_per_layer[-1],
                                            activation=sub_layer_activation,
                                            # input_dim=train_df.shape[1], # Maybe not needed?
                                            use_bias=sub_layer_use_bias,
                                            kernel_initializer=sub_layer_kernel_initializer,
                                            bias_initializer=sub_layer_bias_initializer,
                                            kernel_regularizer=sub_layer_kernel_regularizer,
                                            bias_regularizer=sub_layer_bias_regularizer,
                                            activity_regularizer=sub_layer_activity_regularizer,
                                            kernel_constraint=sub_layer_kernel_constraint,
                                            bias_constraint=sub_layer_bias_constraint,
                                            # **kwargs
                                            ))  # , kernel_regularizer=regularizers.l2(0.01)

            # Second Dropout(s)
            if dropout != None:
                model.add(tf.keras.layers.Dropout(dropout))

        # Subsequent layers
        else:
            for layer_size in neurons_per_layer[1:-1]:
                model.add(tf.keras.layers.Dense(layer_size,
                                                activation=sub_layer_activation,
                                                # input_dim=train_df.shape[1], # Maybe not needed?
                                                use_bias=sub_layer_use_bias,
                                                kernel_initializer=sub_layer_kernel_initializer,
                                                bias_initializer=sub_layer_bias_initializer,
                                                kernel_regularizer=sub_layer_kernel_regularizer,
                                                bias_regularizer=sub_layer_bias_regularizer,
                                                activity_regularizer=sub_layer_activity_regularizer,
                                                kernel_constraint=sub_layer_kernel_constraint,
                                                bias_constraint=sub_layer_bias_constraint,
                                                # **kwargs
                                                ))  # , kernel_regularizer=regularizers.l2(0.01)

            # Final Dropout
            if dropout != None:
                model.add(tf.keras.layers.Dropout(dropout))

        # Final layer
        model.add(tf.keras.layers.Dense(model_output_size, activation=final_layer_activation,
                                        # input_dim=len(train_df.columns.to_list()), # Maybe not needed?
                                        use_bias=final_layer_use_bias,
                                        kernel_initializer=final_layer_kernel_initializer,
                                        bias_initializer=final_layer_bias_initializer,
                                        kernel_regularizer=final_layer_kernel_regularizer,
                                        bias_regularizer=final_layer_bias_regularizer,
                                        activity_regularizer=final_layer_activity_regularizer,
                                        kernel_constraint=final_layer_kernel_constraint,
                                        bias_constraint=final_layer_bias_constraint))

        # Compile
        optimizer.learning_rate = learning_rate
        if learning_rate != None:
            model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=optimizer,
                        metrics=tf.metrics.MeanAbsoluteError())
        else:  # binary_crossentropy, categorical_crossentropy
            model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=optimizer, metrics=tf.metrics.MeanAbsoluteError())

        return model