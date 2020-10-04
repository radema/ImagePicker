model_config = {'name': 'sequential_6',
                'layers': [{'class_name': 'KerasLayer',
                            'config': {'name': 'keras_layer_4',
                                       'trainable': False,
                                       'dtype': 'float32',
                                       'handle': 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'}},
                           {'class_name': 'Dropout',
                            'config': {'name': 'dropout_2',
                                       'trainable': True,
                                       'dtype': 'float32',
                                       'rate': 0.2,
                                       'noise_shape': None,
                                       'seed': None}},
                           {'class_name': 'Dense',
                            'config': {'name': 'dense_5',
                                       'trainable': True,
                                       'dtype': 'float32',
                                       'units': 1,
                                       'activation': 'sigmoid',
                                       'use_bias': True,
                                       'kernel_initializer': {'class_name': 'GlorotUniform',
                                                              'config': {'seed': None, 'dtype': 'float32'}},
                                       'bias_initializer': {'class_name': 'Zeros',
                                                            'config': {'dtype': 'float32'}},
                                       'kernel_regularizer': {'class_name': 'L1L2',
                                                              'config': {'l1': 0.0, 'l2': 9.999999747378752e-05}},
                                       'bias_regularizer': None,
                                       'activity_regularizer': None,
                                       'kernel_constraint': None,
                                       'bias_constraint': None}}]}