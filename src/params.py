def get_params():

	params = {

		'cnn_classifier': {
			'conv1_filters': 32,
			'conv1_kernel_size': 5,

			'conv2_filters': 64,
			'conv2_kernel_size': 5,
		},

		'lstm_encoder': {
			'input_size': 64, # gets from num_classes
			'hidden_size': 40,
			'num_layers': 3,
			'dropout': 0.5
		},

		'seq_length': 200,
		'batch_size': 16,

	}

	return params
