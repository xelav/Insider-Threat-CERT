from pathlib import Path


def get_params():

	params = {

		'model': {
			'cnn_classifier': {
				'conv1_filters': 32,
				'conv1_kernel_size': 5,

				'conv2_filters': 64,
				'conv2_kernel_size': 5,
				# FIXME:
				'max_seq_length': 200, # implied from seq_len
				'lstm_hidden_size': 40, 
			},

			'lstm_encoder': {
				'input_size': 64, # gets from num_classes
				'hidden_size': 40,
				'num_layers': 3,
				'dropout': 0.5,
			},
		},

		'train': {

			'output_dir': Path(r'C:\Users\admin\Google Drive\Datasets\CERT_output'),
			'answers_dir': Path(r"C:\Users\admin\Google Drive\Datasets\CERT\answers"),

			'lstm_encoder': {
				'num_epochs': 5,
				'learning_rate': 1e-3,
				'batch_size': 16,
			},
			'cnn_classifier': {
				'num_epochs': 100,
				'learning_rate': 1e-3,
				'batch_size': 16,
			}
		},

		'seq_length': 200,

	}

	return params
