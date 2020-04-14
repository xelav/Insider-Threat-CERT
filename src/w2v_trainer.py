import torch
import torch.utils.data
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, Metric
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


def create_supervised_trainer_skipgram(model, optimizer, prepare_batch, metrics={},
		device=None,
		log_dir='output/log/',
		checkpoint_dir='output/checkpoints/',
		checkpoint_every=None,
		tensorboard_every=50,
	) -> Engine:

	def _update(engine, batch):
		model.train()
		optimizer.zero_grad()

		batch_loss = model._loss(batch)
		loss = batch_loss.mean()

		loss.backward()
		optimizer.step()

		return {'loss': loss.item(), 'y_pred': scores, 'y': target}

	model.to(device)
	engine = Engine(_update)

	# Metrics
	RunningAverage(output_transform=lambda x: x['loss']).attach(engine, 'average_loss')

	# TQDM
	pbar = ProgressBar(
		persist=True,
	)
	pbar.attach(engine, ['average_loss'])

	# Checkpoint saving
	to_save = {'model': model, 'optimizer': optimizer, 'engine': engine}
	final_checkpoint_handler = Checkpoint(
		{'model': model},
		DiskSaver(checkpoint_dir, create_dir=True),
		n_saved=None,
		filename_prefix='final'
	)

	engine.add_event_handler(Events.COMPLETED, final_checkpoint_handler)

	@engine.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		metrics = engine.state.metrics
		print(f"Epoch results - Avg loss: {metrics['average_loss']:.6f}, Accuracy: {metrics['accuracy']:.6f}, Non-Pad-Accuracy: {metrics['non_pad_accuracy']:.6f}")

	return engine
