import torch
import torch.utils.data
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, Metric
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class AccuracyIgnoringPadding(Accuracy):
	"""
	Same accuracy metric except that it
	ignores single class in calculations.
	This is neccessary so that the metric
	is not overly optimistic
	"""

	def __init__(self, ignored_class, *args, **kwargs):
		self.ignored_class = ignored_class
		super(Accuracy, self).__init__(*args, **kwargs)

	@reinit__is_reduced
	def update(self, output):
		y_pred, y = output

		indices = torch.argmax(y_pred, dim=1)

		mask = (y != self.ignored_class)
		mask &= (indices != self.ignored_class)
		y = y[mask]
		indices = indices[mask]
		correct = torch.eq(indices, y).view(-1)

		self._num_correct += torch.sum(correct).item()
		self._num_examples += correct.shape[0]


def prepare_batch_lstm(batch, device=None, non_blocking=None, num_classes=64, train=True):

	actions = batch['actions']
	# actions = torch.from_numpy(actions).to(device).to(torch.int64)
	actions = actions.to(device).to(torch.int64)
	# actions = F.one_hot(actions, num_classes=64).float()
	if train:
		target = actions[:,1:]
		actions = actions[:,:-1]

		return actions, target
	else:
		return actions, batch['targets']

def prepare_batch_lstm_content(batch, device=None, non_blocking=None, num_classes=64, train=True):

	actions = batch['actions']
	topics = batch['content_topics']
	actions = actions.to(device).to(torch.int64)
	topics = topics.to(device)
	if train:
		target = actions[:,1:]
		actions = actions[:,:-1]
		topics = topics[:,:-1]

		return (actions, topics), target
	else:
		return (actions, topics), batch['targets']


def create_supervised_trainer_lstm(model, optimizer, criterion, prepare_batch, metrics={},
		device=None,
		tensorboard_dir=None,
		checkpoint_dir='output/checkpoints/',
		checkpoint_every=None,
		tensorboard_every=50,
		tqdm_log=False
	) -> Engine:

	def _update(engine, batch):
		model.train()
		optimizer.zero_grad()

		actions, target = prepare_batch(batch, device=device)

		scores = model(actions)
		
		scores = scores.transpose(1,2)
		# target = target.max(dim=2)[1]
		
		loss = criterion(scores, target)
		loss.backward()
		optimizer.step()

		return {'loss': loss.item(), 'y_pred': scores, 'y': target}

	model.to(device)
	engine = Engine(_update)

	# Metrics
	RunningAverage(output_transform=lambda x: x['loss']).attach(engine, 'average_loss')
	Accuracy().attach(engine, 'accuracy')
	AccuracyIgnoringPadding(ignored_class=0).attach(engine, 'non_pad_accuracy')

	# TQDM
	if tqdm_log:
		pbar = ProgressBar(
			persist=True,
		)
		pbar.attach(engine, ['average_loss'])

	# Tensorboard logging
	if tensorboard_dir:
		tb_logger = TensorboardLogger(log_dir=tensorboard_dir + '/train')
		tb_logger.attach(
			engine,
			log_handler=OutputHandler(
				tag="train",
				output_transform=lambda x: {"batch_loss": x['loss']},
				metric_names=['average_loss'],
			),
			event_name=Events.ITERATION_COMPLETED(every=1),
		)
		tb_logger.attach(
			engine,
			log_handler=OutputHandler(
				tag="train",
				output_transform=lambda x: {"epoch_loss": x['loss']},
				metric_names=['non_pad_accuracy', 'accuracy'],
				global_step_transform=global_step_from_engine(engine),
			),
			event_name=Events.EPOCH_COMPLETED,
		)

		tb_logger.attach(
			engine,
			log_handler=GradsScalarHandler(model, reduction=torch.norm, tag="grads"),
			event_name=Events.ITERATION_COMPLETED(every=tensorboard_every)
		)
		tb_logger.attach(
			engine,
			log_handler=GradsHistHandler(model, tag="grads"),
			event_name=Events.ITERATION_COMPLETED(every=tensorboard_every))
	

	# Checkpoint saving
	to_save = {'model': model, 'optimizer': optimizer, 'engine': engine}
	checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoint_dir, create_dir=True), n_saved=3)
	final_checkpoint_handler = Checkpoint(
		{'model': model},
		DiskSaver(checkpoint_dir, create_dir=True),
		n_saved=None,
		filename_prefix='final'
	)

	if checkpoint_every:
		e = Events.ITERATION_COMPLETED(every=checkpoint_every)
	else:
		e = Events.EPOCH_COMPLETED
	engine.add_event_handler(e, checkpoint_handler)
	engine.add_event_handler(Events.COMPLETED, final_checkpoint_handler)

	@engine.on(Events.EPOCH_COMPLETED)
	def log_train_results(engine):
		metrics = engine.state.metrics
		print(f"Epoch {engine.state.epoch} completed!")
		print(f"{'Train Results':20} - Avg loss: {metrics['epoch_loss']:.6f}, Accuracy: {metrics['accuracy']:.6f}, Non-Pad-Accuracy: {metrics['non_pad_accuracy']:.6f}")
		wandb.log({
			"train_loss": metrics['epoch_loss'],
			"train_accuracy": metrics['accuracy'],
			"train_non_pad_accuracy": metrics['non_pad_accuracy']
			})

	return engine


def create_supervised_evaluator_lstm(
	model: torch.nn.Module,
	prepare_batch,
	criterion,
	metrics = None,
	device = None,
	non_blocking: bool = False,
	output_transform = lambda x, y, y_pred: (y_pred, y,),
	tensorboard_dir='output/log/',
	tqdm_log=False,
	checkpoint_dir='output/checkpoints/',
) -> Engine:

	if device:
		model.to(device)

	def _inference(engine, batch):
		model.train()
		with torch.no_grad():
			actions, target = prepare_batch(batch, device=device, non_blocking=non_blocking)
			scores = model(actions)

			scores = scores.transpose(1,2)
			# target = target.max(dim=2)[1]
			# y_pred = y_pred.max(dim=2)[1]
			# y_pred = F.one_hot(y_pred, num_classes=target.shape[2]).float()
			# return output_transform(x, y, y_pred)
			return (scores, target)

	engine = Engine(_inference)

	# Metrics
	Loss(
		criterion, output_transform=lambda x: x
	).attach(engine, 'epoch_loss')
	Accuracy().attach(engine, 'accuracy')
	AccuracyIgnoringPadding(ignored_class=0).attach(engine, 'non_pad_accuracy')

	# TQDM
	if tqdm_log:
		pbar = ProgressBar(persist=True)
		pbar.attach(engine)

	# Tensorboard logging
	if tensorboard_dir:
		tb_logger = TensorboardLogger(log_dir=tensorboard_dir + '/validation')
		tb_logger.attach(
			engine,
			log_handler=OutputHandler(
				tag="validation",
				metric_names="all",
				global_step_transform=lambda x, y : engine.train_epoch,
			),
			event_name=Events.EPOCH_COMPLETED,
		)

	# save the best model
	to_save = {'model': model}
	best_checkpoint_handler = Checkpoint(
		to_save,
		DiskSaver(checkpoint_dir, create_dir=True),
		n_saved=1, 
		filename_prefix='best',
		score_function=lambda x: engine.state.metrics['non_pad_accuracy'],
		score_name="non_pad_accuracy",
		global_step_transform=lambda x, y : engine.train_epoch)
	engine.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

	@engine.on(Events.COMPLETED)
	def log_validation_results(engine):
		metrics = engine.state.metrics
		print(f"{'Validation Results':20} - Avg loss: {metrics['epoch_loss']:.6f}, Accuracy: {metrics['accuracy']:.6f}, Non-Pad-Accuracy: {metrics['non_pad_accuracy']:.6f}")
		wandb.log({
			"val_loss": metrics['epoch_loss'],
			"val_accuracy": metrics['accuracy'],
			"val_non_pad_accuracy": metrics['non_pad_accuracy']
			})

	return engine