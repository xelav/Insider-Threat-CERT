import torch
import torch.utils.data
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, Metric
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.metrics import ROC_AUC


def prepare_batch(batch, device=None, non_blocking=None, num_classes=64):

	actions = batch['actions']
	actions = actions.to(device).to(torch.int64)
	targets = batch['targets'].to(device).long()
	actions = F.one_hot(actions, num_classes=64).float()
	return actions, targets


def create_supervised_trainer(model, optimizer, criterion, prepare_batch, metrics={},
		device=None,
		log_dir='output/log/',
		checkpoint_dir='output/checkpoints/',
		checkpoint_every=1000,
		tensorboard_every=50,
	) -> Engine:

	def _update(engine, batch):
		model.train()
		optimizer.zero_grad()

		actions, target = prepare_batch(batch, device=device)

		scores = model(actions)
		
		loss = criterion(scores, target.long())
		loss.backward()
		optimizer.step()

		return {'loss': loss.item(), 'y_pred': scores, 'y_true': target}

	model.to(device)
	engine = Engine(_update)

	# Metrics
	RunningAverage(output_transform=lambda x: x['loss']).attach(engine, 'average_loss')

	# TQDM
	pbar = ProgressBar(
		persist=True,
		bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:8.0f}%|{bar}{postfix} [{elapsed}<{remaining}]',
	)
	pbar.attach(engine, ['average_loss'])

	# Tensorboard logging
	tb_logger = TensorboardLogger(log_dir=log_dir + '/train')
	tb_logger.attach(
		engine,
		log_handler=OutputHandler(
			tag="metrics", output_transform=lambda x: {"batch_loss": x['loss']}, metric_names="all"
		),
		event_name=Events.ITERATION_COMPLETED(every=1),
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
	tb_logger.attach(
		engine,
		log_handler=OutputHandler(
			tag="metrics",
			output_transform=lambda x: {"epoch_loss": x['loss']},
			global_step_transform=global_step_from_engine(engine),
		),
		event_name=Events.EPOCH_COMPLETED,
	)

	# Checkpoint saving
	to_save = {'model': model, 'optimizer': optimizer, 'engine': engine}
	checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoint_dir, create_dir=True), n_saved=3)
	final_checkpoint_handler = Checkpoint(
		{'model': model},
		DiskSaver(checkpoint_dir, create_dir=True),
		n_saved=None,
		filename_prefix='final'
	)

	engine.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_every), checkpoint_handler)
	engine.add_event_handler(Events.COMPLETED, final_checkpoint_handler)

	return engine


def create_supervised_evaluator(
	model: torch.nn.Module,
	prepare_batch,
	criterion,
	metrics = None,
	device = None,
	non_blocking: bool = False,
	output_transform = lambda x, y, y_pred: (y_pred, y,),
	log_dir='output/log/',
) -> Engine:

	if device:
		model.to(device)

	def _inference(engine, batch):
		model.eval()
		with torch.no_grad():
			actions, target = prepare_batch(batch, device=device, non_blocking=non_blocking)
			scores = model(actions)
			return (scores, target)

	engine = Engine(_inference)

	Loss(
		criterion, output_transform=lambda x: x,
	).attach(engine, 'loss')
	Accuracy(
		# output_transform=lambda x: (x[0].transpose(1,2).contiguous(), x[1].transpose(1,2).contiguous())
		output_transform=lambda x: (x[0] > 0.5, x[1])
	).attach(engine, 'accuracy')
	ROC_AUC(
		output_transform=lambda x: (F.softmax(x[0], dim=1)[:,1], x[1])
	).attach(engine, 'roc_auc')

	pbar = ProgressBar(persist=True)
	pbar.attach(engine)

	# Tensorboard logging
	tb_logger = TensorboardLogger(log_dir=log_dir + '/validation')
	tb_logger.attach(
		engine,
		log_handler=OutputHandler(
			tag="validation",
			metric_names="all",
			global_step_transform=lambda x, y : engine.train_epoch,
		),
		event_name=Events.EPOCH_COMPLETED,
	)

	@engine.on(Events.COMPLETED)
	def log_validation_results(engine):
		metrics = engine.state.metrics
		print(f"Validation Results - Avg loss: {metrics['loss']:.6f}, Accuracy: {metrics['accuracy']:.6f}, ROC AUC: {metrics['roc_auc']:.6f}")

	return engine