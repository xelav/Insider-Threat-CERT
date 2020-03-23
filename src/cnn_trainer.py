import torch
import torch.utils.data
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, Metric
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver


def prepare_batch(batch, device=None, non_blocking=None, num_classes=64):

	actions = batch['actions']
	# actions = torch.from_numpy(actions).to(device).to(torch.int64)
	actions = actions.to(device).to(torch.int64)
	targets = batch['targets'].to(device).float()
	actions = F.one_hot(actions, num_classes=64).float()
	return actions, targets


def create_supervised_trainer(model, optimizer, criterion, prepare_batch, metrics={},
		device=None,
		log_dir='output/log/',
		checkpoint_dir='output/checkpoints/'
	) -> Engine:

	def _update(engine, batch):
		model.train()
		optimizer.zero_grad()

		actions, target = prepare_batch(batch, device=device)

		scores = model(actions)
		
		loss = criterion(scores[:,0], target)
		loss.backward()
		optimizer.step()

		return {'loss': loss.item(), 'y_pred': scores, 'y_true': target}

	model.to(device)
	engine = Engine(_update)

	# Metrics
	RunningAverage(output_transform=lambda x: x['loss']).attach(engine, 'average_loss')

	# TQDM
	pbar = ProgressBar(persist=True)
	pbar.attach(engine, ['average_loss'])

	# Tensorboard logging
	tb_logger = TensorboardLogger(log_dir=log_dir)
	tb_logger.attach(
		engine,
		log_handler=OutputHandler(
			tag="training", output_transform=lambda x: {"batch_loss": x['loss']}, metric_names="all"
		),
		event_name=Events.ITERATION_COMPLETED(every=1),
	)
	tb_logger.attach(
		engine,
		log_handler=GradsScalarHandler(model, reduction=torch.norm, tag="training/grads"),
		event_name=Events.ITERATION_COMPLETED(every=50)
	)
	tb_logger.attach(
		engine,
		log_handler=GradsHistHandler(model, tag="training/grads"),
		event_name=Events.ITERATION_COMPLETED(every=50))
	tb_logger.attach(
		engine,
		log_handler=OutputHandler(
			tag="training", output_transform=lambda x: {"epoch_loss": x['loss']}
		),
		event_name=Events.EPOCH_COMPLETED,
	)

	# Checkpoint saving
	to_save = {'model': model, 'optimizer': optimizer, 'engine': engine}
	handler = Checkpoint(to_save, DiskSaver(checkpoint_dir, create_dir=True), n_saved=3)
	engine.add_event_handler(Events.ITERATION_COMPLETED(every=1000), handler)

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
			y_pred = model(actions)
			y_pred = y_pred.max(dim=2)[1]
			y_pred = F.one_hot(y_pred, num_classes=target.shape[2]).float()
			# return output_transform(x, y, y_pred)
			return (y_pred, target)

	engine = Engine(_inference)

	# for name, metric in metrics.items():
	# 	metric.attach(engine, name)

	Loss(
		criterion, output_transform=lambda x: x
	).attach(engine, 'loss')
	Accuracy(
		output_transform=lambda x: (x[0].transpose(1,2).contiguous(), x[1].transpose(1,2).contiguous())
	).attach(engine, 'accuracy')

	pbar = ProgressBar(persist=True)
	pbar.attach(engine)

	# Tensorboard logging
	# tb_logger = TensorboardLogger(log_dir=log_dir)
	# tb_logger.attach(
	# 	engine,
	# 	log_handler=OutputHandler(
	# 		tag="validation",metric_names="all"
	# 	),
	# 	event_name=Events.EPOCH_COMPLETED,
	# )

	@engine.on(Events.EPOCH_COMPLETED)
	def log_validation_results(engine):
		metrics = engine.state.metrics
		print(f"Validation Results - Epoch: {engine.state.epoch}  Avg loss: {metrics['loss']:.6f}, Accuracy: {metrics['accuracy']:.6f}")

	return engine