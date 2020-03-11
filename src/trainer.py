import torch.utils.data
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

def prepare_batch_lstm(batch, device=None, non_blocking=None):

	actions, _ = batch
	actions = torch.from_numpy(actions).to(device).to(torch.int64)
	actions = F.one_hot(actions).float()
	target = actions[:,1:]
	actions = actions[:,:-1]

	return actions, target

def create_supervised_trainer_lstm(model, optimizer, criterion, metrics={}, device=None):

	def _update(engine, batch):
		model.train()
		optimizer.zero_grad()

		# TODO: make prepare_batch
		actions, _ = batch
		actions = torch.from_numpy(actions).to(device).to(torch.int64)
		target = model.one_hot_encoder(actions).to(torch.float)

		scores = model(actions)
		
		loss = criterion(scores[:,:-1], target[:,1:])
		loss.backward()
		optimizer.step()

		# engine.pbar.set_description(f'Loss: {loss.item()}')

		return loss.item()

	# def _metrics_transform(output):
	# 	return output[1], output[2]

	model.to(device)
	engine = Engine(_update)

	for name, metric in metrics.items():
		# metric._output_transform = _metrics_transform
		metric.attach(engine, name)

	pbar = ProgressBar(persist=True, bar_format="")
	pbar.attach(engine, ['loss'])

	return engine