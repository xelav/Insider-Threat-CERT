import torch
import torch.utils.data
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage, Metric
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.metrics import ROC_AUC

import wandb


def prepare_batch(batch, device=None, non_blocking=None, num_classes=64):

    actions = batch['actions']
    actions = actions.to(device).to(torch.int64)
    X = actions

    targets = batch['targets'].to(device).long()
    y = targets

    if 'content_topics' in batch:
        topics = batch['content_topics'].to(device)
        X = (actions, topics)
    elif 'conditional_vecs' in batch:
        conditional_vecs = batch['conditional_vecs'].to(device).float()
        X = (actions, conditional_vecs)
    # actions = F.one_hot(actions, num_classes=64).float()
    return X, y


def create_supervised_trainer(model,
                              optimizer,
                              criterion,
                              prepare_batch,
                              metrics={},
                              device=None,
                              tqdm_log=False,
                              ) -> Engine:

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        actions, target = prepare_batch(batch, device=device)

        scores = model(actions)

        loss = criterion(scores, target)
        loss.backward()
        optimizer.step()

        return {'loss': loss.item(), 'y_pred': scores, 'y_true': target}

    model.to(device)
    engine = Engine(_update)

    # Metrics
    RunningAverage(output_transform=lambda x: x['loss'])\
        .attach(engine, 'running_average_loss')
    Loss(
        criterion, output_transform=lambda x: (x['y_pred'], x['y_true']),
    ).attach(engine, 'loss')
    ROC_AUC(
        output_transform=lambda x:
        (F.softmax(x['y_pred'], dim=1)[:, 1], x['y_true'])
    ).attach(engine, 'roc_auc')

    # TQDM
    if tqdm_log:
        pbar = ProgressBar(
            persist=True,
        )
        pbar.attach(engine, ['average_loss'])

    @engine.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        metrics = engine.state.metrics
        print(f"Epoch {engine.state.epoch} completed!")
        print(f"{'Train Results':20} - "
              f"Avg loss: {metrics['loss']:.6f}, "
              f"ROC AUC: {metrics['roc_auc']:.6f}")
        wandb.log({
            "train_loss": metrics['loss'],
            "train_roc_auc": metrics['roc_auc']
        }, commit=False)

    return engine


def create_supervised_evaluator(
        model: torch.nn.Module,
        prepare_batch,
        criterion,
        metrics=None,
        device=None,
        non_blocking: bool = False,
        tqdm_log: bool = False,
        checkpoint_dir='output/checkpoints/'
) -> Engine:

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            actions, target = prepare_batch(
                batch, device=device, non_blocking=non_blocking)
            scores = model(actions)
            return (scores, target)

    engine = Engine(_inference)

    Loss(
        criterion, output_transform=lambda x: x,
    ).attach(engine, 'loss')
    ROC_AUC(
        output_transform=lambda x: (F.softmax(x[0], dim=1)[:, 1], x[1])
    ).attach(engine, 'roc_auc')

    if tqdm_log:
        pbar = ProgressBar(persist=True)
        pbar.attach(engine)

    # save the best model
    # to_save = {'model': model}
    # best_checkpoint_handler = Checkpoint(
    # 	to_save,
    # 	DiskSaver(checkpoint_dir, create_dir=True),
    # 	n_saved=1,
    # 	filename_prefix='best',
    # 	score_function=lambda x: engine.state.metrics['roc_auc'],
    # 	score_name="roc_auc",
    # 	global_step_transform=lambda x, y : engine.train_epoch)
    # engine.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

    @engine.on(Events.COMPLETED)
    def log_validation_results(engine):
        metrics = engine.state.metrics
        print(f"{'Validation Results':20} - "
              f"Avg loss: {metrics['loss']:.6f}, "
              f"ROC AUC: {metrics['roc_auc']:.6f}")
        wandb.log({
            "val_loss": metrics['loss'],
            "val_roc_auc": metrics['roc_auc']
        }, commit=True)

    return engine
