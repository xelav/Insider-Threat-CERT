import torch
import torch.utils.data
import torch.nn.functional as F
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver
import wandb

from .utils import AccuracyIgnoringPadding


def prepare_batch_lstm(
        batch,
        device=None,
        non_blocking=None,
        num_classes=64,
        train=True):

    actions = batch['actions']
    actions = actions.to(device).to(torch.int64)

    # prepare additional data
    topics, conditional_vecs = None, None
    if batch.get('content_topics') is not None:
        topics = batch['content_topics'].to(device)
    if batch.get('conditional_vecs') is not None:
        conditional_vecs = batch['conditional_vecs'].to(device).to(torch.float32)

    # TODO: replace tuple with dict
    if train:

        X = actions[:, :-1]
        if topics is not None:
            topics = topics[:, :-1]
            X = (X, topics)
        elif conditional_vecs is not None:
            X = (X, conditional_vecs)
        y = actions[:, 1:]
    else:
        X = actions
        if topics is not None:
            X = (X, topics)
        elif conditional_vecs is not None:
            X = (X, conditional_vecs)
        y = batch['targets']

    return X, y


def prepare_batch_lstm_content(
        batch,
        device=None,
        non_blocking=None,
        num_classes=64,
        train=True):

    actions = batch['actions']
    topics = batch['content_topics']
    actions = actions.to(device).to(torch.int64)
    topics = topics.to(device)
    if train:
        target = actions[:, 1:]
        actions = actions[:, :-1]
        topics = topics[:, :-1]

        return (actions, topics), target
    else:
        return (actions, topics), batch['targets']


def create_supervised_trainer_lstm(model, optimizer, criterion, prepare_batch,
                                   metrics={},
                                   device=None,
                                   tensorboard_dir=None,
                                   checkpoint_dir=None,
                                   checkpoint_every=None,
                                   tensorboard_every=50,
                                   tqdm_log=False
                                   ) -> Engine:

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        actions, target = prepare_batch(batch, device=device)

        scores = model(actions)

        scores = scores.transpose(1, 2)
        # target = target.max(dim=2)[1]

        loss = criterion(scores, target)
        loss.backward()
        optimizer.step()

        return {'loss': loss.item(), 'y_pred': scores, 'y': target}

    model.to(device)
    engine = Engine(_update)

    # Metrics
    RunningAverage(output_transform=lambda x: x[
                   'loss']).attach(engine, 'average_loss')
    Accuracy().attach(engine, 'accuracy')
    AccuracyIgnoringPadding(ignored_class=0).attach(engine, 'non_pad_accuracy')
    Loss(
        criterion, output_transform=lambda x: (x['y_pred'], x['y']),
    ).attach(engine, 'epoch_loss')

    # Checkpoint saving
    to_save = {'model': model, 'optimizer': optimizer, 'engine': engine}
    checkpoint_handler = Checkpoint(
        to_save,
        DiskSaver(
            checkpoint_dir,
            create_dir=True
        ),
        global_step_transform=lambda x, y: engine.state.epoch,
        n_saved=3)
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
        print(f"{'Train Results':20} - Avg loss: {metrics['epoch_loss']:.6f},"
              f" Accuracy: {metrics['accuracy']:.6f},"
              f" Non-Pad-Accuracy: {metrics['non_pad_accuracy']:.6f}")
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
        metrics=None,
        device=None,
        non_blocking: bool = False,
        output_transform=lambda x, y, y_pred: (y_pred, y,),
        tensorboard_dir=None,
        tqdm_log=False,
        checkpoint_dir='output/checkpoints/',
) -> Engine:

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.train()
        with torch.no_grad():
            actions, target = prepare_batch(batch, device=device,
                                            non_blocking=non_blocking)
            scores = model(actions)

            scores = scores.transpose(1, 2)
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

    # save the best model
    to_save = {'model': model}
    best_checkpoint_handler = Checkpoint(
        to_save,
        DiskSaver(checkpoint_dir, create_dir=True),
        n_saved=1,
        filename_prefix='best',
        score_function=lambda x: engine.state.metrics['non_pad_accuracy'],
        score_name="non_pad_accuracy",
        global_step_transform=lambda x, y: engine.train_epoch)
    engine.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

    @engine.on(Events.COMPLETED)
    def log_validation_results(engine):
        metrics = engine.state.metrics
        print(f"{'Validation Results':20} - "
              f"Avg loss: {metrics['epoch_loss']:.6f},"
              f" Accuracy: {metrics['accuracy']:.6f},"
              f" Non-Pad-Accuracy: {metrics['non_pad_accuracy']:.6f}")
        wandb.log({
            "val_loss": metrics['epoch_loss'],
            "val_accuracy": metrics['accuracy'],
            "val_non_pad_accuracy": metrics['non_pad_accuracy']
        })

    return engine
