import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params
# <<< CHANGED LINE: No longer using the simple load_from_checkpoint function
# from utils.torch_utils import load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
        optimizer.zero_grad()
        outputs = net(sample['inputs'].to(device))
        outputs = outputs.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)
        loss = loss_fn['mean'](outputs, ground_truth)
        loss.backward()
        optimizer.step()
        return outputs, ground_truth, loss
  
    def evaluate(net, evalloader, loss_fn, config):
        # This function remains unchanged
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                logits = net(sample['inputs'].to(device))
                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)
                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth
                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())

        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

    #------------------------------------------------------------------------------------------------------------------#
    # CONFIGURATION
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint_path = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)
    copy_yaml(config)
    
    # SETUP MODELS, OPTIMIZER, SCHEDULER
    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    loss_input_fn = get_loss_data_input(config)
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(config, optimizer, num_steps_train)
    writer = SummaryWriter(save_path)

    # <<< MODIFIED BLOCK: This entire block handles loading a full checkpoint for resuming
    start_epoch = 1
    start_global = 1
    BEST_IOU = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        # Handle DataParallel wrapper if the checkpoint was saved with it
        model_state_dict = checkpoint['model_state_dict']
        if len(local_device_ids) > 1 and not isinstance(net, nn.DataParallel):
            net.module.load_state_dict(model_state_dict)
        else:
            net.load_state_dict(model_state_dict)

        # Load optimizer and scheduler states
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state variables
        start_epoch = checkpoint.get('epoch', 1) + 1 # Start from the next epoch
        start_global = checkpoint.get('abs_step', 1)
        BEST_IOU = checkpoint.get('best_iou', 0)
        
        print(f"Resuming training from Epoch {start_epoch}, Step {start_global}")
        print(f"Previous best IOU was {BEST_IOU}")
    else:
        print("No checkpoint found, starting training from scratch.")
    # <<< END OF MODIFIED BLOCK

    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            
            # Train step
            logits, ground_truth, loss = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn=loss_input_fn)
            
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None

            # Print batch statistics
            if abs_step % train_metrics_steps == 0:
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch, step=step)
                write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, ..." % (abs_step, epoch, step + 1, loss, batch_metrics['IOU']))

            # <<< MODIFIED BLOCK: Saving periodic checkpoints
            if abs_step > 0 and abs_step % save_steps == 0:
                # Prepare the full checkpoint dictionary
                checkpoint_data = {
                    'epoch': epoch,
                    'abs_step': abs_step,
                    'model_state_dict': net.state_dict() if len(local_device_ids) <= 1 else net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_iou': BEST_IOU
                }
                save_filename = "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step)
                torch.save(checkpoint_data, save_filename)
                print(f"Saved periodic checkpoint to {save_filename}")
            # <<< END OF MODIFIED BLOCK

            # Evaluate model
            if abs_step > 0 and abs_step % eval_steps == 0:
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
                
                # <<< MODIFIED BLOCK: Saving the best checkpoint
                if eval_metrics[1]['macro']['IOU'] > BEST_IOU:
                    BEST_IOU = eval_metrics[1]['macro']['IOU']
                    print(f"New best IOU found: {BEST_IOU:.4f}. Saving best model.")
                    # Prepare the full checkpoint dictionary
                    checkpoint_data = {
                        'epoch': epoch,
                        'abs_step': abs_step,
                        'model_state_dict': net.state_dict() if len(local_device_ids) <= 1 else net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_iou': BEST_IOU
                    }
                    save_filename = "%s/best.pth" % (save_path)
                    torch.save(checkpoint_data, save_filename)
                    print(f"Saved best checkpoint to {save_filename}")
                # <<< END OF MODIFIED BLOCK

                write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval", optimizer=None)
                net.train()

        scheduler.step_update(abs_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0', type=str, help='gpu ids to use')
    parser.add_argument('--lin', action='store_true', help='train linear classifier only')

    args = parser.parse_args()
    config_file = args.config
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)
    net = get_model(config, device)
    train_and_evaluate(net, dataloaders, config, device)