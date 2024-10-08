import os
import wandb

import numpy as np

import torch

import torch.nn as nn

from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from options.train_options import train_options
from models import create_model
from dataloader import ct_dataset
from utils import save_model, denormalize, apply_window, save_figure, pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, split_image, combine_images
from measure import compute_psnr


def train(opts, model, device):
    
    train_dataset = ct_dataset(opts.train_path, min_value=opts.min_value, max_value=opts.max_value, augmentation=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=True)

    val_dataset = ct_dataset(opts.val_path, min_value=opts.min_value, max_value=opts.max_value)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers, shuffle=False)

    ratio = opts.epochs / 100
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[
                                         int(20 * ratio) - 1,
                                         int(40 * ratio) - 1,
                                         int(60 * ratio) - 1,
                                         int(80 * ratio) - 1
                                        ],
                                        gamma=opts.gamma)
    
    # scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    # criterion = nn.MSELoss(reduction='sum')
    criterion = nn.MSELoss()

    os.makedirs(opts.result_path, exist_ok=True)
    os.makedirs(opts.result_path+'/validation/figure', exist_ok=True)

    project = wandb.init(project=opts.project_name)

    start_epoch = 1
    best_psnr = 0
    total_iters = 0

    data_range = 1.0


    if opts.resume:
        checkpoint = torch.load('results/latest_model.ckpt')
        model.load_state_dict(checkpoint[opts.model_checkpoint])
        optimizer.load_state_dict(checkpoint[opts.optimizer_checkpoint])
        start_epoch = checkpoint['epoch'] + 1
    
    for epoch in range(start_epoch, opts.epochs+1):
        model.train()
        
        train_loss = 0.0

        for iter_, (noisy, clean) in enumerate(train_loader):
            total_iters += 1

            optimizer.zero_grad()

            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy).to(device)

            loss = criterion(output, noisy)
            loss.backward()
            optimizer.step()

            project.log({"Train Loss" : loss})
            train_loss += loss

        avg_train_loss = train_loss / len(train_dataset)

        save_model(opts.result_path, epoch, model, optimizer, 'latest_model', opts.model_checkpoint, opts.optimizer_checkpoint)
        scheduler.step()

        # validation
        model.eval()
        
        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():
            for val_iter, (val_input, val_target) in enumerate(val_loader):
                val_input = val_input.to(device)
                val_target = val_target.to(device)

                output = model(val_input)
                val_output = output.to(device)
                val_loss_ = criterion(val_output, val_target)
                val_loss += val_loss_

                for i in range(len(val_target)):
                    input, target, pred = val_input[i], val_target[i], val_output[i]

                    psnr = compute_psnr(target, pred, data_range)
                    val_psnr += psnr

        avg_val_loss = val_loss / len(val_dataset)
        avg_val_psnr = val_psnr / len(val_dataset)

        project.log({"Validation PSNR" : avg_val_psnr})
        
        print("EPOCH: {} | Train Loss: {} | Validation Loss: {} | Validation PSNR: {}".format(epoch, avg_train_loss, avg_val_loss, avg_val_psnr))
        
        if best_psnr < avg_val_psnr:
            save_model(opts.result_path, epoch, model, optimizer, 'best_model', opts.model_checkpoint, opts.optimizer_checkpoint)
            best_psnr = avg_val_psnr

        # the last validation image visualization
        denormed_input = denormalize(input.cpu().detach().numpy(), min_value=opts.min_value, max_value=opts.max_value)
        denormed_target = denormalize(target.cpu().detach().numpy(), min_value=opts.min_value, max_value=opts.max_value)
        denormed_pred = denormalize(pred.cpu().detach().numpy(), min_value=opts.min_value, max_value=opts.max_value)
        
        windowed_input = apply_window(denormed_input, min_hu=opts.min_hu, max_hu=opts.max_hu)
        windowed_target = apply_window(denormed_target, min_hu=opts.min_hu, max_hu=opts.max_hu)
        windowed_pred = apply_window(denormed_pred, min_hu=opts.min_hu, max_hu=opts.max_hu)
        save_figure('validation', windowed_input, windowed_pred, windowed_target, opts.result_path, epoch, data_range=opts.max_hu-opts.min_hu)


if __name__ == '__main__':
    opts = train_options()

    cudnn.benchmark = True
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    model = create_model(opts.model, opts)
    model = model.to(device)

    train(opts, model, device)
