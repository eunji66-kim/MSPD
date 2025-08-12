import os
import torch

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from measure import compute_psnr, compute_ssim


def load_model(model, model_path, model_name, checkpoint_name):
    path = os.path.join(model_path, '{}.ckpt'.format(model_name))
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint[checkpoint_name], strict=False)


def save_model(result_path, epoch, model, optimizer, file_name, model_checkpoint, optimizer_checkpoint):
    path = os.path.join(result_path, "{}.ckpt".format(file_name))
    torch.save({
        'epoch': epoch,
        model_checkpoint: model.state_dict(),
        optimizer_checkpoint: optimizer.state_dict()
        }, path)


def denormalize(image, min_value=-1024.0, max_value=3072.0):
    image = image * (max_value - min_value) + min_value

    return image


def apply_window(image, min_hu=-160.0, max_hu=240.0):
    windowed_image = np.clip(image, min_hu, max_hu)

    return windowed_image


def save_figure(save_type, input, pred, target, result_path, epoch, data_range):
    input_psnr = compute_psnr(target, input, data_range)
    input_ssim = compute_ssim(target, input, data_range)

    pred_psnr = compute_psnr(target, pred, data_range)
    pred_ssim = compute_ssim(target, pred, data_range)

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    axs[0].imshow(input.transpose((1, 2, 0)), cmap='gray')
    axs[0].set_title('Input (Noisy image)', fontsize=30)
    axs[0].set_xlabel('PSNR: {:.4f}\nSSIM: {:.4f}'.format(input_psnr, input_ssim))

    axs[1].imshow(pred.transpose((1, 2, 0)), cmap='gray')
    axs[1].set_title('Predicted', fontsize=30)
    axs[1].set_xlabel('PSNR: {:.4f}\nSSIM: {:.4f}'.format(pred_psnr, pred_ssim))
    
    axs[2].imshow(target.transpose((1, 2, 0)), cmap='gray')
    axs[2].set_title('Target', fontsize=30)

    fig.savefig(os.path.join(result_path, "{}/figure/result_{}.png".format(save_type, epoch)))
    plt.close()


def save_image(input, pred, target, result_path, iter_):
    np.save(os.path.join(result_path, f'test/npy/input/input_image_{iter_:03d}'), input)
    np.save(os.path.join(result_path, f'test/npy/pred/pred_image_{iter_:03d}'), pred)
    np.save(os.path.join(result_path, f'test/npy/target/target_image_{iter_:03d}'), target)
