import os
import lpips
import wandb

import numpy as np

import torch

import torch.nn

from torch.backends import cudnn
from torch.utils.data import DataLoader

from options.test_options import test_options
from models import create_model
from dataloader import ct_dataset
from utils import load_model, denormalize, apply_window, save_figure, save_image
from measure import compute_psnr, compute_ssim, compute_lpips


def test(opts, model, device):

    load_model(model, opts.result_path, opts.model_name, opts.model_checkpoint)

    test_dataset = ct_dataset(opts.test_path, augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(opts.result_path+'/test/figure', exist_ok=True)
    os.makedirs(opts.result_path+'/test/npy/input', exist_ok=True)
    os.makedirs(opts.result_path+'/test/npy/pred', exist_ok=True)
    os.makedirs(opts.result_path+'/test/npy/target', exist_ok=True)

    project = wandb.init(project=opts.project_name)

    input_psnr, input_ssim, input_lpips = 0.0, 0.0, 0.0
    test_psnr, test_ssim, test_lpips = 0.0, 0.0, 0.0

    windeowed_input_psnr, windeowed_input_ssim = 0.0, 0.0
    windeowed_pred_psnr, windeowed_pred_ssim = 0.0, 0.0

    data_range = opts.max_hu - opts.min_hu

    if opts.lpips_network == 'vgg':
        loss_fn = lpips.LPIPS(net='vgg').to(device)
    else:
        loss_fn = lpips.LPIPS(net='alex').to(device)

    model.eval()

    with torch.no_grad():
        for iter_, (noisy, target) in enumerate(test_loader):
            noisy = noisy.float().to(device)
            target = target.float().to(device)

            output = model(noisy).to(device)
            pred = output.to(device)

            _input_psnr = compute_psnr(target, noisy, data_range=1.0)
            _input_ssim = compute_ssim(target, noisy, data_range=1.0)
            _input_lpips = compute_lpips(target, noisy, loss_fn)

            _test_psnr = compute_psnr(target, pred, data_range=1.0)
            _test_ssim = compute_ssim(target, pred, data_range=1.0)
            _test_lpips = compute_lpips(target, pred, loss_fn)

            project.log({"Test PSNR" : _test_psnr,
                         "Test SSIM" : _test_ssim,
                         "Test LPIPS" : _test_lpips})
            
            input_psnr += _input_psnr
            input_ssim += _input_ssim
            input_lpips += _input_lpips

            test_psnr += _test_psnr
            test_ssim += _test_ssim
            test_lpips += _test_lpips
        
            denormed_input = denormalize(noisy[0].cpu().detach().numpy(), min_value=opts.min_value, max_value=opts.max_value)
            denormed_target = denormalize(target[0].cpu().detach().numpy(), min_value=opts.min_value, max_value=opts.max_value)
            denormed_pred = denormalize(pred[0].cpu().detach().numpy(), min_value=opts.min_value, max_value=opts.max_value)

            windowed_input = apply_window(denormed_input, min_hu=opts.min_hu, max_hu=opts.max_hu)
            windowed_target = apply_window(denormed_target, min_hu=opts.min_hu, max_hu=opts.max_hu)
            windowed_pred = apply_window(denormed_pred, min_hu=opts.min_hu, max_hu=opts.max_hu)

            _windowed_input_psnr = compute_psnr(windowed_target, windowed_input, data_range)
            _windowed_input_ssim = compute_ssim(windowed_target, windowed_input, data_range)

            _windowed_pred_psnr = compute_psnr(windowed_target, windowed_pred, data_range)
            _windowed_pred_ssim = compute_ssim(windowed_target, windowed_pred, data_range)

            windeowed_input_psnr += _windowed_input_psnr
            windeowed_input_ssim += _windowed_input_ssim

            windeowed_pred_psnr += _windowed_pred_psnr
            windeowed_pred_ssim += _windowed_pred_ssim

            save_figure('test', windowed_input, windowed_pred, windowed_target, opts.result_path, iter_, data_range)
            save_image(noisy[0].cpu().detach().numpy(), pred[0].cpu().detach().numpy(), target[0].cpu().detach().numpy(), opts.result_path, iter_)

        input_psnr_avg = input_psnr / len(test_dataset)
        input_ssim_avg = input_ssim / len(test_dataset)
        input_lpips_avg = input_lpips / len(test_dataset)

        test_psnr_avg = test_psnr / len(test_dataset)
        test_ssim_avg = test_ssim / len(test_dataset)
        test_lpips_avg = test_lpips / len(test_dataset)

        windowed_input_psnr_avg = windeowed_input_psnr / len(test_dataset)
        windowed_input_ssim_avg = windeowed_input_ssim / len(test_dataset)

        windowed_pred_psnr_avg = windeowed_pred_psnr / len(test_dataset)
        windowed_pred_ssim_avg = windeowed_pred_ssim / len(test_dataset)

        print('\n')
        print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nLPIPS avg: {:.4f}'.format(input_psnr_avg, input_ssim_avg, input_lpips_avg))
        print('\n')
        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nLPIPS avg: {:.4f}'.format(test_psnr_avg, test_ssim_avg, test_lpips_avg))
        print('\n')
        print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f}'.format(windowed_input_psnr_avg, windowed_input_ssim_avg))
        print('\n')
        print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f}'.format(windowed_pred_psnr_avg, windowed_pred_ssim_avg))
        print('\n')

        with open(os.path.join(opts.result_path, 'average_results.txt'), 'w') as avg_file:
            avg_file.write('Average Results:\n')
            avg_file.write('Original average results:\n')
            avg_file.write(f"PSNR: {input_psnr_avg}, SSIM: {input_ssim_avg}, LPIPS: {input_lpips_avg}\n\n")
            avg_file.write('Predicted average results:\n')
            avg_file.write(f"PSNR: {test_psnr_avg}, SSIM: {test_ssim_avg}, LPIPS: {test_lpips_avg}\n")
            avg_file.write('Original average results:\n')
            avg_file.write(f"PSNR: {windowed_input_psnr_avg}, SSIM: {windowed_input_ssim_avg}\n")
            avg_file.write('Predicted average results:\n')
            avg_file.write(f"PSNR: {windowed_pred_psnr_avg}, SSIM: {windowed_pred_ssim_avg}\n")


if __name__ == '__main__':
    opts = test_options()

    cudnn.benchmark = True
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    model = create_model(opts.model, opts).to(device)

    test(opts, model, device)
