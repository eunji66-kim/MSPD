import torch

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


def compute_psnr(origin, image, data_range):

    if len(origin.shape) == 4:
        origin = origin[0]
    if len(image.shape) == 4:
        image = image[0]
    
    if isinstance(origin, torch.Tensor):
        origin = origin.cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    return peak_signal_noise_ratio(origin, image, data_range=data_range)


def compute_ssim(origin, image, data_range):
    if len(origin.shape) == 4:
        origin = origin[0]
    if len(image.shape) == 4:
        image = image[0]
    
    if len(origin.shape) == 3 and origin.shape[0] == 1:
        origin = origin.squeeze(0)
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    
    if isinstance(origin, torch.Tensor):
        origin = origin.cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    return structural_similarity(origin, image, data_range=data_range)


def compute_rmse(origin, image):
    if len(origin.shape) == 4:
        origin = origin[0]
    if len(image.shape) == 4:
        image = image[0]
    
    if len(origin.shape) == 3 and origin.shape[0] == 1:
        origin = origin.squeeze(0)
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    
    if isinstance(origin, torch.Tensor):
        origin = origin.cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    return mean_squared_error(origin, image)


def compute_lpips(origin, image, loss_fn):
    origin = 2*origin - 1
    image = 2*image - 1

    lpips_score = loss_fn(origin, image)

    return lpips_score.item()
