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


def load_models(model1, model2, model_path):
    path = os.path.join(model_path, 'best_model.ckpt')
    checkpoint = torch.load(path)

    model1.load_state_dict(checkpoint['model1_state_dict'], strict=False)
    model2.load_state_dict(checkpoint['model2_state_dict'], strict=False)


def save_model(result_path, epoch, model, optimizer, file_name, model_checkpoint, optimizer_checkpoint):
    path = os.path.join(result_path, "{}.ckpt".format(file_name))
    torch.save({
        'epoch': epoch,
        model_checkpoint: model.state_dict(),
        optimizer_checkpoint: optimizer.state_dict()
        }, path)
    

def save_models(result_path, epoch, model1, model2, optimizer1, optimizer2, file_name):
    path = os.path.join(result_path, "{}.ckpt".format(file_name))
    torch.save({
        'epoch': epoch,
        'model1_state_dict': model1.state_dict(),
        'model2_state_dict': model2.state_dict(),
        'optimizer1_state_dict': optimizer1.state_dict(),
        'optimizer2_state_dict': optimizer2.state_dict()
        }, path)


def save_3models(result_path, epoch, model1, model2, model3, optimizer1, optimizer2, optimizer3, file_name):
    path = os.path.join(result_path, "{}.ckpt".format(file_name))
    torch.save({
        'epoch': epoch,
        'model1_state_dict' : model1.state_dict(),
        'model2_state_dict' : model2.state_dict(),
        'model3_state_dict' : model3.state_dict(),
        'optimizer1_state_dict' : optimizer1.state_dict(),
        'optimizer2_state_dict' : optimizer2.state_dict(),
        'optimizer3_state_dict' : optimizer3.state_dict()
        }, path)



def lr_decay(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5


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


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


operation_seed_counter = 0

def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def shift_patch_left(patch):
    return patch[:, :, [1, 3, 0, 2], :, :]


def shift_patch_right(patch):
    return patch[:, :, [2, 0, 3, 1], :, :]


def generate_shifted_images(img):
    n, c, h, w = img.shape

    patches = F.unfold(img, kernel_size=2, stride=2)
    patches = patches.view(n, c, 4, h // 2, w // 2)
    
    shifted_patches1 = shift_patch_right(patches)
    shifted_patches2 = shift_patch_left(patches)
    
    shifted_patches1 = shifted_patches1.view(n, c * 4, h // 2 * w // 2)
    shifted_patches2 = shifted_patches2.view(n, c * 4, h // 2 * w // 2)
    
    img1 = F.fold(shifted_patches1, output_size=(h, w), kernel_size=2, stride=2)
    img2 = F.fold(shifted_patches2, output_size=(h, w), kernel_size=2, stride=2)
    
    return img1, img2


def padr(img):
    pad = 20
    pad_mod = 'reflect'
    img_pad = F.pad(input=img, pad=(pad,pad,pad,pad), mode=pad_mod)
    return img_pad


def padr_crop(img):
    pad = 20
    pad_mod = 'reflect'
    img = F.pad(input=img[:,:,pad:-pad,pad:-pad], pad=(pad,pad,pad,pad), mode=pad_mod)
    return img


def std(img, window_size=7):
    assert window_size % 2 == 1
    pad = window_size // 2

    # calculate std on the mean image of the color channels
    img = torch.mean(img, dim=1, keepdim=True)
    N, C, H, W = img.shape
    img = nn.functional.pad(img, [pad] * 4, mode='reflect')
    img = nn.functional.unfold(img, kernel_size=window_size)
    img = img.view(N, C, window_size * window_size, H, W)
    img = img - torch.mean(img, dim=2, keepdim=True)
    img = img * img
    img = torch.mean(img, dim=2, keepdim=True)
    img = torch.sqrt(img)
    img = img.squeeze(2)

    return img


def generate_alpha(input, lower=1, upper=5):
    N, C, H, W = input.shape
    ratio = input.new_ones((N, 1, H, W)) * 0.5
    input_std = std(input)
    ratio[input_std < lower] = torch.sigmoid((input_std - lower))[input_std < lower]
    ratio[input_std > upper] = torch.sigmoid((input_std - upper))[input_std > upper]
    ratio = ratio.detach()

    return ratio


def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)
    

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def split_image(x: torch.Tensor, f: int):
    '''
    Split a tensor image into smaller images based on the factor.
    
    Args:
        x (Tensor) : input tensor with shape (C, H, W) or (B, C, H, W)
        f (int) : factor by which to split the image

    Returns:
        list of Tensor: list of smaller image tensors
    '''
    # single image tensor
    if len(x.shape) == 3:
        c, h, w = x.shape
        split_images = []
        
        split_h, split_w = h // f, w // f
        
        for i in range(f):
            for j in range(f):
                split_images.append(x[:, i * split_h:(i + 1) * split_h, j * split_w:(j + 1) * split_w])
        
        return split_images
    
    # batched image tensor
    elif len(x.shape) == 4:
        b, c, h, w = x.shape
        split_images = []
        
        split_h, split_w = h // f, w // f
        
        for i in range(f):
            for j in range(f):
                split_images.append(x[:, :, i * split_h:(i + 1) * split_h, j * split_w:(j + 1) * split_w])
        
        return split_images


def combine_images(split_images: list, f: int):
    '''
    Combine a list of smaller image tensors into one large image.
    
    Args:
        split_images (list) : list of image tensors to be combined
        f (int) : factor by which the original image was split
    
    Returns:
        Tensor: combined image tensor
    '''
    # Combine a list of smaller images into a single image
    # Assuming images are split using split_image function
    rows = []
    for i in range(f):
        row = torch.cat(split_images[i * f:(i + 1) * f], dim=-1)  # Combine along width
        rows.append(row)
    combined_image = torch.cat(rows, dim=-2)  # Combine along height
    return combined_image


def calculate_differences(denoised_splited, splited):
    # 두 리스트의 길이가 동일한지 확인
    assert len(denoised_splited) == len(splited), "두 리스트의 길이가 같아야 합니다."
    
    # 차이를 저장할 리스트 초기화
    differences = []
    
    # 각 위치의 텐서들 간의 차이 계산
    for denoised_img, img in zip(denoised_splited, splited):
        difference = denoised_img - img  # 차이 계산
        differences.append(difference)
    
    return differences


def square_tensors(tensor_list):
    # 리스트의 각 텐서에 대해 제곱을 계산
    squared_tensors = [tensor ** 2 for tensor in tensor_list]

    # 제곱된 텐서들을 하나로 병합
    combined_tensor = torch.cat([t.view(-1) for t in squared_tensors])
    
    return combined_tensor


class Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all'):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def train(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n,self.width**2,c,h,w), device=img.device)
        masks = torch.zeros((n,self.width**2,1,h,w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:,i,...] = x
            masks[:,i,...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask(img, width=4, mask_type='random'):
    # This function generates random masks with shape (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                       dtype=torch.int64,
                       device=img.device)
    idx_list = torch.arange(
        0, width**2, 1, dtype=torch.int64, device=img.device)
    rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                         dtype=torch.int64,
                         device=img.device)

    if mask_type == 'random':
        torch.randint(low=0,
                      high=len(idx_list),
                      size=(n * h // width * w // width, ),
                      device=img.device,
                      generator=get_generator(device=img.device),
                      out=rd_idx)
    elif mask_type == 'batch':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(n, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(h // width * w // width)
    elif mask_type == 'all':
        rd_idx = torch.randint(low=0,
                               high=len(idx_list),
                               size=(1, ),
                               device=img.device,
                               generator=get_generator(device=img.device)).repeat(n * h // width * w // width)
    elif 'fix' in mask_type:
        index = mask_type.split('_')[-1]
        index = torch.from_numpy(np.array(index).astype(
            np.int64)).type(torch.int64)
        rd_idx = index.repeat(n * h // width * w // width).to(img.device)

    rd_pair_idx = idx_list[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,
                                dtype=torch.int64,
                                device=img.device)

    mask[rd_pair_idx] = 1

    mask = depth_to_space(mask.type_as(img).view(
        n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

    return mask


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def interpolate_mask(tensor, mask, mask_inv):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])

    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv
