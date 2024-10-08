import os
import argparse
import pydicom

import numpy as np


def save_dataset(args):

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path+'/test_dataset/1mm', exist_ok=True)
    os.makedirs(args.save_path+'/test_dataset/3mm', exist_ok=True)
    os.makedirs(args.save_path+'/validation_dataset/1mm', exist_ok=True)
    os.makedirs(args.save_path+'/validation_dataset/3mm', exist_ok=True)
    os.makedirs(args.save_path+'/train_dataset/1mm', exist_ok=True)
    os.makedirs(args.save_path+'/train_dataset/3mm', exist_ok=True)
    
    patients_numbers = sorted([patient for patient in os.listdir(args.data_path)])

    for idx, patient in enumerate(patients_numbers):
        input_paths = os.path.join(args.data_path, patient, "quarter_{}mm".format(args.mm))
        target_paths = os.path.join(args.data_path, patient, "full_{}mm".format(args.mm))

        input_files = [os.path.join(input_paths, f) for f in sorted(os.listdir(input_paths))]
        target_files = [os.path.join(target_paths, f) for f in sorted(os.listdir(target_paths))]

        input_images = []
        target_images = []

        for i in range(len(input_files)):
            input_image = pydicom.dcmread(input_files[i])
            target_image = pydicom.dcmread(target_files[i])

            hu_input = get_pixels_hu(input_image)
            hu_target = get_pixels_hu(target_image)

            input_images.append(hu_input)
            target_images.append(hu_target)
        
        for img_idx in range(len(input_images)):

            input_image = input_images[img_idx]
            target_image = target_images[img_idx]

            if patients_numbers[idx] == args.test_patient:
                np.savez("{}/{}/{}mm/{}_{}mm_{}".format(args.save_path, "test_dataset", args.mm, patients_numbers[idx], args.mm, img_idx+1), input = input_image, target = target_image)

            elif patients_numbers[idx] == args.val_patient:
                np.savez("{}/{}/{}mm/{}_{}mm_{}".format(args.save_path, "validation_dataset", args.mm, patients_numbers[idx], args.mm, img_idx+1), input = input_image, target = target_image)

            else:
                np.savez("{}/{}/{}mm/{}_{}mm_{}".format(args.save_path, "train_dataset", args.mm, patients_numbers[idx], args.mm, img_idx+1), input = input_image, target = target_image)

        print("Patient {} Done.".format(patient))


def get_pixels_hu(image):
    image_array = image.pixel_array.astype(np.float32)
    intercept = image.RescaleIntercept
    slope = image.RescaleSlope
    
    hu_image = image_array * slope + intercept

    return hu_image


def apply_window(image, window_center=40, window_width=200):
    min_hu = window_center - (window_width / 2)
    max_hu = window_center + (window_width / 2)
    windowed_image = np.clip(image, min_hu, max_hu)

    return windowed_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dicom to NumPy")

    parser.add_argument('--data_path', type=str, default='datasets/LDCT_2016/Train')
    parser.add_argument('--save_path', type=str, default='npz_files')
    parser.add_argument('--val_patient', type=str, default='L333')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--mm', type=int, default=3)
    parser.add_argument('--min_range', type=float, default=-1024.0)
    parser.add_argument('--max_range', type=float, default=3072.0)

    args = parser.parse_args()
    save_dataset(args)
