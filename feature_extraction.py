# feature_extraction.py
import os
import numpy as np
from skimage import io, feature
from skimage.color import rgb2gray, rgba2rgb
from PIL import Image
import nibabel as nib
import pydicom
import cv2

HIST_BINS = 32

def read_image_to_2d(path):
    path_l = path.lower()
    if path_l.endswith('.nii') or path_l.endswith('.nii.gz') or path_l.endswith('.hdr') or path_l.endswith('.img'):
        imgobj = nib.load(path)
        data = imgobj.get_fdata()
        if data.ndim == 3:
            z = data.shape[2] // 2
            slice2d = data[:, :, z]
        elif data.ndim == 4:
            z = data.shape[2] // 2
            slice2d = data[:, :, z, 0]
        else:
            slice2d = np.squeeze(data)
        arr = np.array(slice2d, dtype=float)
        if np.nanmax(arr) > 1.0:
            arr = arr - np.nanmin(arr)
            mx = np.nanmax(arr)
            if mx > 0:
                arr = arr / mx
        return arr.clip(0,1)

    if path_l.endswith('.dcm'):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(float)
        if arr.ndim == 3:
            z = arr.shape[0] // 2
            arr = arr[z,:,:]
        arr = arr - np.min(arr)
        mx = np.max(arr)
        if mx > 0:
            arr = arr / mx
        return arr.clip(0,1)

    # other formats (jpg, png, etc.)
    try:
        with Image.open(path) as im:
            im_arr = np.array(im)
    except Exception:
        im_arr = io.imread(path)

    if im_arr.ndim == 3 and im_arr.shape[2] == 4:
        try:
            float_im = im_arr.astype(float) / 255.0
            float_im = rgba2rgb(float_im)
            im_arr = (float_im * 255.0).astype(np.uint8)
        except Exception:
            im_arr = im_arr[..., :3]

    if im_arr.ndim == 3 and im_arr.shape[2] == 1:
        im_arr = np.squeeze(im_arr, axis=2)

    if im_arr.ndim == 3 and im_arr.shape[2] == 3:
        try:
            gray = rgb2gray(im_arr)
        except Exception:
            gray = np.mean(im_arr, axis=2)
        im_arr = gray

    arr = im_arr.astype(float)
    if np.nanmax(arr) > 1.0:
        arr = arr - np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx > 0:
            arr = arr / mx
    arr = np.squeeze(arr)
    return arr.clip(0,1)


def extract_features_from_2d(img):
    mean_val = float(np.nanmean(img))
    std_val = float(np.nanstd(img))
    min_val = float(np.nanmin(img))
    max_val = float(np.nanmax(img))
    hist, _ = np.histogram(img, bins=HIST_BINS, range=(0.0, 1.0), density=True)

    h, w = img.shape
    if h < 8 or w < 8:
        img_for_glcm = cv2.resize(img, (max(8,w), max(8,h)), interpolation=cv2.INTER_LINEAR)
    else:
        img_for_glcm = img

    try:
        g_uint8 = (img_for_glcm * 255).astype('uint8')
        glcm = feature.graycomatrix(g_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = float(feature.graycoprops(glcm, 'contrast')[0,0])
        homogeneity = float(feature.graycoprops(glcm, 'homogeneity')[0,0])
        energy = float(feature.graycoprops(glcm, 'energy')[0,0])
        asm = float(feature.graycoprops(glcm, 'ASM')[0,0])
    except Exception:
        contrast = homogeneity = energy = asm = 0.0

    try:
        edges = cv2.Canny((img * 255).astype('uint8'), 50, 150)
        edge_count = int((edges > 0).sum())
    except Exception:
        edge_count = 0

    feats = [mean_val, std_val, min_val, max_val, contrast, homogeneity, energy, asm, edge_count]
    feats = feats + hist.astype(float).tolist()
    return np.array(feats, dtype=float)


def image_file_to_feature_vector(path, feature_order=None):
    img2d = read_image_to_2d(path)
    feats = extract_features_from_2d(img2d)
    # feats order already matches the CSV produced earlier: first 9 then hist_0..hist_31
    return feats


if __name__ == '__main__':
    import sys
    p = sys.argv[1]
    v = image_file_to_feature_vector(p)
    print('len=', len(v))