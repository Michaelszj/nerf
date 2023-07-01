import skimage.metrics 

def psnr(gt, img):
    return skimage.metrics.peak_signal_noise_ratio(gt, img, data_range=255)

def ssim(gt, img):
    return skimage.metrics.structural_similarity(gt, img, data_range=255, channel_axis=2)