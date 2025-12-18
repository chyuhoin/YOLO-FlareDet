import torch
import numpy as np
from astropy.io import fits
from datetime import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from typing import Tuple
from dateutil.parser import parse

# --- 1. 读取和预处理原始数据 ---
def _readchase_full(file: str) -> Tuple[np.ndarray, dict]:
    # ... (来自有骋的神奇代码，保持不变，一行不敢改) ...
    hdu = fits.open(file, memmap=True)
    try:
        data = hdu[0].data.astype(np.float32)
        header = dict(hdu[0].header)
    except:
        data = hdu[1].data.astype(np.float32)
        header = dict(hdu[1].header)
    if data.ndim != 3:
        data = hdu[1].data.astype(np.float32)
        header = dict(hdu[1].header)
        if data.ndim != 3:
            hdu.close()
            raise TypeError(f'file {file} is not Chase\'s file, please use other function to read.')
    obs_time = datetime.strptime(header['DATE_OBS'], "%Y-%m-%dT%H:%M:%S")

    if obs_time < datetime.strptime('2023-04-18', "%Y-%m-%d"):
        cy = header['CRPIX1']; cx = header['CRPIX2']
    else:
        cx = header['CRPIX1']; cy = header['CRPIX2']
    y0 = int(cy - 1023); y1 = int(cy + 1025)
    x0 = int(cx - 1023); x1 = int(cx + 1025)
    data = data[:, y0:y1, x0:x1]
    
    if data.shape not in [(118,2048,2048), (116,2048,2048)]:
        hdu.close()
        raise TypeError(f'Chase file {file} is corrupted, please check.')
    if data.shape[0] == 116:
        first = data[0:1]
        last  = data[-1:]
        data = np.concatenate([first, data, last], axis=0)
    hdu.close()
    return data, header

# --- 2. GPU加速的拟合核心 ---
class GaussianFitter:
    def __init__(self, wavelengths_1d, ha_rest_wavelength, speed_of_light, device):
        self.wavelengths = torch.tensor(wavelengths_1d, dtype=torch.float32, device=device)
        self.ha_rest_wavelength = float(ha_rest_wavelength)
        self.speed_of_light = float(speed_of_light)
        self.device = device

    def gaussian_model(self, params):
        wl = self.wavelengths[None, :]
        C, A, lc, sg = params.unbind(dim=1)
        C = C[:, None]; A = A[:, None]; lc = lc[:, None]
        sg = torch.clamp(sg[:, None], min=1e-6)
        return C - A * torch.exp(- (wl - lc) ** 2 / (2 * sg ** 2))

    @torch.no_grad()
    def to_device(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def fit_batch(self, spectra_batch, initial_params_batch, num_iterations=100, lr=0.1):
        spectra = self.to_device(spectra_batch)
        params = self.to_device(initial_params_batch).clone()
        params.requires_grad_(True)
        opt = optim.Adam([params], lr=lr)
        for _ in range(num_iterations):
            opt.zero_grad()
            pred = self.gaussian_model(params)
            loss = nn.functional.mse_loss(pred, spectra)
            loss.backward()
            opt.step()
            with torch.no_grad():
                params[:, 0].clamp_(min=0)
                params[:, 1].clamp_(min=0)
                params[:, 3].clamp_(min=1e-6)
        with torch.no_grad():
            C, A, lc, sg = params.unbind(dim=1)
            line_core_intensity = C - A
            doppler_velocity = self.speed_of_light * (lc - self.ha_rest_wavelength) / self.ha_rest_wavelength
            fwhm = 2.354820045 * sg
            line_depth = A
            out = torch.stack([line_core_intensity, doppler_velocity, fwhm, line_depth, C], dim=1)
            return out.cpu().numpy()

def estimate_initial_params_vectorized(spectra, wavelengths, left_n=10, right_n=10, min_sigma_pix=2.0):
    spectra = np.asarray(spectra, dtype=np.float32)
    lam = np.asarray(wavelengths, dtype=np.float32)
    N, D = spectra.shape
    assert lam.shape[0] == D
    C_left = spectra[:, :left_n].mean(axis=1)
    C_right = spectra[:, -right_n:].mean(axis=1)
    C = 0.5 * (C_left + C_right)
    C = np.where(np.isfinite(C) & (C > 0), C, spectra.mean(axis=1))
    C = np.where(np.isfinite(C) & (C > 0), C, np.full_like(C, 1.0))
    w = np.clip(C[:, None] - spectra, 0.0, None)
    sumw = w.sum(axis=1) + 1e-8
    lc = (w * lam[None, :]).sum(axis=1) / sumw
    var = (w * (lam[None, :] - lc[:, None])**2).sum(axis=1) / sumw
    dlam = np.mean(np.diff(lam))
    sg_min = np.abs(dlam) * float(min_sigma_pix)
    sg = np.sqrt(np.clip(var, sg_min**2, None))
    Imin = spectra.min(axis=1)
    A = np.clip(C - Imin, 0.0, None)
    bad = ~np.isfinite(C) | ~np.isfinite(A) | ~np.isfinite(lc) | ~np.isfinite(sg)
    if bad.any():
        C[bad] = np.nan_to_num(C[bad], nan=1.0, posinf=1.0, neginf=1.0)
        A[bad] = np.nan_to_num(A[bad], nan=0.1, posinf=0.1, neginf=0.1)
        lc[bad] = np.nan_to_num(lc[bad], nan=lam[D//2])
        sg[bad] = np.nan_to_num(sg[bad], nan=sg_min)
    return np.stack([C, A, lc, sg], axis=1)

# --- 3. 核心处理函数 ---
def process_single_fits(fits_file_path: str, output_npy_path: str, output_png_path: str, batch_size: int):
    """处理单个FITS文件，生成NPY和PNG"""    
    # --- 数据加载 ---
    image_data_raw, header = _readchase_full(fits_file_path)
    image_data = np.transpose(image_data_raw, (1, 2, 0))
    h, w, d = image_data.shape
    
    first_wavelength = header.get('CRVAL3')
    delta_wavelength = header.get('CDELT3')
    wavelengths = np.array([first_wavelength + delta_wavelength * i for i in range(d)])
    
    HA_REST_WAVELENGTH = header.get('WAVE_LEN', 6562.8)
    SPEED_OF_LIGHT = 299792.458

    # --- 创建掩膜和准备数据 ---
    mean_intensity_map = np.mean(image_data, axis=2)
    threshold = np.nanmean(mean_intensity_map) * 0.2 # 使用固定阈值
    mask = mean_intensity_map > threshold
    
    image_data_flat = image_data.reshape(-1, d)
    mask_flat = mask.flatten()
    spectra_to_fit = image_data_flat[mask_flat]

    if len(spectra_to_fit) == 0:
        print(f"Warning: No valid pixels found for {os.path.basename(fits_file_path)}. Skipping.")
        return

    initial_params = estimate_initial_params_vectorized(spectra_to_fit, wavelengths)
    valid_indices = ~np.isnan(initial_params).any(axis=1)
    spectra_to_fit_valid = spectra_to_fit[valid_indices]
    initial_params_valid = initial_params[valid_indices]
    
    # --- GPU拟合 ---
    fitter = GaussianFitter(wavelengths, HA_REST_WAVELENGTH, SPEED_OF_LIGHT, device)
    all_results = []
    
    for i in range(0, len(spectra_to_fit_valid), batch_size):
        batch_spectra = spectra_to_fit_valid[i:i+batch_size]
        batch_params = initial_params_valid[i:i+batch_size]
        results_batch = fitter.fit_batch(batch_spectra, batch_params)
        all_results.append(results_batch)
        
    # --- 重构结果 ---
    final_results = np.concatenate(all_results, axis=0)
    fitted_params_flat = np.full((h * w, 5), np.nan)
    original_indices = np.where(mask_flat)[0][valid_indices]
    fitted_params_flat[original_indices] = final_results
    fitted_params_per_pixel = fitted_params_flat.reshape(h, w, 5)

    # !!! 关键：在保存NPY和生成PNG之前，垂直翻转图像数据 !!!
    fitted_params_per_pixel = np.flipud(fitted_params_per_pixel)
    
    # --- 保存5通道NPY文件 ---
    # 注意：这里保存的是归一化后的数据，方便下游检测任务
    fitted_params_norm = np.zeros_like(fitted_params_per_pixel)
    for k in range(fitted_params_per_pixel.shape[2]):
        param_channel = fitted_params_per_pixel[:, :, k]
        p_min, p_max = np.nanpercentile(param_channel, 0.5), np.nanpercentile(param_channel, 99.5)
        if p_max > p_min:
            normalized_channel = (np.clip(param_channel, p_min, p_max) - p_min) / (p_max - p_min)
        else:
            normalized_channel = np.zeros_like(param_channel)
            
        # 考虑到下游任务：将 NaN 替换为 0；根据下游任务的不同可以选择是否启用
        fitted_params_norm[:, :, k] = np.nan_to_num(normalized_channel, nan=0.0)
    
    np.save(output_npy_path, fitted_params_norm)

    # --- 保存3通道伪彩色PNG文件 ---
    # 选择通道: 0-线心强度, 1-多普勒速度, 2-半高全宽
    r_channel = fitted_params_per_pixel[:, :, 0] # 线心强度
    g_channel = fitted_params_per_pixel[:, :, 1] # 多普勒速度
    b_channel = fitted_params_per_pixel[:, :, 2] # 半高全宽

    # 独立归一化每个通道到 [0, 255]
    def normalize_channel(channel):
        # 多普勒速度需要对称范围
        if np.array_equal(channel, g_channel):
            v_max = np.nanpercentile(np.abs(channel), 99.5)
            p_min, p_max = -v_max, v_max
        else:
            p_min, p_max = np.nanpercentile(channel, 0.5), np.nanpercentile(channel, 99.5)
        
        if p_max > p_min:
            norm_ch = (np.clip(channel, p_min, p_max) - p_min) / (p_max - p_min) * 255
        else:
            norm_ch = np.zeros_like(channel)
        return np.nan_to_num(norm_ch, nan=0).astype(np.uint8)

    r_norm = normalize_channel(r_channel)
    g_norm = normalize_channel(g_channel)
    b_norm = normalize_channel(b_channel)

    rgb_image = np.stack([r_norm, g_norm, b_norm], axis=2)
    img = Image.fromarray(rgb_image)
    img.save(output_png_path)


# --- 4. 主执行流程 ---
def main():
    INPUT_DIR = '../raw'
    OUTPUT_DIR = './gauss_encode'
    BATCH_SIZE = 500000

    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fits_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.fits')]
    if not fits_files:
        print(f"No .fits files found in '{INPUT_DIR}'.")
        return

    print(f"Found {len(fits_files)} FITS files to process.")

    now = 0

    for filename in sorted(fits_files):
        fits_file_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing file {now+1}/{len(fits_files)}: {filename}", end='\r')
        now += 1
        
        try:
            base = os.path.basename(fits_file_path)
            date_part = base.split("RSM")[1].split("T")[0]
            time_part = base.split("T")[1].split("_")[0]
            dtime = f"{date_part[:4]}_{date_part[4:6]}{date_part[6:]}_{time_part}"
            
            out_npy_path = os.path.join(os.path.join(OUTPUT_DIR, 'npys'), f"{dtime}.npy")
            out_png_path = os.path.join(os.path.join(OUTPUT_DIR, 'pngs'), f"{dtime}.png")

            # 如果文件已存在，则跳过
            if os.path.exists(out_npy_path) and os.path.exists(out_png_path):
                print(f"Skipping {filename}, results already exist.")
                continue

            process_single_fits(fits_file_path, out_npy_path, out_png_path, BATCH_SIZE)

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
            continue

    print("\nAll files have been processed.")

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main()
