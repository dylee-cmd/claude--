"""
Chirp Scaling Algorithm (CSA) for the raw phase history written by sar_simulation_env.py.

Memory / output design (the original version needed ~60 GB):
  * The raw data is streamed from the .npz in row blocks (only channel 0, never the whole
    (channels, pulses, samples) array) and kept as ONE complex64 array (~5 GB for the default scene).
  * The 2-D phase screens (Phi_1/2/3, meshgrids) are never materialised; they are generated per
    row block on the fly (in float64, wrapped to [0, 2pi), then cast to complex64).
  * Steps 2-6 (chirp scaling, range FFT, RC+RCMC, range IFFT, azimuth compression) are all row-local,
    so they are fused into a single pass over row blocks. Only the azimuth FFT / IFFT need column blocks.
  * fftshift / ifftshift pairs of the original code cancel out, so they are dropped (the frequency
    axes are simply kept in FFT order). Stage checkpoints are re-shifted when recorded.
  * Output: only the (cropped) focused image by default. Intermediate stages are optional and are
    saved decimated. Everything is complex64 (the original saved complex128 stages).
  * Compute runs on the GPU with torch (cuFFT) when available, otherwise on the CPU with the same code.
    If the array does not fit in GPU memory it is stored in host RAM and streamed through the GPU.
"""
import argparse
import os
import time
import zipfile

import numpy as np
import torch

C_LIGHT = 299792458.0
RE = 6371000.0
GM = 3.986004418e14
TWO_PI = 2.0 * np.pi

STAGE_NAMES = ['01_raw_time', '02_range_doppler', '03_cs_applied_rd', '04_rc_rcmc_2df',
               '05_rc_rcmc_rd', '06_ac_applied_rd']
# (azimuth axis fftshifted, range axis fftshifted) -- matches the layout of the original checkpoints
STAGE_DOMAIN = {'01_raw_time': (False, False), '02_range_doppler': (True, False),
                '03_cs_applied_rd': (True, False), '04_rc_rcmc_2df': (True, True),
                '05_rc_rcmc_rd': (True, False), '06_ac_applied_rd': (True, False)}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _expj(phase64):
    """exp(j*phase) as complex64. The phase is wrapped in float64 first (|phase| reaches ~4e7 rad)."""
    p = torch.remainder(phase64, TWO_PI).float()
    return torch.polar(torch.ones_like(p), p)


class StageRecorder:
    """Collects decimated checkpoints of intermediate stages while the data streams through in blocks."""

    def __init__(self, names, n_az, n_rg, decim):
        self.buf = {}
        d_az, d_rg = decim
        for name in names:
            az_sh, rg_sh = STAGE_DOMAIN[name]
            j_az, j_rg = np.arange(0, n_az, d_az), np.arange(0, n_rg, d_rg)
            u_az = (j_az - n_az // 2) % n_az if az_sh else j_az  # index in FFT order
            u_rg = (j_rg - n_rg // 2) % n_rg if rg_sh else j_rg
            self.buf[name] = (np.zeros((len(j_az), len(j_rg)), np.complex64), u_az, u_rg)

    def put(self, name, t, r0, c0):
        """t holds rows r0.. and columns c0.. (FFT order) of the full array."""
        if name not in self.buf:
            return
        arr, u_az, u_rg = self.buf[name]
        h, w = t.shape
        ia = np.nonzero((u_az >= r0) & (u_az < r0 + h))[0]
        ic = np.nonzero((u_rg >= c0) & (u_rg < c0 + w))[0]
        if len(ia) == 0 or len(ic) == 0:
            return
        ra = torch.from_numpy(u_az[ia] - r0).to(t.device)
        rc = torch.from_numpy(u_rg[ic] - c0).to(t.device)
        arr[np.ix_(ia, ic)] = t[ra][:, rc].cpu().numpy()


def _open_channel0(raw_file):
    """Open rx_channels.npy inside the npz as a stream positioned at channel 0 (no full load)."""
    zf = zipfile.ZipFile(raw_file)
    f = zf.open('rx_channels.npy')
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"Unsupported .npy version {version}")
    if fortran:
        raise ValueError("Fortran-ordered rx_channels is not supported")
    if len(shape) == 2:
        shape = (1,) + tuple(shape)
    return zf, f, shape[1], shape[2], dtype


def _derive_geometry(data, n_az):
    """Vr and R_ref from the stored trajectory (the simulator's altitude / grazing angle can differ
    from any hard-coded value). Returns None if the file has no trajectory."""
    if not all(k in data.files for k in ('pos_tx', 'vel_tx', 'scene_center')):
        return None
    mid = n_az // 2  # slow time t = 0
    pos = np.asarray(data['pos_tx'][mid], np.float64)
    vel = np.asarray(data['vel_tx'][mid], np.float64)
    r_sat = np.linalg.norm(pos + np.array([0.0, 0.0, RE]))  # scene centre is at the origin, Earth centre at -Re
    v_sat = np.linalg.norm(vel)
    r_ref = np.linalg.norm(np.asarray(data['scene_center'], np.float64) - pos)
    return v_sat * np.sqrt(RE / r_sat), r_ref, r_sat, v_sat


# -----------------------------------------------------------------------------
# main pipeline
# -----------------------------------------------------------------------------
def run_csa_pipeline(raw_file="sar_raw_phase_history.npz", output_file="sar_csa_stages.npz",
                     device="auto", crop_range_m=1500.0, crop_az_m=3000.0,
                     save_stages=(), stage_decim=(8, 8), amplitude_only=False, chunk_mb=128):
    """
    crop_range_m / crop_az_m : half-width of the saved image around the scene centre (None = full image).
    save_stages              : subset of STAGE_NAMES to additionally save (decimated by stage_decim).
    amplitude_only           : save 20*log10|img| as float16 instead of complex64.
    """
    if not os.path.exists(raw_file):
        print(f"Error: {raw_file} not found. Run sar_simulation_env.py first.")
        return
    t_all = time.time()
    print(f"Loading raw phase history from {raw_file}...")
    data = np.load(raw_file)

    t_start_fast = float(data['t_start_fast'])
    fs = float(data['fs'])
    prf = float(data['prf'])
    pulse_width = float(data['pulse_width'])
    center_freq = float(data['center_freq'])
    bandwidth = float(data['bandwidth'])
    slow_time = np.asarray(data['slow_time'], np.float64)

    zf, fh, n_az, n_rg, raw_dtype = _open_channel0(raw_file)
    print(f"Data: {n_az} pulses x {n_rg} samples ({n_az * n_rg * 8 / 2**30:.2f} GiB complex64 per channel), "
          f"PRF {prf} Hz, BW {bandwidth / 1e6} MHz")

    # --- device selection ----------------------------------------------------
    dev = torch.device('cuda' if (device == 'auto' and torch.cuda.is_available()) else
                       ('cpu' if device == 'auto' else device))
    chunk_bytes = int(chunk_mb * 2**20)
    need = n_az * n_rg * 8
    store = torch.device('cpu')
    if dev.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(dev)
        if free > need + 16 * chunk_bytes + 2**30:
            store = dev
    print(f"Compute device: {dev} | full array stored on: {store}")

    # --- geometry --------------------------------------------------------------
    lam = C_LIGHT / center_freq
    Kr = bandwidth / pulse_width
    geo = _derive_geometry(data, n_az)
    if geo is not None:
        Vr, R_ref, r_sat, v_sat = geo
        print(f"Geometry from stored trajectory: R_sat={r_sat / 1e3:.1f} km, V_sat={v_sat:.1f} m/s")
    else:  # legacy fallback (old hard-coded defaults)
        r_sat = RE + 500e3
        v_sat = np.sqrt(GM / r_sat)
        Vr = v_sat * np.sqrt(RE / r_sat)
        th = np.radians(50.0)
        gam = np.arcsin(r_sat / RE * np.sin(th)) - th
        R_ref = np.sqrt(RE**2 + r_sat**2 - 2 * RE * r_sat * np.cos(gam))
        print("WARNING: no trajectory in file, using legacy hard-coded 500 km / 40 deg geometry")
    print(f"Vr: {Vr:.2f} m/s, R_ref: {R_ref:.2f} m")

    # --- axes / per-row terms (1-D, float64) ---------------------------------------
    dt = 1.0 / fs
    tau = t_start_fast + np.arange(n_rg, dtype=np.float64) * dt
    fr = np.fft.fftfreq(n_rg, dt)
    fa = np.fft.fftfreq(n_az, 1.0 / prf)  # FFT order, no shifts needed (they cancel)
    arg = 1.0 - (lam * fa / (2.0 * Vr))**2
    arg[arg < 0] = 1e-9
    D_fa = np.sqrt(arg)
    Cs_fa = 1.0 / D_fa - 1.0
    tau_ref_fa = 2.0 * R_ref / (C_LIGHT * D_fa)
    R_vec = C_LIGHT * tau / 2.0
    cross_range_axis = (slow_time - slow_time.mean()) * Vr

    f64 = dict(device=dev, dtype=torch.float64)
    tau_t, fr_t, R_t = (torch.tensor(a, **f64) for a in (tau, fr, R_vec))
    Cs_t, D_t, tref_t = (torch.tensor(a, **f64) for a in (Cs_fa, D_fa, tau_ref_fa))
    tau_diff_t = tau_t - 2.0 * R_ref / C_LIGHT

    # --- output window ---------------------------------------------------------------
    if crop_range_m is None:
        rg_lo, rg_hi = 0, n_rg
    else:
        idx = np.nonzero(np.abs(R_vec - R_ref) <= crop_range_m)[0]
        if len(idx) == 0:
            print("WARNING: R_ref lies outside the recorded range window; saving the full range extent")
        rg_lo, rg_hi = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, n_rg)
    if crop_az_m is None:
        az_lo, az_hi = 0, n_az
    else:
        c_az = int(np.argmin(np.abs(cross_range_axis)))
        half = int(np.ceil(crop_az_m / (Vr / prf)))
        az_lo, az_hi = max(0, c_az - half), min(n_az, c_az + half)
    print(f"Saved image window: azimuth [{az_lo}:{az_hi}] x range [{rg_lo}:{rg_hi}] "
          f"(full image is {n_az} x {n_rg}) -- use --full to keep everything")

    stages = [s for s in STAGE_NAMES if s in set(save_stages)]
    rec = StageRecorder(stages, n_az, n_rg, stage_decim)

    rows = max(16, chunk_bytes // (n_rg * 8))
    cols = max(16, chunk_bytes // (n_az * 8))

    def sync():
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)

    # --- Step 0: stream raw data into the working array ---------------------------------
    t0 = time.time()
    S = torch.empty((n_az, n_rg), dtype=torch.complex64, device=store)
    row_bytes = n_rg * raw_dtype.itemsize
    for r0 in range(0, n_az, rows):
        r1 = min(r0 + rows, n_az)
        buf = fh.read((r1 - r0) * row_bytes)
        blk = np.frombuffer(buf, dtype=raw_dtype).reshape(r1 - r0, n_rg).astype(np.complex64)
        blk_t = torch.from_numpy(blk).to(dev)
        rec.put('01_raw_time', blk_t, r0, 0)
        S[r0:r1] = blk_t.to(store)
    fh.close()
    zf.close()
    sync()
    print(f"Step 0: raw data loaded ({time.time() - t0:.1f} s)")

    # --- Step 1: azimuth FFT (column blocks) ----------------------------------------------
    t0 = time.time()
    for c0 in range(0, n_rg, cols):
        c1 = min(c0 + cols, n_rg)
        X = torch.fft.fft(S[:, c0:c1].to(dev), dim=0)
        rec.put('02_range_doppler', X, 0, c0)
        S[:, c0:c1] = X.to(store)
    sync()
    print(f"Step 1: azimuth FFT ({time.time() - t0:.1f} s)")

    # --- Steps 2-6: fused row-local processing --------------------------------------------
    t0 = time.time()
    for r0 in range(0, n_az, rows):
        r1 = min(r0 + rows, n_az)
        Cs = Cs_t[r0:r1, None]
        D = D_t[r0:r1, None]
        x = S[r0:r1].to(dev)

        # Step 2: chirp scaling phase  Phi_1(tau, fa)
        x = x * _expj(-np.pi * Kr * Cs * (tau_t[None, :] - tref_t[r0:r1, None])**2)
        rec.put('03_cs_applied_rd', x, r0, 0)

        # Step 3-4: range FFT, range compression + bulk RCMC  Phi_2(fr, fa)
        X = torch.fft.fft(x, dim=1)
        phi2 = np.pi * fr_t[None, :]**2 / (Kr * (1.0 + Cs)) + 4.0 * np.pi * R_ref * Cs * fr_t[None, :] / C_LIGHT
        X = X * _expj(phi2)
        rec.put('04_rc_rcmc_2df', X, r0, 0)

        # Step 5: range IFFT
        x = torch.fft.ifft(X, dim=1)
        rec.put('05_rc_rcmc_rd', x, r0, 0)

        # Step 6: azimuth compression + residual phase  Phi_3(R, fa)
        phi3 = 4.0 * np.pi * R_t[None, :] * D / lam - np.pi * Kr * Cs * (1.0 + Cs) * tau_diff_t[None, :]**2
        x = x * _expj(phi3)
        rec.put('06_ac_applied_rd', x, r0, 0)
        S[r0:r1] = x.to(store)
    sync()
    print(f"Steps 2-6: chirp scaling / RC / RCMC / azimuth compression ({time.time() - t0:.1f} s)")

    # --- Step 7: azimuth IFFT, only for the saved range columns -----------------------------
    t0 = time.time()
    img = np.empty((az_hi - az_lo, rg_hi - rg_lo), dtype=np.complex64)
    for c0 in range(rg_lo, rg_hi, cols):
        c1 = min(c0 + cols, rg_hi)
        y = torch.fft.ifft(S[:, c0:c1].to(dev), dim=0)[az_lo:az_hi]
        img[:, c0 - rg_lo:c1 - rg_lo] = y.cpu().numpy()
    sync()
    del S
    print(f"Step 7: azimuth IFFT ({time.time() - t0:.1f} s)")

    # --- save ----------------------------------------------------------------------------------
    out = dict(range_axis=R_vec[rg_lo:rg_hi], cross_range_axis=cross_range_axis[az_lo:az_hi],
               R_ref=R_ref, Vr=Vr, fs=fs, prf=prf,
               image_window=np.array([az_lo, az_hi, rg_lo, rg_hi]), full_shape=np.array([n_az, n_rg]))
    if amplitude_only:
        out['07_focused_image_db'] = (20.0 * np.log10(np.abs(img) + 1e-30)).astype(np.float16)
    else:
        out['07_focused_image'] = img
    for name in stages:
        out[name] = rec.buf[name][0]
        out[name + '_axes'] = np.array([n_az, n_rg, stage_decim[0], stage_decim[1]])
    print(f"Saving to {output_file}...")
    np.savez(output_file, **out)
    print(f"Output size: {os.path.getsize(output_file) / 2**20:.1f} MiB")
    print(f"SAR CSA Processing Complete ({time.time() - t_all:.1f} s total).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GPU Chirp Scaling Algorithm")
    ap.add_argument("raw_file", nargs="?", default="sar_raw_phase_history.npz")
    ap.add_argument("output_file", nargs="?", default="sar_csa_stages.npz")
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
    ap.add_argument("--crop-range-m", type=float, default=1500.0, help="half-width of saved range window [m]")
    ap.add_argument("--crop-az-m", type=float, default=3000.0, help="half-width of saved azimuth window [m]")
    ap.add_argument("--full", action="store_true", help="save the full, uncropped image")
    ap.add_argument("--save-stages", default="", help="comma list of extra checkpoints, or 'all' (decimated)")
    ap.add_argument("--stage-decim", type=int, default=8, help="decimation factor for extra checkpoints")
    ap.add_argument("--amplitude-only", action="store_true", help="save float16 dB magnitude instead of complex64")
    ap.add_argument("--chunk-mb", type=float, default=128, help="block size used for streaming")
    a = ap.parse_args()
    stg = STAGE_NAMES if a.save_stages == "all" else [s for s in a.save_stages.split(",") if s]
    run_csa_pipeline(a.raw_file, a.output_file, a.device,
                     None if a.full else a.crop_range_m, None if a.full else a.crop_az_m,
                     stg, (a.stage_decim, a.stage_decim), a.amplitude_only, a.chunk_mb)
