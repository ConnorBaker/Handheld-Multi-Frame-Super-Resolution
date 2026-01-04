import torch
import matplotlib.pyplot as plt

T = 32
# Set search_area as a 2D array with a decreasing spike centered
search_area = torch.zeros((T, T), device='cuda') + 0.5
for i in range(T):
    for j in range(T):
        dist = ((i - T//2)**2 + (j - T//2)**2)**0.5
        # toto[i, j] = max(0, 1 - dist / (T//2))
        if dist < 5:
            search_area[i, j] = 1

T_template = 16
        

plt.imshow(search_area.cpu().numpy(), cmap='gray', vmin=0, vmax=1)

moving_fft = torch.fft.fft2(search_area, dim=(-1, -2))
moving_fft = torch.conj(moving_fft) * moving_fft
autocorr = torch.fft.ifft2(moving_fft, s=search_area.shape).real
autocorr = torch.fft.fftshift(autocorr, dim=(-2, -1))

moving_rfft = torch.fft.rfft2(search_area, dim=(-2, -1))
moving_rfft = torch.conj(moving_rfft) * moving_rfft
# Complete the rfft to get a full fft
# first reconstruct the negative frequencies by conjugate symmetry
neg_freqs = torch.conj(torch.flip(moving_rfft[..., 1:-1, :], dims=[-2]))
moving_rfft_completed = torch.cat([moving_rfft, neg_freqs], dim=-2)
autocorr_r = torch.fft.irfft2(moving_rfft, s=search_area.shape)
autocorr_r_completed = torch.fft.ifft2(moving_rfft_completed, s=search_area.shape).real
autocorr_r = torch.fft.fftshift(autocorr_r, dim=(-2, -1))
autocorr_r_completed = torch.fft.fftshift(autocorr_r_completed, dim=(-2, -1))
breakpoint()
# autocorr = autocorr[1:, 1:]
# autocorr_r = autocorr_r[1:, 1:]
plt.figure("FFT spectrum")
plt.imshow(moving_fft.abs().log().cpu().numpy())

plt.figure("RFFT spectrum")
plt.imshow(moving_rfft.abs().log().cpu().numpy())

plt.figure("RFFT completed spectrum")
plt.imshow(moving_rfft_completed.abs().log().cpu().numpy())

plt.figure("RFFT based autocorrelation")
plt.imshow(autocorr_r.cpu().numpy(), aspect='equal')

plt.figure("FFT based autocorrelation")
plt.imshow(autocorr.cpu().numpy(), aspect='equal')

plt.figure("RFFT completed based autocorrelation")
plt.imshow(autocorr_r_completed.cpu().numpy(), aspect='equal')
plt.show()

# corrs = torch.fft.irfft2(torch.conj(ref_fft_lvl) * moving_fft, s=(search_size, search_size))
# corrs = torch.fft.fftshift(corrs, dim=(-2, -1))