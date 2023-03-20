import torch
import torch.nn.functional as F
import pywt
import pytorch_wavelets as pw

def fft_mse_loss(img1, img2):
    img1_fft = torch.fft.fftn(img1, dim=(2,3),norm="ortho")
    img2_fft = torch.fft.fftn(img2, dim=(2,3),norm="ortho")
    # Splitting x and y into real and imaginary parts
    x_real, x_imag = torch.real(img1_fft), torch.imag(img1_fft)
    y_real, y_imag = torch.real(img2_fft), torch.imag(img2_fft)
    # Calculate the MSE between the real and imaginary parts separately
    mse_real = torch.nn.MSELoss()(x_real, y_real)
    mse_imag = torch.nn.MSELoss()(x_imag, y_imag)
    return mse_imag+mse_real

def dwt_mse_loss(x, y ,J=4):
    # Perform 4-level 2D discrete wavelet transform on both images
    x_dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
    y_dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
    x_dwt_f.cuda()
    y_dwt_f.cuda()
    x_dwt=x_dwt_f(x)[1]
    y_dwt=y_dwt_f(y)[1]
    h_mse,v_mse,d_mse=0,0,0
    for i in range(J):
        # Calculate MSE between the coefficients of each subband
        h_mse += torch.nn.functional.mse_loss(x_dwt[i][:,:,0,:,:], y_dwt[i][:,:,0,:,:])
        v_mse += torch.nn.functional.mse_loss(x_dwt[i][:,:,1,:,:], y_dwt[i][:,:,1,:,:])
        d_mse += torch.nn.functional.mse_loss(x_dwt[i][:,:,2,:,:], y_dwt[i][:,:,2,:,:])

    # Sum the MSE losses across subbands and return
    return h_mse + v_mse + d_mse

def image_compare_loss(x, y , alpha=0.2 ,beta=0.1):
    # Calculation of MSE loss in the frequency domain
    loss_fft = fft_mse_loss(x, y)
    # Calculating multilevel discrete wavelet transform MSE losses
    loss_dwt = dwt_mse_loss(x, y)
    return alpha*loss_fft + beta*loss_dwt
