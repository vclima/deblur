from PIL import Image
from os import listdir,makedirs
from os.path import isfile,join,exists

import cv2 as cv
import numpy as np
from numpy.fft import fft2
from numpy.fft import ifft2
from scipy.fftpack import fftn, ifftn
from scipy import ndimage
from skimage.restoration import (denoise_tv_chambolle,denoise_wavelet,denoise_bilateral)


from PIL import Image
from skimage.color import rgb2yuv, yuv2rgb



def deblur(input_path, output_path, categoryNbr):
  exclude=["LSF","PSF"]
  ip1=join(input_path,"CAM01")
  ip2=join(input_path,"CAM02")

  op1=join(output_path,"RW")
  op2=join(output_path,"RS")

  img_files=[f for f in listdir(ip1) if (isfile(join(ip1,f)) and f.endswith(".tif"))]
  for img_path in img_files:
    if "PSF" in img_path:
      psf1=Image.open(join(ip1,img_path))
      psf2=Image.open(join(ip2,img_path))

  for img_path in img_files:
    if not any(ignore for ignore in exclude if ignore in img_path):
      im1=Image.open(join(ip1,img_path))
      im2=Image.open(join(ip2,img_path))

      scale=1
      Rw,Rsparse=deblur_fp_06(im1,im2,psf1,psf2,scale)
      Rw_image=Image.fromarray(Rw)
      Rsparse_image=Image.fromarray(Rsparse)


      if not exists(op1):
        makedirs(op1)
      if not exists(op2):
        makedirs(op2)

      Rw_image.save(join(op1,img_path))
      Rsparse_image.save(join(op2,img_path))


  return

def deblur_fp_06(cam01,cam02,psf1,psf2,scale=1):#Arrumar os parametros daqui
  #im = Image.open('/content/sample_data/focusStep_5_timesR_size_30_sample_0001_cam01.tif')
  im = cam01
  t = np.asarray(im)
  print(t.shape)
  t = (t-np.min(t))/(np.max(t)-np.min(t))
  t = 1-t
  t_aux=t

  #im2  = Image.open('/content/sample_data/focusStep_5_timesR_size_30_sample_0001_cam02.tif')
  im2 = cam02
  X = np.asarray(im2)
  X = (X-np.min(X))/(np.max(X)-np.min(X))
  X = 1-X

  #im3 = Image.open('/content/sample_data/focusStep_5_PSF_cam01.tif')
  im3 = psf1
  tpsf = np.asarray(im3)
  tpsf = (tpsf-np.min(tpsf))/(np.max(tpsf)-np.min(tpsf))
  tpsf = 1-tpsf

  #im4 = Image.open('/content/sample_data/focusStep_5_PSF_cam02.tif')
  im4 = psf2
  Xpsf = np.asarray(im4)
  Xpsf = (Xpsf-np.min(Xpsf))/(np.max(Xpsf)-np.min(Xpsf))
  Xpsf = 1-Xpsf

  width = int(t.shape[1] * scale)
  height = int(t.shape[0] * scale)
  dim = (width, height)

  scaled_t = cv.resize(t, dim, interpolation = cv.INTER_AREA)
  scaled_X = cv.resize(X, dim, interpolation = cv.INTER_AREA)
  scaled_tpsf = cv.resize(tpsf, dim, interpolation = cv.INTER_AREA)
  scaled_Xpsf = cv.resize(Xpsf, dim, interpolation = cv.INTER_AREA)
  '''
  t    = t.copy()
  X    = X.copy()
  R0   = X.copy()
  tpsf = tpsf.copy()
  Xpsf = Xpsf.copy()
  '''
  t    = scaled_t.copy()
  X    = scaled_X.copy()
  R0   = scaled_X.copy()
  tpsf = scaled_tpsf.copy()
  Xpsf = scaled_Xpsf.copy()

  sig  = 0.1
  x = np.linspace(-1, 1, tpsf.shape[1])
  y = np.linspace(-1, 1, tpsf.shape[0])*tpsf.shape[0]/tpsf.shape[1]
  x, y = np.meshgrid(x, y) 
  wind = gaus2d(x, y,sx=sig,sy=sig)
  wind = (wind-np.min(wind))/(np.max(wind)-np.min(wind))

  tpsf = tpsf*wind
  Xpsf = Xpsf*wind

  fftR = fft2(tpsf)
  fftX = fft2(Xpsf)
  OTF0 = ((np.conjugate(fftR)*fftX)/(fftR*np.conjugate(fftR)))

  Niter     = 100
  lamb      = 0.01
  lambdaPSF = 5
  
  f=denoise_wavelet
  Rw,OTFEw=blind_decon2D_red_fp_wavelet(R0,X,OTF0,Niter,lamb,lambdaPSF,f,t)

  f=denoise_wavelet
  Rsparse,OTFEsparse=blind_decon2D_red_fp_wavelet_sparse(R0,X,OTF0,Niter,lamb,lambdaPSF,f,t)

  Rw = (Rw-np.min(Rw))/(np.max(Rw)-np.min(Rw))
  Rw = 1-Rw

  Rsparse = (t-np.min(Rsparse))/(np.max(Rsparse)-np.min(Rsparse))
  Rsparse = 1-Rsparse
  
  return Rw,Rsparse

def ptrans(f):
  t=f.shape[0]//2+1,f.shape[1]//2+1
  rr,cc = t
  H,W = f.shape
  
  r=H-rr%H
  c=W-cc%W
      
  h=np.zeros([2*H,2*W])
  h[0:H,0:W]=f
  h[H:2*H,0:W]=f
  h[0:H,W:2*W]=f
  h[H:2*H,W:2*W]=f
  
  g=np.zeros([H,W])
  g=h[r:r+H,c:c+W]
  
  return g

def PSNR(original, compressed):
  mse = np.mean((original - compressed) ** 2)
  if(mse == 0):  # MSE is zero means no noise is present in the signal .
                # Therefore PSNR have no importance.
      return 100
  original=(original-np.min(original))/(np.max(original)-np.min(original))
  compressed=(compressed-np.min(compressed))/(np.max(compressed)-np.min(compressed))
  max_pixel = 1.0
  psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
  return psnr

# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=0.01, sy=0.01):
  return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def psf2otf(psf, otf_size):
    # calculate otf from psf with size >= psf size
    
    if psf.any(): # if any psf element is non-zero    
        # pad PSF with zeros up to image size  
        pad_size = ((0,otf_size[0]-psf.shape[0]),(0,otf_size[1]-psf.shape[1]))
        psf_padded = np.pad(psf, pad_size, 'constant')    
        
        # circularly shift psf   
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[0]/2)), axis=0)    
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[1]/2)), axis=1)       
       
       #calculate otf    
        otf = fftn(psf_padded)
        # this condition depends on psf size    
        num_small = np.log2(psf.shape[0])*4*np.spacing(1)    
        if np.max(abs(otf.imag))/np.max(abs(otf)) <= num_small:
            otf = otf.real 
    else: # if all psf elements are zero
        otf = np.zeros(otf_size)
    return otf

def otf2psf(otf, psf_size):
    # calculate psf from otf with size <= otf size
    
    if otf.any(): # if any otf element is non-zero
        # calculate psf     
        psf = ifftn(otf)
        # this condition depends on psf size    
        num_small = np.log2(otf.shape[0])*4*np.spacing(1)    
        if np.max(abs(psf.imag))/np.max(abs(psf)) <= num_small:
            psf = psf.real 
        
        # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0]/2)), axis=0)    
        psf = np.roll(psf, int(np.floor(psf_size[1]/2)), axis=1) 
        
        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else: # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf

def blind_decon2D_red_fp_wavelet(R0,X,OTF,Niter,lamb,lambdaPSF,f,R_true):
  R=R0.copy()
  OTF0 = OTF.copy()
  scale_percent = 100
  width = int(OTF0.shape[1] * scale_percent / 100)
  height = int(OTF0.shape[0] * scale_percent / 100)
  dim = (height, width)
  PSF  = otf2psf(OTF0, dim)
  OTF  = psf2otf(PSF, OTF0.shape)

  fftX=fft2(X)
  fftR=fft2(R)
  fftHtH=np.conjugate(OTF)*OTF
  fftHtX=np.conjugate(OTF)*fftX 

  for ii in range(Niter):
    # denoise R

    RD=f(R)

    b=fftHtX+lamb*(fft2(RD))
    A=fftHtH+lamb
    fftR=np.nan_to_num(b/A)
    R=np.real(ifft2(fftR))

    # update OTF
    OTF0 = ((np.conjugate(fftR)*fftX)/(fftR*np.conjugate(fftR)+lambdaPSF))
    PSF = otf2psf(OTF0, dim)
    OTF = psf2otf(PSF, OTF0.shape)

    fftHtH=np.conjugate(OTF)*OTF
    fftHtX=np.conjugate(OTF)*fftX 

    psnr=PSNR(R_true,R)
    print('iter: ',ii,'\t PSNR=',psnr)
    
  return R,OTF

def blind_decon2D_red_fp_wavelet_sparse(R0,X,OTF,Niter,lamb,lambdaPSF,f,R_true):
  R=R0.copy()
  OTF0 = OTF.copy()
  scale_percent = 100
  width = int(OTF0.shape[1] * scale_percent / 100)
  height = int(OTF0.shape[0] * scale_percent / 100)
  dim = (height, width)
  PSF  = otf2psf(OTF0, dim)
  OTF  = psf2otf(PSF, OTF0.shape)

  fftX=fft2(X)
  fftR=fft2(R)
  fftHtH=np.conjugate(OTF)*OTF
  fftHtX=np.conjugate(OTF)*fftX 

  for ii in range(Niter):
    # denoise R

    RD=f(R)

    RD = RD - np.mean(RD)
    RD[RD<0] = 0
    RD[RD>1] = 1
    RD=RD/np.max(RD)

    RD=ndimage.uniform_filter(RD,3)

    b=fftHtX+lamb*(fft2(RD))
    A=fftHtH+lamb
    fftR=np.nan_to_num(b/A)
    R=np.real(ifft2(fftR))

    # update OTF
    OTF0 = ((np.conjugate(fftR)*fftX)/(fftR*np.conjugate(fftR)+lambdaPSF))
    PSF = otf2psf(OTF0, dim)
    OTF = psf2otf(PSF, OTF0.shape)

    fftHtH=np.conjugate(OTF)*OTF
    fftHtX=np.conjugate(OTF)*fftX 

    psnr=PSNR(R_true,R)
    print('iter: ',ii,'\t PSNR=',psnr)
    
  return R,OTF

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def YUV(endereco):
    return rgb2yuv(endereco)

def RGB(endereco):
    return yuv2rgb(endereco)
