import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from os import listdir,makedirs
from os.path import isfile,join,exists
from PIL import Image

def hist_norm(img):
    img=np.asarray(img)/255
    p2, p98 = np.percentile(img, (4, 96))
    img_eq = exposure.rescale_intensity(img, in_range=(p2, p98))
    img_eq[img_eq<0.3] = 0
    img_eq[img_eq>0.3] = 1
    img_eq=(img_eq*255).astype(np.uint8)
    return Image.fromarray(img_eq,mode="L")
    


input_path="E:\Documentos\DECOM\Mestrado\Deblur_Challenge\output\step5_red"
output_path="E:\Documentos\DECOM\Mestrado\Deblur_Challenge\output\step5_red_norm"

ip1=join(input_path,"RW")
ip2=join(input_path,"RS")
ip3=join(input_path,"RSL1")
ip4=join(input_path,"RA")
ip5=join(input_path,"RC")

op1=join(output_path,"RW")
op2=join(output_path,"RS")
op3=join(output_path,"RSL1")
op4=join(output_path,"RA")
op5=join(output_path,"RC")

img_files=[f for f in listdir(ip1) if (isfile(join(ip1,f)) and f.endswith(".png"))]
for img_path in img_files:
    InputFile=join(ip1,img_path)
    img=Image.open(InputFile)
    img_eq=hist_norm(img)

    if not exists(op1):
        makedirs(op1)
    img_eq.save(join(op1,img_path))

img_files=[f for f in listdir(ip2) if (isfile(join(ip2,f)) and f.endswith(".png"))]
for img_path in img_files:
    InputFile=join(ip2,img_path)
    img=Image.open(InputFile)
    img_eq=hist_norm(img)
 
    if not exists(op2):
        makedirs(op2)
    img_eq.save(join(op2,img_path))

img_files=[f for f in listdir(ip3) if (isfile(join(ip3,f)) and f.endswith(".png"))]
for img_path in img_files:
    InputFile=join(ip3,img_path)
    img=Image.open(InputFile)
    img_eq=hist_norm(img)
 
    if not exists(op3):
        makedirs(op3)
    img_eq.save(join(op3,img_path))

img_files=[f for f in listdir(ip4) if (isfile(join(ip4,f)) and f.endswith(".png"))]
for img_path in img_files:
    InputFile=join(ip4,img_path)
    img=Image.open(InputFile)
    img_eq=hist_norm(img)
 
    if not exists(op4):
        makedirs(op4)
    img_eq.save(join(op4,img_path))


img_files=[f for f in listdir(ip5) if (isfile(join(ip5,f)) and f.endswith(".png"))]
for img_path in img_files:
    InputFile=join(ip5,img_path)
    img=Image.open(InputFile)
    img_eq=hist_norm(img)
 
    if not exists(op5):
        makedirs(op5)
    img_eq.save(join(op5,img_path))