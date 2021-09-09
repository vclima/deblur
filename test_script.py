import OCR_evaluation
import deblur_script

from os import listdir
from os.path import isfile,join

input_path="E:\Documentos\DECOM\Mestrado\Deblur_Challenge\step5_red\Verdana"
output_path="E:\Documentos\DECOM\Mestrado\Deblur_Challenge\output\step5_red"

deblur_script.deblur(input_path,output_path,0)

ip1=join(input_path,"CAM01")
ip2=join(input_path,"CAM02")


op1=join(output_path,"RW")
op2=join(output_path,"RS")

exclude=["LSF","PSF"]
i=0
score=0
img_files=[f for f in listdir(ip1) if (isfile(join(ip1,f)) and f.endswith(".tif"))]
for img_path in img_files:
    if not any(ignore for ignore in exclude if ignore in img_path):
        InputFile=join(op1,img_path)
        trueText=join(ip1,img_path)
        trueText=trueText.replace('tif','txt')
        ocr_result = OCR_evaluation.evaluateImage(InputFile, trueText,verbose=1)
        if type(ocr_result) == float:
            score += ocr_result
        else:
            score += 0
        i=i+1
print('Step accuracy - Wavelet: ',score/i)

i=0
score=0
for img_path in img_files:
    if not any(ignore for ignore in exclude if ignore in img_path):
        InputFile=join(op2,img_path)
        trueText=join(ip1,img_path)
        trueText=trueText.replace('tif','txt')
        ocr_result = OCR_evaluation.evaluateImage(InputFile, trueText,verbose=1)
        if type(ocr_result) == float:
            score += ocr_result
        else:
            score += 0
        i=i+1
print('Step accuracy - Sparse Denoise: ',score/i)