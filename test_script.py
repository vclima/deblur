import OCR_evaluation
import deblur_script

from os import listdir
from os.path import isfile,join
from contextlib import redirect_stdout



if __name__ == '__main__':
    input_path="C:\Deblur_Challenge\step7\Verdana"
    output_path="C:\Deblur_Challenge\output\step7_norm"

    #deblur_script.deblur(input_path,output_path,0)

    ip1=join(input_path,"CAM01")
    ip2=join(input_path,"CAM02")


    op1=join(output_path,"RW")
    op2=join(output_path,"RS")
    op3=join(output_path,"RSL1")
    op4=join(output_path,"RA")
    op5=join(output_path,"RC")

    with open('out_7_norm.txt', 'w') as f:
        with redirect_stdout(f):
            exclude=["LSF","PSF"]
            i=0
            score=0
            img_files=[f for f in listdir(ip1) if (isfile(join(ip1,f)) and f.endswith(".tif"))]
            for img_path in img_files:
                if not any(ignore for ignore in exclude if ignore in img_path):
                    InputFile=join(op1,img_path)
                    trueText=join(ip1,img_path)
                    InputFile=InputFile.replace('tif','png')
                    trueText=trueText.replace('tif','txt')
                    ocr_result = OCR_evaluation.evaluateImage(InputFile, trueText,verbose=1)
                    if type(ocr_result) == float:
                        score += ocr_result
                    else:
                        score += 0
                    i=i+1
            print('Step accuracy - Wavelet: ',score/i)
            print('\n')

            i=0
            score=0
            for img_path in img_files:
                if not any(ignore for ignore in exclude if ignore in img_path):
                    InputFile=join(op2,img_path)
                    trueText=join(ip1,img_path)
                    InputFile=InputFile.replace('tif','png')
                    trueText=trueText.replace('tif','txt')
                    ocr_result = OCR_evaluation.evaluateImage(InputFile, trueText,verbose=1)
                    if type(ocr_result) == float:
                        score += ocr_result
                    else:
                        score += 0
                    i=i+1
            print('Step accuracy - Wavelet Sparse Denoise: ',score/i)
            print('\n')

            i=0
            score=0
            for img_path in img_files:
                if not any(ignore for ignore in exclude if ignore in img_path):
                    InputFile=join(op3,img_path)
                    trueText=join(ip1,img_path)
                    InputFile=InputFile.replace('tif','png')
                    trueText=trueText.replace('tif','txt')
                    ocr_result = OCR_evaluation.evaluateImage(InputFile, trueText,verbose=1)
                    if type(ocr_result) == float:
                        score += ocr_result
                    else:
                        score += 0
                    i=i+1
            print('Step accuracy - L1 Sparse Denoise: ',score/i)
            print('\n')

            i=0
            score=0
            for img_path in img_files:
                if not any(ignore for ignore in exclude if ignore in img_path):
                    InputFile=join(op4,img_path)
                    trueText=join(ip1,img_path)
                    InputFile=InputFile.replace('tif','png')
                    trueText=trueText.replace('tif','txt')
                    ocr_result = OCR_evaluation.evaluateImage(InputFile, trueText,verbose=1)
                    if type(ocr_result) == float:
                        score += ocr_result
                    else:
                        score += 0
                    i=i+1
            print('Step accuracy - L1 Anti-sparse Denoise: ',score/i)
            print('\n')

            i=0
            score=0
            for img_path in img_files:
                if not any(ignore for ignore in exclude if ignore in img_path):
                    InputFile=join(op5,img_path)
                    trueText=join(ip1,img_path)
                    InputFile=InputFile.replace('tif','png')
                    trueText=trueText.replace('tif','txt')
                    ocr_result = OCR_evaluation.evaluateImage(InputFile, trueText,verbose=1)
                    if type(ocr_result) == float:
                        score += ocr_result
                    else:
                        score += 0
                    i=i+1
            print('Step accuracy - Combined Denoise: ',score/i)
            print('\n')
