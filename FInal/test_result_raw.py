#!/usr/bin/env python
import torch
import csv
import torchvision
from baseline_nlfc import *
import torch.nn as nn
import time
import argparse

import numpy as np
import os.path
import shutil
from scipy.io.matlab.mio import savemat, loadmat


def vali_pre_process(img,bayer_pattern):
    if bayer_pattern=="RGGB":
        img = np.pad(img, ((1,1),(1,1)), 'reflect')
        img = np.pad(img, ((128,128+30),(128,128+30)), 'reflect')

    elif bayer_pattern=="GRBG":
        img = np.pad(img, ((1,1),(0,0)), 'reflect')
        img = np.pad(img, ((128,128+30),(128,128)), 'reflect')

    else:
        img = np.pad(img, ((128,128),(128,128)), 'reflect')

    return img


def vali_post_process(img,bayer_pattern):

    if bayer_pattern=="RGGB":
        img=img[129:-(129+30),129:-(129+30)]
    elif bayer_pattern=="GRBG":
        img=img[129:-(129+30),128:-128]
    else:
        img=img[128:-128,128:-128]

    return img
#  0: no change, 1~3:roat 90~270 4: horizon flip-0, 5: horizon flip-90, 6: horizon flip-180, 7: horizon flip-270
def vali_rot_pre_process(img,rot):
    if rot == 0:
        return img
    elif rot == 1:
        img = np.pad(img, ((1,1),(0,0)), 'reflect')
        img = np.pad(img, ((0,30),(0,0)), 'reflect')
        img = np.rot90(img,-rot)
        return img
    elif rot == 2:
        img = np.pad(img, ((1,1),(1,1)), 'reflect')
        img = np.pad(img, ((0,30),(0,30)), 'reflect')
        img = np.rot90(img,-rot)
        return img
    elif rot == 3:
        img = np.pad(img, ((0,0),(1,1)), 'reflect')
        img = np.pad(img, ((0,0),(0,30)), 'reflect')
        img = np.rot90(img,-rot)
        return img

    elif rot == 4:
        img = np.pad(img, ((0,0),(1,1)), 'reflect')
        img = np.pad(img, ((0,0),(0,30)), 'reflect')
        img = np.flip(img , 1)
        return img
    elif rot== 5:
        img = np.pad(img, ((1,1),(1,1)), 'reflect')
        img = np.pad(img, ((0,30),(0,30)), 'reflect')
        img = np.flip(img , 1)
        img = np.rot90(img,-(rot-4))
        return img
        
    elif rot == 6:
        img = np.pad(img, ((1,1),(0,0)), 'reflect')
        img = np.pad(img, ((0,30),(0,0)), 'reflect')
        img = np.flip(img , 1)
        img = np.rot90(img,-(rot-4))
        return img
    elif rot == 7:
        img = np.flip(img , 1)
        img = np.rot90(img,-(rot-4))
        return img


def vali_rot_post_process(img,rot):
    if rot==0:
        return img
    elif rot==1:
        img = np.rot90(img,rot)
        img = img[1:-31,:]
        return img
    elif rot==2:
        img = np.rot90(img,rot)
        img = img[1:-31,1:-31]
        return img
    elif rot==3:
        img = np.rot90(img,rot)
        img = img[:,1:-31]
        return img
    elif rot==4:
        img = np.flip(img , 1)
        img = img[:,1:-31]
        return img
    elif rot==5:
        img = np.rot90(img,(rot-4))
        img = np.flip(img , 1)
        img = img[1:-31,1:-31]
        return img
    elif rot==6:
        img = np.rot90(img,(rot-4))
        img = np.flip(img , 1)
        img = img[1:-31,:]
        return img
    elif rot==7:
        img = np.rot90(img,(rot-4))
        img = np.flip(img , 1)
        return img



def vali_pack(img):
    img_pack=np.zeros((1,4,int(img.shape[0]/2),int(img.shape[1]/2)))
    img_pack[0,0,:,:]=img[::2,::2]
    img_pack[0,1,:,:]=img[::2,1::2]
    img_pack[0,2,:,:]=img[1::2,::2]
    img_pack[0,3,:,:]=img[1::2,1::2]

    return img_pack

def vali_unpack(img):
    img_unpack=np.zeros((int(img.shape[2]*2),int(img.shape[3]*2)))
    img_unpack[::2,::2]=img[0,0,:,:]
    img_unpack[::2,1::2]=img[0,1,:,:]
    img_unpack[1::2,::2]=img[0,2,:,:]
    img_unpack[1::2,1::2]=img[0,3,:,:]

    return img_unpack



def denoiser(noisy,net,img_bayer_pattern,nlfs,Device):
    # TODO: plug in your denoiser here
    Input=noisy

    #Input=vali_pack(vali_pre_process(Input,img_bayer_pattern))
    #Input=torch.Tensor(Input).to(Device)
    #denoised = net(Input)
    #denoised = denoised.detach().cpu().numpy()
    #denoised=vali_unpack(denoised)
    #denoised=vali_post_process(denoised,img_bayer_pattern)
    pred0=noisy.copy()*0
    Runtime=0
    pixels=0
    with torch.no_grad():
        for m in range(0,8):
            Input=vali_pre_process(noisy,img_bayer_pattern)
            Input=vali_rot_pre_process(Input,m)
            Input=vali_pack(Input)
            pixels += Input.shape[0]*Input.shape[1]*Input.shape[2]*Input.shape[3]
            nlf_patch=np.ones((1,2,Input.shape[2],Input.shape[3])).astype(np.float32)
            nlf_patch[0,0,:,:]=nlf_patch[0,0,:,:]*(nlfs[0]+nlfs[2]+nlfs[4])
            nlf_patch[0,1,:,:]=nlf_patch[0,1,:,:]*(nlfs[1]+nlfs[3]+nlfs[5])
            Input=np.concatenate((Input,nlf_patch),1)
    
            Input=torch.Tensor(Input).to(device)
            torch.cuda.synchronize()
            start=time.time()
            pred = net(Input)
            torch.cuda.synchronize()
            end=time.time()
            Runtime+=(end-start) 
            Input = pred.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()

            pred=vali_unpack(pred)
            pred=vali_rot_post_process(pred,m)
            pred=vali_post_process(pred,img_bayer_pattern)
            pred0+=pred
    #end=time.time()

    denoised=pred0/8
    pixels=pixels/8

    #print(end-start,',',pixels)
    Runtime=Runtime/(pixels/1e6)
    return denoised,Runtime


parser = argparse.ArgumentParser()

parser.add_argument('--Model_dir', type=str, default='./checkpoints/Denoising_nlfc_res512_dct/', help='Model directory')
parser.add_argument('--Model', type=str, default='190_networks.pth', help='Name of the model')
parser.add_argument('--GPU_id', type=str, default='2', help='GPU_id')

args = parser.parse_args()

# TODO: Download noisy images from:
#  https://competitions.codalab.org/my/datasets/download/4d26bd29-ab8b-4fe7-8fa1-7a33b32154c7

# TODO: update your working directory; it should contain the .mat file containing noisy images
work_dir = './'
Model_dir = args.Model_dir
Model = args.Model
# load noisy images
noisy_fn = 'siddplus_test_noisy_raw.mat'
noisy_key = 'siddplus_test_noisy_raw'
noisy_mat = loadmat(os.path.join(work_dir, noisy_fn))[noisy_key]

bayer_pattern = os.path.join(work_dir, "siddplus_bayer_patterns.csv")
bayer = {}
with open(bayer_pattern, 'r') as f:

    reader = csv.reader(f)
    i=0
    for row in reader:
        bayer[i] = row[0]
        i+=1

nlfs = []

with open( os.path.join(work_dir, "siddplus_test_nlfs.csv")) as file: 
    next(file)
    for line in file:
        line = line.strip('\n')
        line = line.split(",")
        nlf = map(float, line)
        nlf=list(nlf)
        nlfs.append(nlf)


#network load
device = torch.device("cuda:" + str(args.GPU_id) if torch.cuda.is_available() else "cpu")
#net = Baseline_classic()
net = Baseline()
net = net.to(device)
net.load_state_dict(torch.load(os.path.join(Model_dir, Model)))


# denoise
n_im, h, w = noisy_mat.shape
results = noisy_mat.copy()
all_run_time = 0
for i in range(n_im):
    print(i)
    noisy = np.reshape(noisy_mat[i, :, :], (h, w))
    denoised,runtime_sig = denoiser(noisy,net,bayer[i//32],nlfs[i//32],device)
    results[i, :, :] = denoised
    print(runtime_sig)
    all_run_time+=runtime_sig
# create results directory
res_dir = 'res_dir'
os.makedirs(os.path.join(work_dir, res_dir), exist_ok=True)

# save denoised images in a .mat file with dictionary key "results"
res_fn = os.path.join(work_dir, res_dir, 'results.mat')
res_key = 'results'  # Note: do not change this key, the evaluation code will look for this key
savemat(res_fn, {res_key: results})

# submission indormation
# TODO: update the values below; the evaluation code will parse them
runtime = all_run_time/n_im  # seconds / megapixel
cpu_or_gpu = 0  # 0: GPU, 1: CPU
use_metadata = 1  # 0: no use of metadata, 1: metadata used
other = '(optional) any additional description or information'

# prepare and save readme file
readme_fn = os.path.join(work_dir, res_dir, 'readme.txt')  # Note: do not change 'readme.txt'
with open(readme_fn, 'w') as readme_file:
    readme_file.write('Runtime (seconds / megapixel): %s\n' % str(runtime))
    readme_file.write('CPU[1] / GPU[0]: %s\n' % str(cpu_or_gpu))
    readme_file.write('Metadata[1] / No Metadata[0]: %s\n' % str(use_metadata))
    readme_file.write('Other description: %s\n' % str(other))

# compress results directory
res_zip_fn = 'results_dir'
shutil.make_archive(os.path.join(work_dir, res_zip_fn), 'zip', os.path.join(work_dir, res_dir))

#  TODO: upload the compressed .zip file here:
#  https://competitions.codalab.org/competitions/22230#participate-submit_results

