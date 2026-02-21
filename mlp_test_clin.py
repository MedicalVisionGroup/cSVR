import os
import gc
import math
import os.path as path
import torch 
import os
torch.cuda.synchronize()
import interpol
import models
import models.losses
import datasets
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
import cornucopia as cc
from matplotlib import pyplot as plt
import pdb
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
from pytorch_lightning import seed_everything
import sys
import cornucopia 
from cornucopia.utils.warps import affine_flow
from grid_utils import og_slice_pos_pre, make_grid_one, divide_into_stacks
import sys


# CHECK GPU
if torch.cuda.is_available():
    print("CUDA AVAILABLE")
else:
    print("no CUDA")


# SET RUNNING PARAMETERS
seed_num = 1 #2
seed_everything(seed_num, workers=True)
save_images = True #False #True #True
subsample=1
img_num = 0 # 0 # which image in dataset want to test
tot_test = 9 # CHANGE THIS TO 3 when doing actual tests
imgnum_test = 0



# LOAD MODEL
folder = 'run5'
model1024 = True
model256 = False
model512 = False
mlp_test = True
slice_size = 128
ckpt_path_recon = "feta3d0_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_multi_crop_l22_loss_grid_multiscale_rot40_nodes100_bigger_model_v2_out_plane12_augs_v2_"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_onehot_loss_40_0.03_0.1_70_12_argparse_test"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_onehot_loss_40_0.03_0.1_0_0_mlp_diff_order"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_onehot_loss_40_0.03_0.1_360_12_mlp_diff_order"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_mlp_rot_also"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_lr0.5"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_onehot_loss_40_0.03_0.1_180_12_lr5e-05"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_mlp_rot_also"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_lr0.5"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_onehot_loss_40_0.03_0.1_180_12_lr5e-05"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_lr0.5"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_stack_order_40_0.03_0.1_180_12_lr1e-05_stack_predict_loss_fix_loss_sigmoid3_print_vals"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_stack_order_40_0.03_0.1_180_12_lr1e-05_more_channels_bigger_image_more_slices32"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_stack_order_40_0.03_0.1_180_12_lr1e-05_more_channels_bigger_image_more_slices32"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_stack_order_40_0.03_0.1_0_12_lr1e-05_slices32_fix_loss_no_rot_h200_no_plane_rot_2"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_stack_order_40_0.03_0.1_180_12_lr1e-05_slices32_fix_loss_no_rot_h200_no_plane_rot"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_h200"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_bigger_mlp"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss2_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_loss_smooth_huge_mlp4"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss2_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_loss_smooth_huge_mlp4_drop_slice0.4"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_bigger_mlp"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss2_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_loss_smooth_huge_mlp4"
ckpt_path_recon = "feta3d0_mlp_multi_stack_svr_final_sb2_crop_flow_SNet3d2_1024_MLP_classification_multihot_loss2_40_0.03_0.1_180_12_lr1e-05_mlp_norm_64size_vec_rots_loss_smooth_huge_mlp4_drop_slice0.6"


root_path_mine = '/data/vision/polina/users/mfirenze/svr_my_train_2024/checkpoints/'
root_path_mine = '/data/vision/polina/users/mfirenze/cSVR/checkpoints/'

model_name = ckpt_path_recon[-15:]
print(model_name)#
trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
if model1024:
   trainee = models.segment(model=models.flow_SNet3d2_1024_multi_crop())
   if(mlp_test):
     trainee = models.segment(model=models.flow_SNet3d2_1024_MLP())
elif model512:
  trainee = models.segment(model=models.flow_SNet3d2_512_multi_crop())
elif model256:
   trainee = models.segment(model=models.flow_SNet3d2_256_multi_crop())

print("no load!!")
#trainee.load_state_dict(torch.load(path.join(root_path_mine, ckpt_path_recon, 'last.ckpt'))['state_dict'], strict = False)
trainee.load_state_dict(torch.load(path.join(root_path_mine, ckpt_path_recon, 'last.ckpt'))['state_dict'], strict = False)

#(path.join(root_path_mine, ckpt_path_recon, 'last.ckpt'))['state_dict'])
model = trainee.model.cuda()
#sets = datasets.feta3d0_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3)

sets = datasets.feta3d0_mlp_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3, rotations=20, translations=0.03, bulk_rotations_plane=180, bulk_rotations_tr_plane=12)
sets = datasets.feta3d0_mlp_multi_stack_svr_final_sb2_crop(subsample=subsample, zooms=0.3, rotations=20, translations=0.03, bulk_rotations_plane=0, bulk_rotations_tr_plane=12)

# SET initial PARAMETERS
avg_loss = 0
flip_row_12 = np.array([[-1, 0, 0, 0], # SAVE IN SAME COORDINATE SYSTEM AS 
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0,  0, 0, 1]])
loss_2 = 0

num_correct = 0
for imgnum in range(img_num, img_num+tot_test): #range(len(sets[1])):
    with torch.no_grad():
        # Get original dataset before transforms
        true, segs, _ = sets[1].__getitem__(imgnum, gpu=False)  # sets[1] gets validation set
        true = true.cuda()  # torch.Size([1, 256, 256, 256]), true.min = 0  true.max = 1490.2251
        segs = segs.cuda() #torch.Size([1, 256, 256, 256])
        item = sets[1].transforms(true, segs, cpu=False, gpu=True) # this is where the augementation happens
        if not mlp_test:
          mask = item[1][None][:,-1:]
          target = item[1][None,:,:,::2,::2]
          target[:,0:3] = target[:,0:3]*0.5 
        else :  
          target = item[1]

        init_stacks = item[0][1].clone()
        init_stacks[:,:,0:3,3] = init_stacks[:,:,0:3,3]/2

        start = time.time()
        downsampled_input = item[0][0][None,:,:,::2,::2]



        if(imgnum >= 5):
           line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/test/test.txt").read().splitlines()[imgnum_test]
           downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_rot_cm_t.pt' % line).to(item[0][0].device) #[:,:,:-1]
           init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_rot_cm_t.pt' % line).to(item[0][0].device) #[:,:-1]
           
         #  init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_no_rot.pt' % line).to(item[0][0].device) #[:,:-1]
         #  downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_no_rot.pt' % line).to(item[0][0].device) #[:,:,:-1]
           
           
           imgnum_test = imgnum_test + 1


        else:
          line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/validation/val.txt").read().splitlines()[imgnum]
          init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_rot_cm.pt' % line).to(item[0][0].device) #[:,:-1]
          downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_rot_cm.pt' % line).to(item[0][0].device) #[:,:,:-1]


         # init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_no_rot.pt' % line).to(item[0][0].device) #[:,:-1]
         # downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_no_rot.pt' % line).to(item[0][0].device) #[:,:,:-1]
        #line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/test/test.txt").read().splitlines()[imgnum]
        # print("new auto2")
        init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_auto2.pt' % line).to(item[0][0].device) #[:,:-1]
        downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_auto2.pt' % line).to(item[0][0].device) #[:,:,:-1]


      #  line = open("/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/test/test.txt").read().splitlines()[imgnum]
       # downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_rot_cm.pt' % line).to(item[0][0].device) #[:,:,:-1]
       # downsampled_input = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/%s_rot_cm_t.pt' % line).to(item[0][0].device) #[:,:,:-1]

        #init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_rot_cm.pt' % line).to(item[0][0].device) #[:,:-1]
       # init_stacks = torch.load('/data/vision/polina/users/mfirenze/svr_my_train_2024/data_sampled/init_stack_%s_rot_cm_t.pt' % line).to(item[0][0].device) #[:,:-1]

      # #  pdb.set_trace()

        stack1, stack2 ,stack3 = divide_into_stacks(init_stacks, downsampled_input)
        stack = model((downsampled_input,init_stacks))


        stack_pred = stack[:,0:6]
        order_reverse = stack_pred.argmax(dim=1)  % 2

        rot_pred = stack[:,6:]
        rot90_amount = rot_pred.argmax(dim=1)  % 4
        
      
        
        stacks = [stack1, stack2, stack3]

        for i in range(3):
 #  stacks[i] = stacks[i].flip(dims=[2]) # make this 18-
            
            
            if rot90_amount[i] == 1:
              stacks[i] = torch.rot90(stacks[i], k=-1, dims=(3,4))
            if rot90_amount[i] == 2:
              stacks[i] = torch.rot90(stacks[i], k=1, dims=(3,4))
            if rot90_amount[i] == 3:
              stacks[i] = torch.rot90(stacks[i], k=2, dims=(3,4))

            if order_reverse[i] == 1:
               stacks[i] = torch.rot90(stacks[i], k=2, dims=(2,3))
           

        stack1, stack2, stack3 = stacks
      



    #    stack = target
    #    pdb.set_trace() #models.losses.cclassification_onehot_loss(stack, torch.tensor([[0,1,0]]).cuda())
        end = time.time()
        # print(f"Output")
        # print(stack)
        # print("Ground truth")
        # print(target)
        num_stacks = 6 # 3
        num_rots = 4
       # num_rots = 0
        num_order = 2


        idx_stack = torch.argmax(stack[:,0:num_stacks], dim=1)
        one_hot_stack = F.one_hot(idx_stack, num_classes = num_stacks).float()

        if(num_rots > 0):
          idx_rot = torch.argmax(stack[:,num_stacks:num_stacks+num_rots], dim=1)
          one_hot_rot = F.one_hot(idx_rot, num_classes = num_rots).float()
        #  idx_order = (torch.sigmoid(stack[:,-1]) > 0.5).float().unsqueeze(1)
        #  one_hot = torch.cat((one_hot_stack, one_hot_rot, idx_order), dim=1)
          one_hot = torch.cat((one_hot_stack, one_hot_rot), dim=1)
        else:
          idx_order =  torch.argmax(stack[:,-num_order:], dim=1)
          one_hot_order = F.one_hot(idx_order, num_classes = num_order).float()
          one_hot = torch.cat((one_hot_stack, one_hot_order), dim=1)
           
        
        print(stack)
        print("One hot new model val, img_num" + str(imgnum))
        print(one_hot)

      #   target= torch.tensor([
      #       [1,0,0, 1,0,0,0,0],
      #       [0,1,0, 1,0,0,0,0],
      #       [0,0,1, 1,0,0,0,0],
      #   ], dtype=torch.float, device='cuda:0')
        
      #   similar = (one_hot == target).all(dim=1)
      #   is_correct = torch.sum(similar)/similar.shape[0]

      #  # is_correct = (one_hot == target).float().mean()
      #   print(is_correct)
      #   num_correct = num_correct + is_correct.item()
        # if(is_correct ==1):
        #     num_correct += 1
  
        
      #  print(models.losses.classification_multihot_loss(stack, target.cuda()))
        if (save_images==True ) :
              imgnames = ['input', 'mask', 'stack1', 'stack2', 'stack3']
              
              #imgnames = ['input','input_ds','splat','splat_gt','target_before','target_up','target_all','splat_inter']
              imgs = [downsampled_input[0][0].detach(), downsampled_input[0][1].detach(), stack1[0][0].detach(), stack2[0][0].detach(), stack3[0][0].detach()]

        if save_images:
            imgs = [img.cpu() for img in imgs]
        for i in range(len(imgs)):
          #  imgs[i] = torch.flip(imgs[i], dims=[0, 1])  # flip Z and Y
            initial_np = imgs[i].numpy()                
            # nii_image = nib.Nifti1Image(initial_np, affine=flip_row_12*0.8)  # You might need to specify the affine transformation matrix
            I = np.eye(4)
            I[0:3,0:3] = I[0:3,0:3]*1.406
            nii_image = nib.Nifti1Image(initial_np, affine=I)  
            #  nii_image = nib.Nifti1Image(initial_np, affine=np.eye(4))  # You might need to specify the affine transformation matrix
            nib.save(nii_image, '/data/vision/polina/users/mfirenze/cSVR/outputs_clin/vol_%d_seed_%d_model_%s_%s_val.nii.gz' % (imgnum, seed_num, model_name, imgnames[i]))
            print("DONE 1:D")
print("ESTIMATION QUALITY")
print(num_correct/tot_test)

