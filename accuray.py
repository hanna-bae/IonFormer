from tqdm import tqdm 
import itertools
import torch 
import numpy as np 
import torch.nn as nn 

def epe(input_flow, target_flow):
    '''
    End-point-Error computatoin followed by DGC-Net
    Args:
        input_flow : estimated flow [BXHXWX2]
        target_flow : ground-truth flow [BXHXWX2]

    Output:
        Averaged end-point-error (value)
    '''
    return torch.norm(target_flow - input_flow, p=2, dim=1).mean()


def calculate_pck(model, data_loader, device, thershold_range, alptha=1, img_size=240):
    '''
    Compute PCK for HPatches dataset follwed by DGC-Net and GLU-Net
    Args:
        model : trained model 
        data_loader : input dataloader 
        device : 'cpu' or 'gpu'
        thershold_range : range of threshold 
        alpha : threshold to compute PCK
        img_size : size of input images

    Output: 
        pck 
    '''

    # followed by GLU-Net
    pck_1_over_image = []
    pck_5_over_image = []

    n_registered_pxs = 0.0
    array_n_correct_correspondences = np.zeros(thershold_range.shape, dtype=np.float32)

    # What does that mean ? 
    pbar = tqdm(enumerate(data_loader), total = len(data_loader))

    for batch, mini_batch in pbar:
        source_img = mini_batch['source_image']
        target_img = mini_batch['target_image']
        mask_gt = mini_batch['correspondence_mask'].to(device)
        flow_gt = mini_batch['flow map'].to(device)
        if flow_gt.shape[1] != 2:
            # shape is BXHXWX2
            flow_gt = flow_gt.permute(0,3,1,2)
            # BX2XHXW
        bs, ch_g, h_g, w_g = flow_gt.shape

        flow_estimated = model.estimate_flow(source_img, target_img, device, mode='channel_first')

        # resize the image 
        if flow_estimated.shape[2] != h_g or flow_estimated.shape[3] != w_g:
            ratio_h = float(h_g) / float(flow_estimated.shape[2])
            ratio_w = float(w_g) / float(flow_estimated.shape[3])
            flow_estimated = nn.functional.interpolate(flow_estimated, size=(h_g, w_g), mode='bilinear', 
                                                       align_corners=False)
            # nn.functional.interpolate -> upsampling
            flow_estimated[:, 0, :, :]*= ratio_w
            flow_estimated[:, 1, :, :]*= ratio_h
        assert flow_estimated.shape == flow_gt.shape

        # BXHXWX2
        flow_target_x = flow_gt.permute(0, 2, 3, 1)[:,:,:,0]
        flow_target_y = flow_gt.permute(0, 2, 3, 1)[:,:,:,1]
        flow_est_x = flow_estimated.permute(0, 2, 3, 1)[:,:,:,0] # Bxh_gxw_g
        flow_est_y = flow_estimated.permute(0, 2, 3, 1)[:,:,:,1]

        flow_target =\
            torch.cat((flow_target_x[mask_gt].unsqueeze(1),
                       flow_target_y[mask_gt].unsqueeze(1)), dim=1) # change the dimension and concat? 
        
        flow_est =\
            torch.cat((flow_est_x[mask_gt].unsqueeze(1),
                       flow_est_y[mask_gt].unsqueeze(1)), dim=1)

        img_size = max(mini_batch['source_image_size'][0])
        
        
