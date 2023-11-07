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


def correct_correspondences(input_flow, target_flow, alpha, img_size):
    '''
    Computation PCK, i.e. number of pixels within a certain threshold
    Args:
        input_flow : estimated flow [BXHXW, 2]
        target_flow : ground-truth flow [BXHXW, 2]
        alpha : thresold
        img_size : image size

    Output:
        PCK metric 
    '''
    # input flow is shape(BXH_gtXW_gt, 2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    pck_threshold = alpha * img_size
    mask = dist.le(pck_threshold) # 1 if dist <= pck_threshold, 0 else
    return mask.sum().item()


def calculate_pck(model, data_loader, device, alptha=1, img_size=240):
    '''
    Compute PCK for HPatches dataset follwed by DGC-Net and GLU-Net
    Args:
        model : trained model 
        data_loader : input dataloader 
        device : 'cpu' or 'gpu'
        alpha : threshold to compute PCK
        img_size : size of input images

    Output: 
        pck 
    '''

    # followed by GLU-Net
    pck_1_over_image = []
    pck_5_over_image = []

    n_registered_pxs = 0.0


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

        img_size = max(mini_batch['source_image_size'][0], mini_batch['source_image_size'][1]).float().to(device)
        px_1 = correct_correspondences(flow_est, flow_target, alpha=1.0/float(img_size),img_size=img_size)
        px_5 = correct_correspondences(flow_est, flow_target, alpha=5.0/float(img_size), img_size = img_size)

        pck_1_over_image.append(px_1/flow_target.shape[0])
        pck_5_over_image.append(px_5/flow_target.shape[0])
        
        output = {'pck_1_over_image' : np.mean(pck_1_over_image),
                  'pck_5_over_image' : np.mean(pck_5_over_image)}
        
        return output

def calculate_epe_hpatches(model, data_loader, device, img_size=240):
    '''
    Compute EPE for HPatches dataset
    Args:
        model: trained model
        data_loader: input dataloader
        device: 'cpu' or 'gpu'
        img_size: size of input images

    Output:
        aepe_array: averaged EPE for the whole sequence of HPatches
    '''

    aepe_array = []
    n_registered_pxs = 0

    pbar = tqdm(enumerate(data_loader), total = len(data_loader))
    for _, mini_batch in pbar:

        source_img = mini_batch['source_image'].to(device)
        target_img = mini_batch['target_image'].to(device)
        bs, _, _, _ = source_img.shape

        # model prediction
        estimated_grid, estimated_mask = model(source_img, target_img)

        flow_est = estimated_grid[-1].permute(0, 2, 3, 1).to(device)
        flow_target = mini_batch['correspondence_map'].to(device)

        # applying mask 
        mask_x_gt = \
            flow_target[:,:,:,0].ge(-1) & flow_target[:,:,:,0].le(1)
        mask_y_gt = \
            flow_target[:,:,:,1].ge(-1) & flow_target[:,:,:,1].le(1)
        mask_xx_gt = mask_x_gt & mask_y_gt
        mask_gt = torch.cat((mask_xx_gt.unsqueeze(3),
                             mask_xx_gt.unsqueeze(3)), dim=3)
        
        for i in range(bs):
            # unnormalize the flow: [-1; 1] -> [0; im_size - 1]
            flow_target[i] = (flow_target[i] + 1) * (img_size - 1)/2
            flow_est[i] = (flow_est[i] + 1) * (img_size - 1)/2

        flow_target_x = flow_target[:,:,:,0]
        flow_target_y = flow_target[:,:,:,1]
        flow_est_x = flow_est[:,:,:,0]
        flow_est_y = flow_est[:,:,:,1]

        flow_target = \
            torch.cat((flow_target_x[mask_gt[:,:,:,0]].unsqueeze(1),
                       flow_target_y[mask_gt[:,:,:,1]].unsqueeze(1)), dim=1)
        
        flow_est = \ 
            torch.cat((flow_est_x[mask_gt[:,:,:,0]].unsqueeze(1),
                       flow_est_y[mask_gt[:,:,:,1]].unsqueeze(1)), dim=1)
        
        aepe = epe(flow_est, flow_target)
        aepe_array.append(aepe.item())
        n_registered_pxs += flow_target.shape[0]
    
    return aepe_array 
