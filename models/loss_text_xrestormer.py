import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.text_utils import text_to_task_list
from math import exp


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image

class L_Intensity_Max_RGB(nn.Module):
    def __init__(self):
        super(L_Intensity_Max_RGB, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, max_mode="l1"):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()

        fused_image = mask * image_infrared + (1 - mask) * image_visible
        if max_mode == "l1":
            Loss_intensity = F.l1_loss(fused_image, image_fused)
        else:
            Loss_intensity = F.mse_loss(fused_image, image_fused)
        return Loss_intensity

class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused))/2
        else:
            Loss_intensity = (F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused))/2
        return Loss_intensity

# use the GradientMaxLoss or L_Grad
class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        sobel_x = torch.FloatTensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = torch.FloatTensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + \
               F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.tensor_RGB2GRAY(image_visible)
        gray_infrared = self.tensor_RGB2GRAY(image_infrared)
        gray_fused = self.tensor_RGB2GRAY(image_fused)

        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)
        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)
    
    def tensor_RGB2GRAY(self, image):
        b,c,h,w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        device = img1.device
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        window = create_window(window_size)
        self.register_buffer('window', window)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.window.size(0):
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device)
            self.register_buffer('window', window)

        return ssim(img1, img2, window=window, window_size=self.window_size)


def structure_loss(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1 = F.conv2d(img1, window, padding=padd, groups=channel) - mu1
    sigma2 = F.conv2d(img2, window, padding=padd, groups=channel) - mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C2 = (0.03 * L) ** 2
    loss=(2*sigma12+C2)/(2*sigma1*sigma2+C2)

    if size_average:
        ret = loss.mean()
    else:
        ret = loss.mean(1).mean(1).mean(1)

    if full:
        return 1 - ret
    return ret

def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm

class TextFusionLoss(nn.Module):
    def __init__(self, feature_weight=1.0, base_weight=1.0, result_weight=1.0, feature_dim=[48, 16]):
        super(TextFusionLoss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=48)
        self.loss_func_Grad = GradientMaxLoss()
        self.loss_func_Max = L_Intensity_Max_RGB()
        self.loss_func_Consist = L_Intensity_Consist()
        self.loss_func_color = L_color()
        
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # 损失权重配置
        self.distill_weights = {
            'feature': feature_weight, 
            'base': base_weight,      
            'result': result_weight   
        }
        self.feature_adapters = nn.ModuleList([
            nn.Conv2d(feature_dim[0] * 2 ** 3, feature_dim[1] * 2 ** 3, 1),  # level 4: 384->128 
            nn.Conv2d(feature_dim[0] * 2 ** 2, feature_dim[1] * 2 ** 2, 1),   # level 3: 192->64
            nn.Conv2d(feature_dim[0] * 2 ** 1, feature_dim[1] * 2 ** 1, 1),    # level 2: 96->32
            nn.Conv2d(feature_dim[0] * 2 ** 1, feature_dim[1] * 2 ** 1, 1)     # level 1: 48->16
        ])
        self.task_configs = {
            'default': {
                'max_ratio': 6,
                'consist_ratio': 4,
                'text_ratio': 4,
                'ssim_ratio': 2,
                'color_ratio': 12
            }
        }

    def forward(self, image_A, image_B, image_fused, image_fused_distilled=None, stage='parent', text=None, 
                teacher_feats=None, student_feats=None):
        device = image_A.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dict = {}

        if stage == 'parent':
            tasks = text_to_task_list(text)
            batch_size = image_A.shape[0]
            total_ssim_loss = torch.tensor(0.0, device=device)
            total_max_loss = torch.tensor(0.0, device=device)
            total_color_loss = torch.tensor(0.0, device=device)
            total_grad_loss = torch.tensor(0.0, device=device)
            
            for batch_idx in range(batch_size):
                sample_tasks = tasks[batch_idx]
                sample_loss = torch.tensor(0.0, device=device, requires_grad=True)
                sample_ssim_loss = 0
                sample_max_loss = 0
                sample_color_loss = 0
                sample_grad_loss = 0
                
                for task in sample_tasks:
                    task_type = task['event']
                    config = self.task_configs.get(task_type, self.task_configs['default'])
                    loss_args = {
                        'image_visible': image_A[batch_idx:batch_idx+1],
                        'image_infrared': image_B[batch_idx:batch_idx+1],
                        'image_fused': image_fused[batch_idx:batch_idx+1],
                        'max_ratio': config.get('max_ratio', 6),
                        'consist_ratio': config.get('consist_ratio', 4), 
                        'text_ratio': config.get('text_ratio', 4),
                        'ssim_ratio': config.get('ssim_ratio', 2),
                        'color_ratio': config.get('color_ratio', 12),
                        'ir_compose': config.get('ir_compose', 1)
                    }
                    
                    loss, ssim, max_l, color_l, grad_l = self.compute_fusion_loss(**loss_args)
                    sample_loss = sample_loss + loss
                    sample_ssim_loss += ssim
                    sample_max_loss += max_l
                    sample_color_loss += color_l
                    sample_grad_loss += grad_l

                if len(sample_tasks) > 0:
                    sample_loss = sample_loss / len(sample_tasks)
                    sample_ssim_loss = sample_ssim_loss / len(sample_tasks)
                    sample_max_loss = sample_max_loss / len(sample_tasks)
                    sample_color_loss = sample_color_loss / len(sample_tasks)
                    sample_grad_loss = sample_grad_loss / len(sample_tasks)
                    
                total_loss = total_loss + sample_loss
                total_ssim_loss = total_ssim_loss + sample_ssim_loss
                total_max_loss = total_max_loss + sample_max_loss
                total_color_loss = total_color_loss + sample_color_loss
                total_grad_loss = total_grad_loss + sample_grad_loss

            batch_loss = total_loss / batch_size
            ssim_loss = total_ssim_loss / batch_size
            max_loss = total_max_loss / batch_size
            color_loss = total_color_loss / batch_size
            grad_loss = total_grad_loss / batch_size
            
            loss_dict.update({
                'ssim_loss': ssim_loss,
                'max_loss': max_loss,
                'color_loss': color_loss,
                'grad_loss': grad_loss
            })
            
        elif stage == 'distill':
            default_config = self.task_configs['default']
            fusion_loss_parent, ssim_loss_parent, max_loss_parent, color_loss_parent, grad_loss_parent = self.compute_fusion_loss(
                image_A, image_B, image_fused,
                **default_config 
            )

            base_loss_parent = fusion_loss_parent * self.distill_weights['base']

            if image_fused_distilled is not None:
                fusion_loss_distill, ssim_loss_distill, max_loss_distill, color_loss_distill, grad_loss_distill = self.compute_fusion_loss(
                    image_A, image_B, image_fused_distilled,
                    **default_config
                )
                
                base_loss_distill = fusion_loss_distill * self.distill_weights['base']
            else:
                fusion_loss_distill = torch.tensor(0.0, device=device)
                ssim_loss_distill = torch.tensor(0.0, device=device)
                max_loss_distill = torch.tensor(0.0, device=device)
                color_loss_distill = torch.tensor(0.0, device=device)
                grad_loss_distill = torch.tensor(0.0, device=device)
                base_loss_distill = torch.tensor(0.0, device=device)
            
            if teacher_feats is not None and student_feats is not None:
                feature_loss = 0
                for i, (t_feat, s_feat) in enumerate(zip(teacher_feats, student_feats)):
                    # # debug
                    # print(i)
                    # print(t_feat.shape)
                    # print(s_feat.shape)
                    t_feat = self.feature_adapters[i](t_feat) 
                    t_feat = F.normalize(t_feat, p=2, dim=1)
                    s_feat = F.normalize(s_feat, p=2, dim=1)
                    feature_loss += self.smooth_l1_loss(s_feat, t_feat)
                feature_loss *= self.distill_weights['feature']
                loss_dict['feature_loss'] = feature_loss
            else:
                feature_loss = torch.tensor(0.0, device=device)
            
            if image_fused_distilled is not None:
                result_loss = self.smooth_l1_loss(image_fused, image_fused_distilled) * self.distill_weights['result']
                loss_dict['result_loss'] = result_loss
            else:
                result_loss = torch.tensor(0.0, device=device)
                
            total_loss = base_loss_parent + base_loss_distill + feature_loss + result_loss
            
            loss_dict.update({
                'base_loss_parent': base_loss_parent,
                'base_loss_distill': base_loss_distill,
                'ssim_loss_parent': ssim_loss_parent,
                'ssim_loss_distill': ssim_loss_distill,
                'max_loss_parent': max_loss_parent,
                'max_loss_distill': max_loss_distill,
                'color_loss_parent': color_loss_parent,
                'color_loss_distill': color_loss_distill,
                'grad_loss_parent': grad_loss_parent,
                'grad_loss_distill': grad_loss_distill,
                'result_loss': result_loss
            })
            
        return total_loss, loss_dict

    def compute_fusion_loss(self, image_visible, image_infrared, image_fused, 
                          max_ratio=4, consist_ratio=1, ssim_ratio=1, 
                          text_ratio=10, color_ratio=12, ir_compose=1,
                          ssim_ir_ratio=1, max_mode="l1", consist_mode="l1"):
        device = image_visible.device
        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)
        
        loss_ssim = ssim_ratio * (
            self.loss_func_ssim(image_visible, image_fused) + 
            ssim_ir_ratio * self.loss_func_ssim(image_infrared_gray, image_fused_gray)
        )
        loss_max = max_ratio * self.loss_func_Max(image_visible, image_infrared, image_fused, max_mode)
        loss_consist = consist_ratio * self.loss_func_Consist(
            image_visible_gray, image_infrared_gray, image_fused_gray, ir_compose, consist_mode
        )
        loss_color = color_ratio * self.loss_func_color(image_visible, image_fused)
        loss_text = text_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)
        
        total_loss = loss_ssim + loss_max + loss_consist + loss_color + loss_text
        return total_loss, loss_ssim, loss_max, loss_color, loss_text

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray