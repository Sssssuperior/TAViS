import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from skimage.measure import find_contours

from collections import defaultdict, deque
import time
from PIL import Image
import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F

import functools

from datetime import datetime, timedelta



def mask_iou(pred, target, eps=1e-7):
    r"""
    Calculate IoU for multiple categories.
    
    param:
        pred: size [N x C x H x W]
        target: size [N x H x W]
    output:
        iou_per_image: size [N x C], IoU for each image and each class
        mean_iou_per_class: size [C], mean IoU for each class
        mean_iou: scalar, mean IoU over all classes and images
    """
    assert len(pred.shape) == 4 and len(target.shape) == 3
    assert pred.shape[0] == target.shape[0] and pred.shape[2:] == target.shape[1:]
    
    N, C, H, W = pred.shape
    
    # Convert predictions to binary
    pred = (torch.sigmoid(pred) > 0.5).float()
    
    # Convert target to one-hot encoding
    target_one_hot = torch.zeros_like(pred)
    for c in range(C):
        target_one_hot[:, c, :, :] = (target == c).float()
    
    # Calculate intersection and union for each category and each image
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = (pred + target_one_hot).clamp(max=1).sum(dim=(2, 3))
    
    # Handle cases where a category is not present in both pred and target
    valid_mask = union > 0
    
    # Calculate IoU
    iou_per_image = torch.zeros_like(intersection)
    iou_per_image[valid_mask] = intersection[valid_mask] / (union[valid_mask] + eps)
    
    # Calculate mean IoU per class
    class_has_valid_pixels = valid_mask.sum(dim=0) > 0
    mean_iou_per_class = torch.zeros(C, device=pred.device)
    mean_iou_per_class[class_has_valid_pixels] = iou_per_image[:, class_has_valid_pixels].mean(dim=0)
    
    # Calculate overall mean IoU
    mean_iou = mean_iou_per_class[class_has_valid_pixels].mean()
    
    return iou_per_image, mean_iou_per_class, mean_iou
    


def save_single_mask(pred_mask, save_path, img_size=(224,224)):
    pred_mask = pred_mask.unsqueeze(dim=0).unsqueeze(dim=0)
    pred_mask = F.interpolate(
            pred_mask,
            img_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

    pred_mask = (torch.sigmoid(pred_mask) > 0.5).int()
    pred_mask = pred_mask.cpu().data.numpy().astype(np.uint8)
    pred_mask *= 255
    im = Image.fromarray(pred_mask).convert('P')
    im.save(save_path, format='PNG')



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



def save_mask_to_img(mask, path):
    mask = mask.cpu().data.numpy().astype(np.uint8)
    mask *= 255
    im = Image.fromarray(mask).convert('P')
    im.save(path, format='PNG')

def tensor2img(img, imtype=np.uint8, resolution=(224,224), unnormalize=True):
    img = img.cpu()
    if len(img.shape) == 4:
        img = img[0]
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    
    if unnormalize:
        img = img * std[:, None, None] + mean[:, None, None]
    
    img_numpy = img.numpy()
    img_numpy *= 255.0
    img_numpy = np.transpose(img_numpy, (1,2,0))
    img_numpy = img_numpy.astype(imtype)
    
    if resolution:
        img_numpy = cv2.resize(img_numpy, resolution) 

    return img_numpy



def normalize_img(value, vmax=None, vmin=None):
    '''
    Normalize heatmap
    '''
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


def vis_heatmap_bbox(heatmap_arr, img_array, img_name=None, bbox=None, ciou=None,  testset=None, img_size=224, save_dir=None ):
    '''
    visualization for both image with heatmap and boundingbox if it is available
    heatmap_array shape [1,1,14,14]
    img_array     shape [3 , H, W]
    '''
    if bbox == None:
        img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_img(-heatmap)

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
        
        # return np.array(heatmap_on_img)
        heatmap_on_img_BGR = cv2.cvtColor(heatmap_on_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir , heatmap_on_img_BGR )



    # Add comments
    else:
        img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        ori_img = img
        img = cv2.resize(img,(img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_img(-heatmap)

        # bbox = False
        if bbox:
            for box in bbox:
                lefttop = (box[0], box[1])
                rightbottom = (box[2], box[3])
                img = cv2.rectangle(img, lefttop, rightbottom, (0, 0, 255), 1)

        # img_box = img
        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

        # if ciou:
        #     cv2.putText(heatmap_on_img, 'IoU:' + '%.4f' % ciou , org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        #                fontScale=0.5, color=(255,255,255), thickness=1)

        if save_dir:
            save_dir = save_dir + '/heat_img_vis/' 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            heatmap_on_img_BGR = cv2.cvtColor(heatmap_on_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_dir +'/' + img_name + '_' + '%.4f' % ciou + '.jpg', heatmap_on_img_BGR )
        




def eval_pr(y_pred, y, num):
    prec, recall = torch.zeros(num, device=y_pred.device), torch.zeros(num, device=y_pred.device)
    thlist = torch.linspace(0, 1 - 1e-10, num, device=y_pred.device)
    
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    
    return prec, recall


def Eval_Fmeasure(pred, gt, beta2=0.3, pr_num=255):
    r"""
    Calculate F-measure for multiple classes.
    
    param:
        pred: size [N x C x H x W]
        gt: size [N x H x W]
        beta2: beta squared parameter for F-measure
        pr_num: number of precision-recall pairs to compute
    output:
        f_score_per_image: size [N x C], F-score for each image and each class
        mean_f_score_per_class: size [C], mean F-score for each class
        mean_f_score: scalar, mean F-score over all classes and images
    """
    assert len(pred.shape) == 4 and len(gt.shape) == 3
    assert pred.shape[0] == gt.shape[0] and pred.shape[2:] == gt.shape[1:]
    
    N, C, H, W = pred.shape
    
    f_score_per_image = torch.zeros((N, C), device=pred.device)
    
    for n in range(N):
        for c in range(C):
            pred_c = pred[n, c]  # [H, W]
            gt_c = (gt[n] == c).float()  # [H, W]
            
            # Skip if gt is all zeros for this class
            if torch.mean(gt_c) == 0.0:
                continue
            
            prec, recall = eval_pr(pred_c, gt_c, pr_num)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # Handle NaN
            f_score_per_image[n, c] = f_score.max()
    
    # Calculate mean F-score per class
    class_has_valid_pixels = (f_score_per_image > 0).sum(dim=0) > 0
    mean_f_score_per_class = torch.zeros(C, device=pred.device)
    mean_f_score_per_class[class_has_valid_pixels] = f_score_per_image[:, class_has_valid_pixels].mean(dim=0)
    
    # Calculate overall mean F-score
    mean_f_score = mean_f_score_per_class[class_has_valid_pixels].mean()
    
    return f_score_per_image, mean_f_score_per_class, mean_f_score



class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
        


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))




def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0




