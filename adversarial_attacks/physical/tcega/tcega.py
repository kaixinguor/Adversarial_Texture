import os
import torch
import itertools
import json
from tqdm import tqdm
import numpy as np
from easydict import EasyDict
from scipy.interpolate import interp1d
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torchvision import transforms

# 临时使用yolov2，后续扩展检测器后再整理
from adversarial_attacks.detectors.yolo2 import load_data
from adversarial_attacks.detectors.yolo2 import utils as yolo2_utils
from adversarial_attacks.detectors.load_models import load_models
from .tps_grid_gen import TPSGridGen
from .generator_dim import GAN_dis
from .cfg import get_cfgs

unloader = transforms.ToPILImage()

def random_crop(cloth, crop_size, pos=None, crop_type=None, fill=0):
    """
    @adversarial_texture/train_utils.py random_crop
    """
    w = cloth.shape[2]
    h = cloth.shape[3]
    if crop_size == 'equal':
        crop_size = [w, h]
    if crop_type is None:
        d_w = w - crop_size[0]
        d_h = h - crop_size[1]
        if pos is None:
            r_w = np.random.randint(d_w + 1)
            r_h = np.random.randint(d_h + 1)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
        else:
            r_w = pos[0]
            r_h = pos[1]

        p1 = max(0, 0 - r_h)
        p2 = max(0, r_h + crop_size[1] - h)
        p3 = max(0, 0 - r_w)
        p4 = max(0, r_w + crop_size[1] - w)
        cloth_pad = F.pad(cloth, [p1, p2, p3, p4], value=fill)
        patch = cloth_pad[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    elif crop_type == 'recursive':
        if pos is None:
            r_w = np.random.randint(w)
            r_h = np.random.randint(h)
        elif pos == 'center':
            r_w, r_h = (np.array(cloth.shape[2:]) - np.array(crop_size)) // 2
            if r_w < 0:
                r_w = r_w % w
            if r_h < 0:
                r_h = r_h % h
        else:
            r_w = pos[0]
            r_h = pos[1]
        expand_w = (w + crop_size[0] - 1) // w + 1
        expand_h = (h + crop_size[1] - 1) // h + 1
        cloth_expanded = cloth.repeat([1, 1, expand_w, expand_h])
        patch = cloth_expanded[:, :, r_w:r_w + crop_size[0], r_h:r_h + crop_size[1]]

    else:
        raise ValueError
    return patch, r_w, r_h

class TCEGA:
    """
    目标检测TCEGA对抗攻击工具类
    
    参数:
        model_name: 预训练的目标检测模型 (默认为YOLOv2)
        method: 攻击方法 ('RCA', 'TCA', 'EGA', 'TCEGA')
        class_mapping: 类别标签映射字典
        device: 计算设备 ('cuda' 或 'cpu')

        epsilon: 总扰动上限
        alpha: 单次迭代扰动大小
        num_iter: 迭代次数
    """

    def __init__(self,
                 method='TCEGA',
                 model_name='yolov2',
                 class_mapping=None, 
                 device=None,
                 conf_thresh=0.5, # 检测后处理参数
                 nms_thresh=0.4, # 检测后处理参数
                 args=None,
                 kwargs=None):
        
        self.model_name = model_name
        self.method = method
        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # 加载模型
        self._load_model(model_name)
            
        # 加载类别名称
        if class_mapping is None:
            self.class_names = yolo2_utils.load_class_names('./data/coco.names')
        else:
            self.class_names = class_mapping
            
        # 初始化组件
        self._init_components()

        # 加载预训练的对抗攻击模型
        default_load_path = {'RCA': 'pretrained/RCA2.npy',
                             'TCA': 'pretrained/TCA.npy',
                             'EGA': 'pretrained/EGA.pkl',
                             'TCEGA': 'pretrained/TCEGA_z.npy'}
        
        if self.method in ['RCA', 'TCA', 'EGA']:
            self._load_pretrained_attack(default_load_path[self.method])
        elif self.method in ['TCEGA']:
            self._load_pretrained_attack(default_load_path['EGA'], default_load_path['TCEGA'])
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # self.adv_patch = self.generate_adv_patch()
    
    def _load_model(self, model_name):
        """加载模型"""
        if model_name == "yolov2":
            args, kwargs = get_cfgs('yolov2', self.method)
            print("model cfg", args, kwargs)
            self.model = load_models(**kwargs)
            self.model = self.model.eval().to(self.device)
            self.args = args
            self.kwargs = kwargs
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
    def _init_components(self):
        """初始化TCEGA所需的组件"""
        target_func = lambda obj, cls: obj
        self.patch_applier = load_data.PatchApplier().to(self.device)
        self.patch_transformer = load_data.PatchTransformer().to(self.device)
        self.prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, 
                                                       self.kwargs['name'] if self.kwargs else 'yolov2').to(self.device)
        self.total_variation = load_data.TotalVariation().to(self.device)
        
        # TPS网格生成器
        target_control_points = torch.tensor(list(itertools.product(
            torch.arange(-1.0, 1.00001, 2.0 / 4),
            torch.arange(-1.0, 1.00001, 2.0 / 4),
        )))
        self.tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
        self.tps.to(self.device)

    def _load_pretrained_attack(self, load_path=None, load_path_z=None):
        """加载预训练的对抗攻击模型"""
        if self.method in ['RCA', 'TCA']:
            if load_path is None:
                result_dir = f'./results/result_{self.model_name}_{self.method}'
                img_path = os.path.join(result_dir, f'patch{self.args.n_epochs}.npy')
            else:
                img_path = load_path
            self.cloth = torch.from_numpy(np.load(img_path)[:1]).to(self.device)
            self.test_cloth = self.cloth.detach().clone()
            self.test_gan = None
            self.test_z = None
            self.test_type = 'patch'
            
        elif self.method in ['EGA', 'TCEGA']:
            self.gan = GAN_dis(DIM=128, z_dim=128, img_shape=(324, )*2)
            if load_path is None:
                result_dir = f'./results/result_{self.model_name}_{self.method}'
                cpt = os.path.join(result_dir, f'{self.model_name}_{self.method}.pkl')
            else:
                cpt = load_path
            d = torch.load(cpt, map_location='cpu')
            self.gan.load_state_dict(d)
            self.gan.to(self.device)
            self.gan.eval()
            for p in self.gan.parameters():
                p.requires_grad = False
            self.test_cloth = None
            self.test_gan = self.gan
            
            if self.method == 'EGA':
                self.test_z = None
                self.test_type = 'gan'
                self.cloth = self.gan.generate(torch.randn(1, 128, *self.args.z_size, device=self.device))
            else:  # TCEGA
                if load_path_z is None:
                    result_dir = f'./results/result_{self.model_name}_{self.method}'
                    z_path = os.path.join(result_dir, 'z2000.npy')
                else:
                    z_path = load_path_z
                z = np.load(z_path)
                z = torch.from_numpy(z).to(self.device)
                self.test_z = z
                self.test_type = 'z'
                z_crop, _, _ = random_crop(z, self.args.z_crop_size, 
                                         pos=self.args.z_pos, crop_type=self.args.z_crop_type)
                self.cloth = self.gan.generate(z_crop)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def preprocess_image(self, image):
        """
        对单张图片进行预处理：填充和缩放
        """
        padded_image = self._pad_and_scale(image, self.args.img_size)
        return padded_image  
    
    def detect(self, image):
        """
        检测图像
        """
        # 预处理原始图片
        if isinstance(image, Image.Image):
            preprocessed_image = self.preprocess_image(image)
            input_tensor = self.transform(preprocessed_image).unsqueeze(0).to(self.device)
        else:
            input_tensor = image.to(self.device)
          
        # 检测原始图片
        with torch.no_grad():
            output = self.model(input_tensor)
            boxes7 = self._postprocess_detection_output(output, conf_thresh=self.conf_thresh, nms_thresh=self.nms_thresh)

        boxes7 = boxes7.detach().cpu().numpy()
        bboxes = boxes7[:, [0, 1, 2, 3]]
        labels = boxes7[:, [6]]
        scores = boxes7[:, [4]]

        bboxes[:, [0]] = bboxes[:, [0]] * image.size[0]  # center_x * width
        bboxes[:, [1]] = bboxes[:, [1]] * image.size[1]  # center_y * height  
        bboxes[:, [2]] = bboxes[:, [2]] * image.size[0]  # width * width
        bboxes[:, [3]] = bboxes[:, [3]] * image.size[1]  # height * height

        # 从中心格式转换为角点格式
        center_x = bboxes[:, 0]
        center_y = bboxes[:, 1]
        width = bboxes[:, 2]
        height = bboxes[:, 3]

        bboxes[:, 0] = center_x - width / 2   # x1
        bboxes[:, 1] = center_y - height / 2 # y1
        bboxes[:, 2] = center_x + width / 2  # x2
        bboxes[:, 3] = center_y + height / 2 # y2

        return dict(bboxes=bboxes, labels=labels, scores=scores)

    def generate_adversarial_example(self, image):
        """
        生成对抗样本
        Args:
            PIL Image: RGB [H,W,C]
            
        Returns:
            PIL Image: RGB [H,W,C] 对抗样本
        """
        # 预处理原始图片
        if isinstance(image, Image.Image):
            preprocessed_image = self.preprocess_image(image)
            input_tensor = self.transform(preprocessed_image).unsqueeze(0).to(self.device)
        else:
            input_tensor = image.to(self.device)

        # 检测原始图片
        with torch.no_grad():
            output = self.model(input_tensor)
            # [xs/w, ys/h, ws/w, hs/h, det_confs, cls_max_confs, cls_max_ids]
            boxes7 = self._postprocess_detection_output(output, conf_thresh=0.5, nms_thresh=0.4)
        
        if self.adv_patch is None:
            self.adv_patch = self.generate_adv_patch()

        # 使用patch transformer处理patch
        target = boxes7[:, [6, 0, 1, 2, 3]]
        target = target.unsqueeze(0).to(self.device)
        adv_batch_t = self.patch_transformer(self.adv_patch, target, self.args.img_size, 
                                           do_rotate=True, rand_loc=False,
                                           pooling=self.args.pooling, 
                                           old_fasion=self.kwargs.get('old_fasion', True))
        
        # 应用patch
        adv_image_input_tensor = self.patch_applier(input_tensor, adv_batch_t)

        # 转回图片
        adversarial_image = unloader(adv_image_input_tensor[0].detach().cpu())
        return adversarial_image
    
    def _pad_and_scale(self, image, img_size):
        """@yolo2.load_data.py InriaDataset pad_and_scale
        注意和@utils.py中的pad_and_scale用了不同的resize函数
        
        Args:
            image: PIL Image
            img_size: int

        Returns:
            PIL Image
        """
        # 使用与InriaDataset相同的pad_and_scale
        w, h = image.size
        if w == h:
            padded_image = image
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_image = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_image.paste(image, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_image = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_image.paste(image, (0, int(padding)))
        
        # 使用与InriaDataset相同的resize方式
        resize = transforms.Resize((img_size, img_size))
        padded_image = resize(padded_image)
        return padded_image
    
    def _postprocess_detection_output(self, output, conf_thresh=0.5, nms_thresh=0.4):

        all_boxes = yolo2_utils.get_region_boxes_general(output, self.model, conf_thresh, self.kwargs['name'])
        boxes7 = all_boxes[0]
        boxes7 = yolo2_utils.nms(boxes7, self.nms_thresh) # [xs/w, ys/h, ws/w, hs/h, det_confs, cls_max_confs, cls_max_ids]
        return boxes7
    
    def generate_adv_patch(self):
        """
        为单张图片生成对抗patch
        """
        if self.test_type == 'gan':
            z = torch.randn(1, 128, *self.args.z_size, device=self.device)
            cloth = self.test_gan.generate(z)
            adv_patch, x, y = random_crop(cloth, self.args.crop_size, 
                                        pos=self.args.pos, crop_type=self.args.crop_type)
        elif self.test_type == 'z':
            z_crop, _, _ = random_crop(self.test_z, self.args.z_crop_size, 
                                    pos=self.args.z_pos, crop_type=self.args.z_crop_type)
            cloth = self.test_gan.generate(z_crop)
            adv_patch, x, y = random_crop(cloth, self.args.crop_size, 
                                        pos=self.args.pos, crop_type=self.args.crop_type)
        elif self.test_type == 'patch':
            adv_patch, x, y = random_crop(self.test_cloth, self.args.crop_size, 
                                        pos=self.args.pos, crop_type=self.args.crop_type)
        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")
        
        return adv_patch

    def _test_model(self, loader, adv_cloth=None, gan=None, z=None, type=None, 
                   conf_thresh=0.5, nms_thresh=0.4, iou_thresh=0.5, num_of_samples=100,
                   old_fasion=True):
        """测试模型性能"""
        self.model.eval()
        total = 0.0
        proposals = 0.0
        correct = 0.0
        batch_num = len(loader)

        with torch.no_grad():
            positives = []
            for batch_idx, (data, target) in tqdm(enumerate(loader), total=batch_num, position=0):
                data = data.to(self.device)
                target = target.to(self.device)

                adv_patch = self.generate_adv_patch()
                adv_batch_t = self.patch_transformer(adv_patch, target, self.args.img_size, 
                                                    do_rotate=True, rand_loc=False,
                                                    pooling=self.args.pooling, 
                                                    old_fasion=old_fasion)
                data = self.patch_applier(data, adv_batch_t)
                
                output = self.model(data)
                all_boxes = yolo2_utils.get_region_boxes_general(output, self.model, 
                                                        conf_thresh, self.kwargs['name'] if self.kwargs else 'yolov2')
                
                for i in range(len(all_boxes)):
                    boxes = all_boxes[i]
                    boxes = yolo2_utils.nms(boxes, nms_thresh)
                    truths = target[i].view(-1, 5)
                    truths = label_filter(truths, labels=[0])
                    num_gts = truths_length(truths)
                    truths = truths[:num_gts, 1:]
                    truths = truths.tolist()
                    total = total + num_gts
                    
                    for j in range(len(boxes)):
                        if boxes[j][6].item() == 0:
                            best_iou = 0
                            best_index = 0

                            for ib, box_gt in enumerate(truths):
                                iou = yolo2_utils.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_index = ib
                            if best_iou > iou_thresh:
                                del truths[best_index]
                                positives.append((boxes[j][4].item(), True))
                            else:
                                positives.append((boxes[j][4].item(), False))
                                
            positives = sorted(positives, key=lambda d: d[0], reverse=True)

            tps = []
            fps = []
            confs = []
            tp_counter = 0
            fp_counter = 0
            for pos in positives:
                if pos[1]:
                    tp_counter += 1
                else:
                    fp_counter += 1
                tps.append(tp_counter)
                fps.append(fp_counter)
                confs.append(pos[0])

            precision = []
            recall = []
            for tp, fp in zip(tps, fps):
                recall.append(tp / total)
                precision.append(tp / (fp + tp))

        if len(precision) > 1 and len(recall) > 1:
            p = np.array(precision)
            r = np.array(recall)
            p_start = p[np.argmin(r)]
            samples = np.arange(0., 1., 1.0 / num_of_samples)
            interpolated = interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
            avg = sum(interpolated) / len(interpolated)
        elif len(precision) > 0 and len(recall) > 0:
            avg = precision[0] * recall[0]
        else:
            avg = float('nan')

        return precision, recall, avg, confs

    def prepare_data(self, img_ori_dir):

        conf_thresh = 0.5
        nms_thresh = 0.4
        img_dir = './data/test_padded'
        lab_dir = './data/test_lab_%s' % self.kwargs['name']
        data_nl = load_data.InriaDataset(img_ori_dir, None, self.kwargs['max_lab'], self.args.img_size, shuffle=False)
        loader_nl = torch.utils.data.DataLoader(data_nl, batch_size=self.args.batch_size, shuffle=False, num_workers=10)
        if lab_dir is not None:
            if not os.path.exists(lab_dir):
                os.makedirs(lab_dir)
        if img_dir is not None:
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
        print('preparing the test data')
        with torch.no_grad():
            for batch_idx, (data, labs) in tqdm(enumerate(loader_nl), total=len(loader_nl)):
                data = data.to(self.device)
                output = self.model(data)
                all_boxes = yolo2_utils.get_region_boxes_general(output, self.model, conf_thresh, self.kwargs['name'])
                for i in range(data.size(0)):
                    boxes = all_boxes[i]
                    boxes = yolo2_utils.nms(boxes, nms_thresh)
                    new_boxes = boxes[:, [6, 0, 1, 2, 3]]
                    new_boxes = new_boxes[new_boxes[:, 0] == 0]
                    new_boxes = new_boxes.detach().cpu().numpy()
                    if lab_dir is not None:
                        save_dir = os.path.join(lab_dir, labs[i])
                        np.savetxt(save_dir, new_boxes, fmt='%f')
                        img = unloader(data[i].detach().cpu())
                    if img_dir is not None:
                        save_dir = os.path.join(img_dir, labs[i].replace('.txt', '.png'))
                        img.save(save_dir)
        print('preparing done')

    def run_evaluation(self, img_ori_dir, save_dir='./test_results', prepare_data=False):
        """运行完整的评估流程"""
        if not hasattr(self, 'test_cloth'):
            raise ValueError("Please load pretrained attack model first using load_pretrained_attack()")
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, f'yolov2_{self.method}')

        if prepare_data:
            self.prepare_data(img_ori_dir)
        
        # 创建测试数据加载器
        img_dir_test = './data/test_padded'
        lab_dir_test = f'./data/test_lab_{self.kwargs["name"] if self.kwargs else "yolov2"}'
        test_data = load_data.InriaDataset(img_dir_test,
                                           lab_dir_test, 
                                         self.kwargs['max_lab'], 
                                         self.args.img_size, 
                                         shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, self.args.batch_size, shuffle=False, num_workers=10)
        
        # 运行测试
        plt.figure(figsize=[15, 10])
        prec, rec, ap, confs = self._test_model(test_loader, adv_cloth=self.test_cloth, 
                                               gan=self.test_gan, z=self.test_z, 
                                               type=self.test_type, conf_thresh=0.01, 
                                               old_fasion=self.kwargs['old_fasion'] if self.kwargs else True)
        
        # 保存结果
        np.savez(save_path, prec=prec, rec=rec, ap=ap, confs=confs, 
                adv_patch=self.cloth.detach().cpu().numpy())
        print(f'AP is {ap:.4f}')
        
        plt.plot(rec, prec)
        leg = [f'{self.method}: ap {ap:.3f}']
        unloader(self.cloth[0]).save(save_path + '.png')
        
        return prec, rec, ap, confs


# 辅助函数
def label_filter(truths, labels=None):
    """过滤标签"""
    if labels is not None:
        new_truths = truths.new(truths.shape).fill_(-1)
        c = 0
        for t in truths:
            if t[0].item() in labels:
                new_truths[c] = t
                c = c + 1
        return new_truths
    return truths


def truths_length(truths):
    """计算真实标签的长度"""
    for i in range(50):
        if truths[i][1] == -1:
            return i
    return len(truths)
