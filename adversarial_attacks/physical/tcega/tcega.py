import os
import torch
import itertools
import numpy as np
from PIL import Image

import torch.nn.functional as F
from torchvision import transforms

# 临时使用yolov2，后续扩展检测器后再整理
from adversarial_attacks.detectors.yolo2 import load_data
from adversarial_attacks.detectors.yolo2 import utils as yolo2_utils
from adversarial_attacks.detectors.load_models import load_models
from .tps_grid_gen import TPSGridGen
from .generator_dim import GAN_dis
from .cfg import get_cfgs
from .utils import random_crop

unloader = transforms.ToPILImage()

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

        self.adv_patch = None
    
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

        # 使用与yolo2.utils.py plot_boxes_cv2一致的坐标转换方法
        img_width, img_height = preprocessed_image.size[0], preprocessed_image.size[1]

        for i in range(len(bboxes)):
            box = bboxes[i]
            x1 = int(round((box[0] - box[2] / 2.0) * img_width))
            y1 = int(round((box[1] - box[3] / 2.0) * img_height))
            x2 = int(round((box[0] + box[2] / 2.0) * img_width))
            y2 = int(round((box[1] + box[3] / 2.0) * img_height))
            
            bboxes[i, 0] = x1
            bboxes[i, 1] = y1
            bboxes[i, 2] = x2
            bboxes[i, 3] = y2

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
            boxes7 = self._postprocess_detection_output(output, conf_thresh=self.conf_thresh, nms_thresh=self.nms_thresh)
        
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
        """
        [xs/w, ys/h, ws/w, hs/h, det_confs, cls_max_confs, cls_max_ids]
         
         
         
         """
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