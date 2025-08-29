import os
import torch
import itertools
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from yolo2 import load_data
from yolo2 import utils
from utils import *
from cfg import get_cfgs
from tps_grid_gen import TPSGridGen
from load_models import load_models
from generator_dim import GAN_dis

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
                 model_name='yolov2',
                 method='TCEGA',
                 class_mapping=None, 
                 device=None,
                 args=None,
                 kwargs=None):
        
        self.model_name = model_name
        self.method = method
        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        
        # 加载模型
        self._load_model(model_name)
            
        # 加载类别名称
        if class_mapping is None:
            self.class_names = utils.load_class_names('./data/coco.names')
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
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
        ])
    
    def _load_model(self, model_name):
        """加载模型"""
        if model_name == "yolov2":
            args, kwargs = get_cfgs('yolov2', self.method, 'test')
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
            self.adv_cloth = torch.from_numpy(np.load(img_path)[:1]).to(self.device)
            self.test_cloth = self.adv_cloth.detach().clone()
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
        
    def load_dataset(self, image_folder, label_file):
        """
        加载自定义数据集

        参数:
            image_folder: 图像文件夹路径
            label_file: 标签文件路径

        返回:
            InriaDataset实例
        """
        if self.kwargs:
            return load_data.InriaDataset(image_folder, label_file, 
                                        self.kwargs['max_lab'], 
                                        self.args.img_size if self.args else 416, 
                                        shuffle=False)
        else:
            # 如果没有kwargs，使用默认参数
            return load_data.InriaDataset(image_folder, label_file, 15, 416, shuffle=False)

    def detect(self, image):
        """
        检测图像
        """
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
            
        output = self.model(image_tensor)
        return output[0]
    
    def generate_adversarial_example(self, image):
        """
        生成对抗样本
        Args:
            PIL Image: RGB [H,W,C]
            
        Returns:
            PIL Image: RGB [H,W,C] 对抗样本
        """
        # 这里实现具体的对抗样本生成逻辑
        # 根据不同的method实现不同的攻击策略
        if self.method == 'TCEGA':
            return self._generate_tcega_example(image)
        elif self.method == 'EGA':
            return self._generate_ega_example(image)
        elif self.method in ['RCA', 'TCA']:
            return self._generate_patch_example(image)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _generate_tcega_example(self, image):
        """生成TCEGA对抗样本"""
        # 实现TCEGA的具体逻辑
        # 这里需要根据原始代码实现
        pass
        
    def _generate_ega_example(self, image):
        """生成EGA对抗样本"""
        # 实现EGA的具体逻辑
        pass
        
    def _generate_patch_example(self, image):
        """生成patch-based对抗样本"""
        # 实现patch-based攻击的具体逻辑
        pass

    def tcega_attack(self, images, labels=[]):
        """
        TCEGA对抗攻击实现

        参数:
            images: 输入图像张量 (带batch维度)
            labels: 目标标签
            epsilon: 总扰动上限 (默认0.04)
            alpha: 单次迭代扰动大小 (默认0.01)
            num_iter: 迭代次数 (默认10)

        返回:
            对抗样本张量
        """
        # 实现TCEGA攻击逻辑
        # 这里需要根据原始代码实现具体的攻击算法
        pass
      
    def evaluate_attack(self, data_loader, original_results_folder, adversarial_results_folder,
                        epsilon=0.04, alpha=0.01, num_iter=10, debug=False):
        """
        评估TCEGA对抗攻击效果

        参数:
            data_loader: 数据加载器
            original_results_folder: 原始检测结果保存文件夹
            adversarial_results_folder: 对抗样本检测结果保存文件夹
            epsilon: 总扰动上限
            alpha: 单次迭代扰动大小
            num_iter: 迭代次数

        返回:
            评估结果字典
        """
        # 创建保存目录
        if not os.path.exists(original_results_folder):
            os.makedirs(original_results_folder)
        if not os.path.exists(adversarial_results_folder):
            os.makedirs(adversarial_results_folder)
            
        # 评估原始模型性能
        print("Evaluating original model performance...")
        orig_prec, orig_rec, orig_ap, orig_confs = self._test_model(data_loader, None, None, None, None)
        
        # 生成对抗样本并评估
        print("Generating adversarial examples...")
        adv_prec, adv_rec, adv_ap, adv_confs = self._test_model(data_loader, None, None, None, None, 
                                                               is_adversarial=True)
        
        # 保存结果
        self._save_results(original_results_folder, orig_prec, orig_rec, orig_ap, orig_confs, "original")
        self._save_results(adversarial_results_folder, adv_prec, adv_rec, adv_ap, adv_confs, "adversarial")
        
        # 计算攻击效果指标
        results = {
            'original': {
                'precision': orig_prec,
                'recall': orig_rec,
                'AP': orig_ap,
                'confidence_scores': orig_confs
            },
            'adversarial': {
                'precision': adv_prec,
                'recall': adv_rec,
                'AP': adv_ap,
                'confidence_scores': adv_confs
            },
            'attack_effectiveness': {
                'AP_drop': orig_ap - adv_ap,
                'AP_drop_ratio': (orig_ap - adv_ap) / orig_ap if orig_ap > 0 else 0
            }
        }
        
        return results
    
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
                
                if type == 'gan':
                    z = torch.randn(1, 128, *self.args.z_size, device=self.device)
                    cloth = gan.generate(z)
                    adv_patch, x, y = random_crop(cloth, self.args.crop_size, 
                                                pos=self.args.pos, crop_type=self.args.crop_type)
                elif type == 'z':
                    z_crop, _, _ = random_crop(z, self.args.z_crop_size, 
                                                pos=self.args.z_pos, crop_type=self.args.z_crop_type)
                    cloth = gan.generate(z_crop)
                    adv_patch, x, y = random_crop(cloth, self.args.crop_size, 
                                                pos=self.args.pos, crop_type=self.args.crop_type)
                elif type == 'patch':
                    adv_patch, x, y = random_crop(adv_cloth, self.args.crop_size, 
                                                pos=self.args.pos, crop_type=self.args.crop_type)
                elif type is not None:
                    raise ValueError
                    
                if adv_patch is not None:
                    target = target.to(self.device)
                    adv_batch_t = self.patch_transformer(adv_patch, target, self.args.img_size, 
                                                        do_rotate=True, rand_loc=False,
                                                        pooling=self.args.pooling, 
                                                        old_fasion=old_fasion)
                    data = self.patch_applier(data, adv_batch_t)
                
                output = self.model(data)
                all_boxes = utils.get_region_boxes_general(output, self.model, 
                                                        conf_thresh, self.kwargs['name'] if self.kwargs else 'yolov2')
                
                for i in range(len(all_boxes)):
                    boxes = all_boxes[i]
                    boxes = utils.nms(boxes, nms_thresh)
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
                                iou = utils.bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
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
    
    def _save_results(self, save_dir, precision, recall, ap, confs, prefix):
        """保存测试结果"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, f"{prefix}_results")
        np.savez(save_path, prec=precision, rec=recall, ap=ap, confs=confs)
        
        # 绘制PR曲线
        plt.figure(figsize=[15, 10])
        plt.plot(recall, precision)
        plt.title(f'{prefix.capitalize()} PR-curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.ylim([0, 1.05])
        plt.xlim([0, 1.05])
        plt.savefig(os.path.join(save_dir, f'{prefix}_PR-curve.png'), dpi=300)
        plt.close()
            
    def run_evaluation(self, save_dir='./test_results'):
        """运行完整的评估流程"""
        if not hasattr(self, 'test_cloth'):
            raise ValueError("Please load pretrained attack model first using load_pretrained_attack()")
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, f'yolov2_{self.method}')
        
        # 创建测试数据加载器
        img_dir_test = './data/test_padded'
        lab_dir_test = f'./data/test_lab_{self.kwargs["name"] if self.kwargs else "yolov2"}'
        test_data = load_data.InriaDataset(img_dir_test, lab_dir_test, 
                                         self.kwargs['max_lab'] if self.kwargs else 15, 
                                         self.args.img_size if self.args else 416, 
                                         shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False, num_workers=10)
        
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
