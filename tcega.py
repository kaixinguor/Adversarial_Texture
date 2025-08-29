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

    def single_image_inference(self, image, conf_thresh=0.5, nms_thresh=0.4, iou_thresh=0.5):
        """
        单图推理模式：对单张图片进行预处理、贴patch、后处理等完整流程
        
        Args:
            image: PIL Image 或 numpy array，输入图片
            conf_thresh: 置信度阈值
            nms_thresh: NMS阈值
            iou_thresh: IoU阈值
            
        Returns:
            tuple: (adversarial_image, detection_results)
                - adversarial_image: PIL Image，贴了patch的图片
                - detection_results: dict，包含检测结果的字典
        """
        # 确保输入是PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image or numpy array")
        
        # 1. 预处理：填充和缩放
        processed_image, processed_label = self._preprocess_single_image(image)
        
        # 2. 生成对抗patch
        adv_patch = self._generate_adv_patch_for_single_image()
        
        # 3. 应用patch
        adversarial_image_tensor = self._apply_patch_to_single_image(processed_image, adv_patch, processed_label)
        
        # 4. 后处理：检测
        detection_results = self._postprocess_single_image(adversarial_image_tensor, conf_thresh, nms_thresh, iou_thresh)
        
        # 5. 转换回PIL Image
        adversarial_image = unloader(adversarial_image_tensor[0].detach().cpu())
        
        return adversarial_image, detection_results
    
    def _preprocess_single_image(self, image):
        """
        对单张图片进行预处理：填充和缩放
        """
        # 使用与训练时相同的预处理逻辑
        if hasattr(self, 'args') and hasattr(self.args, 'img_size'):
            img_size = self.args.img_size
        else:
            img_size = 416  # 默认值
            
        # 创建虚拟标签（用于patch变换）
        # 假设图片中心有一个目标，大小为图片的20%
        h, w = image.size[1], image.size[0]
        center_x, center_y = 0.5, 0.5
        target_w, target_h = 0.2, 0.4  # 相对尺寸
        
        # 创建标签格式：[class_id, center_x, center_y, width, height]
        # class_id=0 表示person类别
        label = torch.tensor([[0, center_x, center_y, target_w, target_h]], dtype=torch.float32)
        
        # 填充和缩放图片
        padded_image = self._pad_and_scale_image(image, img_size)
        
        return padded_image, label
    
    def _pad_and_scale_image(self, image, target_size):
        """
        对图片进行填充和缩放，使其达到目标尺寸
        """
        # 计算填充
        w, h = image.size
        target_w, target_h = target_size, target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图片
        resized_image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 创建目标尺寸的图片并居中放置
        target_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
        
        # 计算偏移量使图片居中
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        
        target_image.paste(resized_image, (offset_x, offset_y))
        
        return target_image
    
    def _generate_adv_patch_for_single_image(self):
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
    
    def _apply_patch_to_single_image(self, image, adv_patch, label):
        """
        将对抗patch应用到单张图片上
        """
        # 转换为tensor
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        # 扩展label维度以匹配batch处理
        label_batch = label.unsqueeze(0).to(self.device)
        
        # 使用patch transformer处理patch
        adv_batch_t = self.patch_transformer(adv_patch, label_batch, self.args.img_size, 
                                           do_rotate=True, rand_loc=False,
                                           pooling=self.args.pooling, 
                                           old_fasion=self.kwargs.get('old_fasion', True))
        
        # 应用patch
        adversarial_image = self.patch_applier(image_tensor, adv_batch_t)
        
        return adversarial_image
    
    def _postprocess_single_image(self, image_tensor, conf_thresh, nms_thresh, iou_thresh):
        """
        对单张图片进行后处理：目标检测
        """
        # 确保模型处于评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 前向传播
            output = self.model(image_tensor)
            
            # 获取检测框
            all_boxes = utils.get_region_boxes_general(output, self.model, 
                                                    conf_thresh, 
                                                    self.kwargs.get('name', 'yolov2'))
            
            # 处理检测结果
            detection_results = []
            for i in range(len(all_boxes)):
                boxes = all_boxes[i]
                # 应用NMS
                boxes = utils.nms(boxes, nms_thresh)
                
                # 解析检测结果
                for box in boxes:
                    if box[6].item() == 0:  # person类别
                        detection_result = {
                            'class_id': int(box[6].item()),
                            'class_name': 'person',
                            'confidence': float(box[4].item()),
                            'bbox': {
                                'x1': float(box[0].item()),
                                'y1': float(box[1].item()),
                                'x2': float(box[2].item()),
                                'y2': float(box[3].item())
                            }
                        }
                        detection_results.append(detection_result)
            
            # 按置信度排序
            detection_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'detections': detection_results,
                'num_detections': len(detection_results),
                'model_output': output[0].detach().cpu().numpy() if output[0] is not None else None
            }

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

    def prepare_data(self):

        conf_thresh = 0.5
        nms_thresh = 0.4
        img_ori_dir = './data/INRIAPerson/Test/pos'
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
                all_boxes = utils.get_region_boxes_general(output, self.model, conf_thresh, self.kwargs['name'])
                for i in range(data.size(0)):
                    boxes = all_boxes[i]
                    boxes = utils.nms(boxes, nms_thresh)
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

    def run_evaluation(self, prepare_data=False, save_dir='./test_results'):
        """运行完整的评估流程"""
        if not hasattr(self, 'test_cloth'):
            raise ValueError("Please load pretrained attack model first using load_pretrained_attack()")
            
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, f'yolov2_{self.method}')

        if prepare_data:
            self.prepare_data()
        
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
