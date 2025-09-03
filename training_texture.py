import os
import torch
import torch.optim as optim
import itertools
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import time
import argparse
import sys
import numpy as np

# 添加PyTorch 2.6兼容性支持
try:
    from easydict import EasyDict
    torch.serialization.add_safe_globals([EasyDict])
except ImportError:
    pass

from adversarial_attacks.detectors.yolo2 import load_data
from adversarial_attacks.detectors.yolo2 import utils
from adversarial_attacks.physical.tcega.utils import get_det_loss, random_crop
from adversarial_attacks.physical.tcega.cfg import get_cfgs
from adversarial_attacks.physical.tcega.tps_grid_gen import TPSGridGen
from adversarial_attacks.physical.tcega.generator_dim import GAN_dis
from adversarial_attacks.detectors.load_models import load_models

# 全局变量用于保存训练状态
training_state = {
    'current_epoch': 0,
    'current_batch': 0,
    'best_loss': float('inf'),
    'start_time': None
}


def save_checkpoint(checkpoint_dir, state, filename):
    """保存检查点"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(checkpoint_dir,filename):
    """加载检查点"""
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading checkpoint: {filepath}")
        return torch.load(filepath, map_location='cpu', weights_only=False)
    return None

def load_latest_checkpoint(checkpoint_dir, prefix):
    """加载最新的检查点"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找匹配的检查点文件
    import glob
    pattern = f'{prefix}_epoch*.pth'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        return None
    
    # 按文件名中的epoch数排序，找到最新的
    def extract_epoch(filename):
        import re
        match = re.search(r'epoch(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    
    return torch.load(latest_checkpoint, map_location='cpu', weights_only=False)


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--net', default='yolov2', help='target net name')
parser.add_argument('--method', default='TCEGA', help='method name')
parser.add_argument('--suffix', default=None, help='suffix name')
parser.add_argument('--gen_suffix', default=None, help='generator suffix name')
parser.add_argument('--epoch', type=int, default=None, help='')
parser.add_argument('--z_epoch', type=int, default=None, help='')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--save_freq', type=int, default=50, help='')
pargs = parser.parse_args()


args, kwargs = get_cfgs(pargs.net, pargs.method, mode='training')
if pargs.epoch is not None:
    args.n_epochs = pargs.epoch
if pargs.z_epoch is not None:
    args.z_epochs = pargs.z_epoch
if pargs.suffix is None:
    pargs.suffix = pargs.net + '_' + pargs.method

device = torch.device(pargs.device)

darknet_model = load_models(**kwargs)
darknet_model = darknet_model.eval().to(device)

class_names = utils.load_class_names('./data/coco.names')
img_dir_train = './data/INRIAPerson/Train/pos'
lab_dir_train = './data/train_labels'
target_label = 0
train_data = load_data.InriaDataset(img_dir_train, lab_dir_train, kwargs['max_lab'], args.img_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=kwargs['batch_size'], shuffle=True, num_workers=10)
target_func = lambda obj, cls: obj
patch_applier = load_data.PatchApplier().to(device)
patch_transformer = load_data.PatchTransformer().to(device)
if kwargs['name'] == 'ensemble':
    prob_extractor_yl2 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov2').to(device)
    prob_extractor_yl3 = load_data.MaxProbExtractor(0, 80, target_func, 'yolov3').to(device)
else:
    prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)
total_variation = load_data.TotalVariation().to(device)

target_control_points = torch.tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))

tps = TPSGridGen(torch.Size([300, 300]), target_control_points)
tps.to(device)

target_func = lambda obj, cls: obj
prob_extractor = load_data.MaxProbExtractor(0, 80, target_func, kwargs['name']).to(device)

result_root_dir = './training_results'
# 结果文件路径
results_dir = os.path.join(result_root_dir, pargs.suffix + '_result')

print(results_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 检查点文件路径
checkpoint_dir = os.path.join(result_root_dir, pargs.suffix + '_checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 日志文件路径
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
writer_logdir = os.path.join(result_root_dir, pargs.suffix + '_runs', TIMESTAMP + '_' + pargs.suffix)

loader = train_loader
epoch_length = len(loader)
print(f'One epoch is {len(loader)}')

def train_patch():
    def generate_patch(type):
        cloth_size_true = np.ceil(np.array(args.cloth_size) / np.array(args.pixel_size)).astype(np.int64)
        if type == 'gray':
            adv_patch = torch.full((1, 3, cloth_size_true[0], cloth_size_true[1]), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((1, 3, cloth_size_true[0], cloth_size_true[1]))
        else:
            raise ValueError
        return adv_patch

    writer = SummaryWriter(logdir=writer_logdir)

    # 尝试加载最新的检查点
    checkpoint = load_latest_checkpoint(checkpoint_dir, f'{pargs.suffix}')
    
    if checkpoint is not None:
        print(f"Resuming training from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1
        adv_patch = checkpoint['adv_patch'].to(device)
        optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, best_loss: {best_loss:.6f}")
    else:
        print("Starting new training...")
        start_epoch = 1
        adv_patch = generate_patch("gray").to(device)
        optimizer = optim.Adam([adv_patch], lr=args.learning_rate, amsgrad=True)
        best_loss = float('inf')

    adv_patch.requires_grad_(True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=args.learning_rate / 100)

    et0 = time.time()
    training_state['start_time'] = et0
    
    for epoch in range(start_epoch, args.n_epochs + 1):
        training_state['current_epoch'] = epoch
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        
        print(f"\nEpoch {epoch}/{args.n_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        for i_batch, (img_batch, lab_batch, img_path_batch, lbl_path_batch) in tqdm(enumerate(loader), desc=f'Running epoch {epoch}',
                                                    total=epoch_length):
            training_state['current_batch'] = i_batch
            
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)
            adv_patch_crop, x, y = random_crop(adv_patch, args.crop_size, pos=args.pos, crop_type=args.crop_type)
            adv_patch_tps, _ = tps.tps_trans(adv_patch_crop, max_range=0.1, canvas=0.5) # random tps transform
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, target_label, args.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch_crop)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().cpu().numpy()
            ep_tv_loss += tv_loss.detach().cpu().numpy()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            adv_patch.data.clamp_(0, 1)  # keep patch in image range

            bt1 = time.time()
            if i_batch % pargs.save_freq == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)


            # if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
            if i_batch % pargs.save_freq == 0:
                writer.add_image('patch', adv_patch.squeeze(0), iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                
                # 保存最佳模型
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_patch_path = os.path.join(results_dir, 'best_patch.pth')
                    torch.save({
                        'epoch': epoch,
                        'adv_patch': adv_patch.detach().cpu(),
                        'loss': best_loss
                    }, best_patch_path)
                    print(f"New best loss: {best_loss:.6f}, saved to {best_patch_path}")
            
            bt0 = time.time()
            
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        
        print(f"Epoch {epoch} completed:")
        print(f"  Detection Loss: {ep_det_loss:.6f}")
        print(f"  TV Loss: {ep_tv_loss:.6f}")
        print(f"  Total Loss: {ep_loss:.6f}")
        print(f"  Time: {et1 - et0:.2f}s")
        
        if epoch > 300:
            scheduler.step(ep_loss)
        et0 = time.time()
        writer.flush()
        
        # 每pargs.save_freq个epoch保存一次检查点
        if epoch % pargs.save_freq == 0:
            checkpoint_state = {
                'epoch': epoch,
                'adv_patch': adv_patch.detach().cpu(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'args': args,
                'kwargs': kwargs
            }
            save_checkpoint(checkpoint_dir, checkpoint_state, f'{pargs.suffix}_epoch{epoch}.pth')
            print(f"Checkpoint saved at epoch {epoch}")
    
    writer.close()
    print("Training completed.")
    return 0


def train_EGA():
    gen = GAN_dis(DIM=args.DIM, z_dim=args.z_dim, img_shape=args.patch_size)
    
    # 尝试加载最新的检查点
    checkpoint = load_latest_checkpoint(checkpoint_dir, f'{pargs.suffix}')
    
    if checkpoint is not None:
        print(f"Resuming EGA training from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1
        gen.load_state_dict(checkpoint['gen_state'])
        best_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, best_loss: {best_loss:.6f}")
    else:
        print("Starting new EGA training...")
        start_epoch = 1
        best_loss = float('inf')
    
    gen.to(device)
    gen.train()

    writer = SummaryWriter(logdir=writer_logdir)

    optimizerG = optim.Adam(gen.G.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerD = optim.Adam(gen.D.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    
    # 如果加载了检查点，恢复优化器状态
    if checkpoint is not None:
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])

    et0 = time.time()
    training_state['start_time'] = et0
    
    for epoch in range(start_epoch, args.n_epochs + 1):
        training_state['current_epoch'] = epoch
        ep_det_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        D_loss = 0
        bt0 = time.time()
        
        print(f"\nEGA Epoch {epoch}/{args.n_epochs}")
        print(f"Generator LR: {optimizerG.param_groups[0]['lr']:.6f}")
        print(f"Discriminator LR: {optimizerD.param_groups[0]['lr']:.6f}")
        
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running EGA epoch {epoch}',
                                                    total=epoch_length):
            training_state['current_batch'] = i_batch
            
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)

            z = torch.randn(img_batch.shape[0], args.z_dim, args.z_size, args.z_size, device=device)

            adv_patch = gen.generate(z)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)

            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            disc, pj, pm = gen.get_loss(adv_patch, z[:adv_patch.shape[0]], args.gp)
            tv_loss = tv * args.tv_loss
            disc_loss = disc * args.disc if epoch >= args.dim_start_epoch else disc * 0.0

            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device)) + disc_loss
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()

            loss.backward()
            optimizerG.step()
            optimizerD.step()
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            bt1 = time.time()
            if i_batch % pargs.save_freq == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.item(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.item(), iteration)
                writer.add_scalar('loss/tv_loss', tv.item(), iteration)
                writer.add_scalar('loss/disc_loss', disc.item(), iteration)
                writer.add_scalar('loss/disc_prob_true', pj.mean().item(), iteration)
                writer.add_scalar('loss/disc_prob_fake', pm.mean().item(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizerG.param_groups[0]["lr"], iteration)

            # if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
            if epoch % pargs.save_freq == 0:
                writer.add_image('patch', adv_patch[0], iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                torch.save(gen.state_dict(), os.path.join(results_dir, f"{pargs.suffix}_epoch{epoch}.pkl"))
                
                # 保存最佳模型
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_gen_path = os.path.join(results_dir, 'best_gen.pth')
                    torch.save({
                        'epoch': epoch,
                        'gen_state': gen.state_dict(),
                        'loss': best_loss
                    }, best_gen_path)
                    print(f"New best loss: {best_loss:.6f}, saved to {best_gen_path}")

            bt0 = time.time()
            
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        D_loss = D_loss / len(loader)
        
        print(f"EGA Epoch {epoch} completed:")
        print(f"  Detection Loss: {ep_det_loss:.6f}")
        print(f"  TV Loss: {ep_tv_loss:.6f}")
        print(f"  Total Loss: {ep_loss:.6f}")
        print(f"  Time: {et1 - et0:.2f}s")
        
        et0 = time.time()
        writer.flush()
        
        # 每pargs.save_freq个epoch保存一次检查点
        if epoch % pargs.save_freq == 0:
            checkpoint_state = {
                'epoch': epoch,
                'gen_state': gen.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'best_loss': best_loss,
                'args': args,
                'kwargs': kwargs
            }
            save_checkpoint(checkpoint_dir, checkpoint_state, f'{pargs.suffix}_epoch{epoch}.pth')
            print(f"EGA checkpoint saved at epoch {epoch}")
    
    writer.close()
    print("EGA training completed.")
    return gen


def train_z(gen=None):
    if gen is None:
        gen = GAN_dis(DIM=128, z_dim=128, img_shape=(324,) * 2)
        suffix_load = pargs.gen_suffix # 提供gen训练的模型地址，用于恢复gen
        print(f"debug: suffix_load is {suffix_load}")
        result_dir = os.path.join(result_root_dir, 'result_' + suffix_load)
        d = torch.load(os.path.join(result_dir, suffix_load + '.pkl'), map_location='cpu', weights_only=False)
        gen.load_state_dict(d)
    
    # 尝试加载z训练的最新检查点
    checkpoint = load_latest_checkpoint(checkpoint_dir, 'z')
    
    if checkpoint is not None:
        print(f"Resuming z training from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1
        z = checkpoint['z'].to(device)
        best_loss = checkpoint['best_loss']
        print(f"Loaded z checkpoint: epoch {checkpoint['epoch']}, best_loss: {best_loss:.6f}")
    else:
        print("Starting new z training...")
        start_epoch = 1
        # Generate starting point
        z0 = torch.randn(*args.z_shape, device=device)
        z = z0.detach().clone()
        best_loss = float('inf')
    
    gen.to(device)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False

    writer = SummaryWriter(logdir=writer_logdir + '_z')

    z.requires_grad_(True)

    optimizer = optim.Adam([z], lr=args.learning_rate_z, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=500,
                                                     min_lr=args.learning_rate_z / 100)
    
    # 如果加载了检查点，恢复优化器状态
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    et0 = time.time()
    training_state['start_time'] = et0
    
    for epoch in range(start_epoch, args.z_epochs + 1):
        training_state['current_epoch'] = epoch
        ep_det_loss = 0
        #     ep_nps_loss = 0
        ep_tv_loss = 0
        ep_loss = 0
        bt0 = time.time()
        
        print(f"\nZ Training Epoch {epoch}/{args.z_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        for i_batch, (img_batch, lab_batch) in tqdm(enumerate(loader), desc=f'Running z epoch {epoch}',
                                                    total=epoch_length):
            training_state['current_batch'] = i_batch
            
            img_batch = img_batch.to(device)
            lab_batch = lab_batch.to(device)
            z_crop, _, _ = random_crop(z, args.crop_size_z, pos=args.pos, crop_type=args.crop_type_z)

            adv_patch = gen.generate(z_crop)
            adv_patch_tps, _ = tps.tps_trans(adv_patch, max_range=0.1, canvas=0.5, target_shape=adv_patch.shape[-2:])
            adv_batch_t = patch_transformer(adv_patch_tps, lab_batch, args.img_size, do_rotate=True, rand_loc=False,
                                            pooling=args.pooling, old_fasion=kwargs['old_fasion'])
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            det_loss, valid_num = get_det_loss(darknet_model, p_img_batch, lab_batch, args, kwargs)
            if valid_num > 0:
                det_loss = det_loss / valid_num

            tv = total_variation(adv_patch)
            tv_loss = tv * args.tv_loss
            loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).to(device))
            ep_det_loss += det_loss.detach().item()
            ep_tv_loss += tv_loss.detach().item()
            ep_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            bt1 = time.time()
            if i_batch % pargs.save_freq == 0:
                iteration = epoch_length * epoch + i_batch

                writer.add_scalar('loss/total_loss', loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                writer.add_scalar('loss/tv_loss', tv.detach().cpu().numpy(), iteration)
                writer.add_scalar('misc/epoch', epoch, iteration)
                writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

            # if epoch % max(min((args.n_epochs // 10), 100), 1) == 0:
            if epoch % pargs.save_freq == 0:
                writer.add_image('patch', adv_patch.squeeze(0), iteration)
                rpath = os.path.join(results_dir, 'patch%d' % epoch)
                np.save(rpath, adv_patch.detach().cpu().numpy())
                rpath = os.path.join(results_dir, 'z%d' % epoch)
                np.save(rpath, z.detach().cpu().numpy())
                
                # 保存最佳模型
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_z_path = os.path.join(results_dir, 'best_z.pth')
                    torch.save({
                        'epoch': epoch,
                        'z': z.detach().cpu(),
                        'loss': best_loss
                    }, best_z_path)
                    print(f"New best z loss: {best_loss:.6f}, saved to {best_z_path}")
            
            bt0 = time.time()
            
        et1 = time.time()
        ep_det_loss = ep_det_loss / len(loader)
        ep_tv_loss = ep_tv_loss / len(loader)
        ep_loss = ep_loss / len(loader)
        
        print(f"Z Training Epoch {epoch} completed:")
        print(f"  Detection Loss: {ep_det_loss:.6f}")
        print(f"  TV Loss: {ep_tv_loss:.6f}")
        print(f"  Total Loss: {ep_loss:.6f}")
        print(f"  Time: {et1 - et0:.2f}s")
        
        if epoch > 300:
            scheduler.step(ep_loss)
        et0 = time.time()
        writer.flush()
        
        # 每pargs.save_freq个epoch保存一次检查点
        if epoch % pargs.save_freq == 0:
            checkpoint_state = {
                'epoch': epoch,
                'z': z.detach().cpu(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'args': args,
                'kwargs': kwargs
            }
            save_checkpoint(checkpoint_dir, checkpoint_state, f'z_epoch{epoch}.pth')
            print(f"Z checkpoint saved at epoch {epoch}")
    
    writer.close()
    print("Z training completed.")
    return 0


if __name__ == "__main__":
    print("=" * 60)
    print(f"Starting training with method: {pargs.method}")
    print(f"Target network: {pargs.net}")
    print(f"Device: {pargs.device}")
    print(f"Results directory: {results_dir}")
    print("=" * 60)
    
    if pargs.method == 'RCA':
        print("Training with RCA method...")
        train_patch()
    elif pargs.method == 'TCA':
        print("Training with TCA method...")
        train_patch()
    elif pargs.method == 'EGA':
        print("Training with EGA method...")
        train_EGA()
    elif pargs.method == 'TCEGA':
        print("Training with TCEGA method...")
        print("Phase 1: Training EGA generator...")
        gen = train_EGA()
        print('Phase 2: Start optimizing z...')
        train_z(gen)
    else:
        print(f"Unknown method: {pargs.method}")
        print("Available methods: RCA, TCA, EGA, TCEGA")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

