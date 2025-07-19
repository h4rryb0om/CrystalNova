# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:44:38 2025

@author: 19754
"""

import argparse
import os
import torch
import numpy as np
from pymatgen.core import Structure
from torch.utils.data import Dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 确保导入 F 模块
from pymatgen.core.periodic_table import Element
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from pymatgen.core.lattice import Lattice  # 导入 Lattice 类
from pymatgen.analysis import structure_matcher
import matplotlib.pyplot as plt
import traceback

torch.autograd.set_detect_anomaly(True)
# 修改点1：新增全局参数
MAX_ATOMS = 32  # 最大原子数
ELEMENT_TYPES = 83  # 根据元素周期表设置
LATENT_DIM = 64
LATTICE_DIM = 9  # 3x3矩阵展平后的维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def element_to_index(element):
    """将元素符号转换为原子序数"""
    from pymatgen.core.periodic_table import Element
    return Element(element).Z - 1  # 0-based索引


class CrystalFeaturizer:
    """修改点2：重构特征提取器"""

    def __init__(self, global_elements):
        self.global_elements = global_elements
        self.element_to_idx = {e: i for i, e in enumerate(global_elements)}

    def __call__(self, struct): \
            # 提取晶胞参数（3x3矩阵展平为9维）
        # 计算归一化因子（平均晶格长度）
        lattice_matrix = struct.lattice.matrix
        scale_factor = 10

        # 归一化晶胞参数和坐标
        lattice = (lattice_matrix.flatten() / scale_factor).astype(np.float32)
        coords = struct.frac_coords
        # 提取元素和坐标
        elements = [site.species.elements[0].symbol for site in struct]

        # 转换为索引并填充
        elem_indices = [self.element_to_idx[e] for e in elements]
        elem_indices = elem_indices[:MAX_ATOMS] + [0] * (MAX_ATOMS - len(elements))

        # 坐标填充
        coords = coords[:MAX_ATOMS]
        coords_padded = np.zeros((MAX_ATOMS, 3))
        coords_padded[:len(coords)] = coords
        # print(torch.tensor(coords_padded, dtype=torch.float),)
        # 成分向量
        composition = np.zeros(len(self.global_elements))
        for e in elements:
            composition[self.element_to_idx[e]] += 1
        composition = composition / composition.sum()

        # 生成掩码
        mask = np.zeros(MAX_ATOMS)
        mask[:len(elements)] = 1

        return {
            'lattice': torch.tensor(lattice, dtype=torch.float),  # 新增
            'elements': torch.tensor(elem_indices, dtype=torch.long),
            'coords': torch.tensor(coords_padded, dtype=torch.float),
            'composition': torch.tensor(composition, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float)
        }


class CrystalDataset(Dataset):
    """修改点3：重构数据集类"""

    def __init__(self, cif_files, featurizer, training=False):
        self.cif_files = cif_files
        self.featurizer = featurizer

    def __len__(self):
        return len(self.cif_files)

    def __getitem__(self, idx):
        struct = Structure.from_file(self.cif_files[idx])
        data = self.featurizer(struct)
        return data


class CrystalVAE(nn.Module):
    """修改点4：重构VAE模型"""

    def __init__(self):
        super().__init__()

        # 元素嵌入层
        self.element_emb = nn.Embedding(ELEMENT_TYPES, 16)

        # 原子特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(16 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 128)
        )
        # 晶胞参数编码器
        self.lattice_encoder = nn.Sequential(
            nn.Linear(LATTICE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        # 潜在变量生成层
        self.fc_mu = nn.Linear(128 + 64, LATENT_DIM)  # 融合原子和晶胞特征
        self.fc_logvar = nn.Linear(128 + 64, LATENT_DIM)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + ELEMENT_TYPES, 512),
            nn.Unflatten(1, (64, 2, 2, 2)),  # 调整为5D张量 [batch, 64, 2, 2, 2]
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出尺寸计算: (2-1)*2 +3 =5
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出尺寸: (5-1)*2 +3 =11
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8 * 8, MAX_ATOMS * (ELEMENT_TYPES + 3))  # 根据实际输出尺寸调整
        )
        # 晶胞参数解码器
        self.decoder_lattice = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, LATTICE_DIM)
        )
        # 新增图注意力协调层
        self.gat_layer = nn.ModuleList([
            GATConv(in_channels=3 + 16,  # 坐标(3)+元素嵌入(16)
                    out_channels=64,
                    heads=3,
                    dropout=0.1),
            GATConv(64 * 3,  # 3 heads
                    3,  # 输出坐标调整量
                    heads=1,
                    concat=False,
                    dropout=0.1)
        ])

    def encode(self, atom_features, lattice_features):
        # 原子特征处理
        batch_size = atom_features.size(0)
        x_flat = atom_features.view(-1, atom_features.size(-1))
        encoded_atoms = self.encoder(x_flat)  # [batch*30, 128]
        encoded_atoms = encoded_atoms.view(batch_size, MAX_ATOMS, -1)
        encoded_atoms_global = encoded_atoms.mean(dim=1)  # [batch, 128]

        # 晶胞特征处理
        encoded_lattice = self.lattice_encoder(lattice_features)  # [batch, 64]

        # 特征融合
        combined = torch.cat([encoded_atoms_global, encoded_lattice], dim=1)
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        return mu, logvar

    def decode(self, z, composition):
        combined = torch.cat([z, composition], dim=1)
        decoded = self.decoder(combined)
        raw_output = decoded.view(-1, MAX_ATOMS, ELEMENT_TYPES + 3)

        # 分离元素和坐标预测
        element_logits = raw_output[..., :ELEMENT_TYPES]
        initial_coords = torch.sigmoid(raw_output[..., ELEMENT_TYPES:])

        # 生成图结构数据
        batch_size = initial_coords.size(0)
        node_features = []

        # 为每个原子构建特征：元素嵌入 + 初始坐标
        elem_emb = self.element_emb(torch.argmax(element_logits, dim=-1))  # [batch, MAX_ATOMS, 16]
        node_features = torch.cat([elem_emb, initial_coords], dim=-1)  # [batch, MAX_ATOMS, 16+3]

        # 转换为图处理需要的形状 [batch*MAX_ATOMS, features]
        node_features = node_features.view(-1, 19)

        # 构建全连接边（考虑所有原子间相互作用）
        if hasattr(self, 'full_edge_index'):
            edge_index = self.full_edge_index.repeat(1, batch_size) + \
                         (torch.arange(batch_size, device=DEVICE) * MAX_ATOMS).repeat_interleave(MAX_ATOMS ** 2).view(1,
                                                                                                                      -1)
        else:
            edge_index = []
            for b in range(batch_size):
                # 为每个样本生成完全连接的边
                num_nodes = MAX_ATOMS
                rows, cols = torch.meshgrid(
                    torch.arange(num_nodes, device=DEVICE),
                    torch.arange(num_nodes, device=DEVICE)
                )
                edge_index_b = torch.stack([rows.flatten(), cols.flatten()], dim=0)
                # 添加批次偏移（关键修复点）
                offset = b * num_nodes
                edge_index_b += offset
                edge_index.append(edge_index_b)
            edge_index = torch.cat(edge_index, dim=1).to(DEVICE)

        # 通过GAT层调整坐标
        x = F.elu(self.gat_layer[0](node_features, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        coord_adjust = torch.tanh(self.gat_layer[1](x, edge_index)) * 0.1  # 限制调整幅度

        # 应用坐标调整
        adjusted_coords = initial_coords.view(-1, 3) + coord_adjust
        adjusted_coords = adjusted_coords.view(batch_size, MAX_ATOMS, 3)

        # 合并最终输出
        final_output = torch.cat([element_logits, adjusted_coords], dim=-1)

        # 应用周期性约束
        final_output[..., ELEMENT_TYPES:] = final_output[..., ELEMENT_TYPES:] % 1.0

        return final_output, edge_index

    def forward(self, batch):
        # 原子特征嵌入
        elem_emb = self.element_emb(batch['elements'])

        atom_features = torch.cat([elem_emb, batch['coords']], dim=-1)

        # 编码
        mu, logvar = self.encode(atom_features, batch['lattice'])
        z = self.reparameterize(mu, logvar)

        # 解码
        recon_atom, edge_index = self.decode(z, batch['composition'])
        recon_lattice = self.decoder_lattice(z)

        return {
            'mu': mu,
            'logvar': logvar,
            'recon': recon_atom,
            'recon_lattice': recon_lattice,  # 新增
            'mask': batch['mask'],
            'edge_index': edge_index  # 新增
        }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def periodic_coord_loss(pred, target):
    # 计算两种可能的差值（考虑周期性边界）
    diff1 = torch.abs(pred - target)
    diff2 = 1 - diff1
    min_diff = torch.min(diff1, diff2)
    return torch.mean(min_diff ** 2)


# 原子间距离约束
def distance_loss(coords, mask):
    # 计算有效原子间的距离矩阵
    valid = coords[mask.bool()]
    if valid.shape[0] < 2:
        return torch.tensor(0.0, device=coords.device)
    dists = torch.pdist(valid)
    repel = torch.exp(-dists / 0.1)
    return repel.mean() * 0.5


def vae_loss(output, batch):
    """修改点5：改进的损失函数"""
    recon = output['recon']
    recon_lattice = output['recon_lattice']
    mu = output['mu']
    logvar = output['logvar']
    mask = batch['mask']

    # 检查输入是否正常
    def check_finite(name, tensor):
        if not torch.isfinite(tensor).all():
            print(f"❗张量 `{name}` 包含 NaN 或 Inf")
            raise ValueError(f"{name} 包含非法值")

    check_finite("recon", recon)
    check_finite("recon_lattice", recon_lattice)
    check_finite("mu", mu)
    check_finite("logvar", logvar)
    check_finite("mask", mask)
    # 元素分类损失
    element_pred = recon[:, :, :ELEMENT_TYPES]
    # === 新增logits值监控 ===
    print(f"元素logits范围: {element_pred.min().item():.2f} ~ {element_pred.max().item():.2f}")
    if torch.isnan(element_pred).any():
        print("⚠️ 检测到元素logits含NaN！")
    check_finite("element_pred", element_pred)
    per_atom_loss = F.cross_entropy(
        element_pred.permute(0, 2, 1),  # [batch, ELEMENT_TYPES, MAX_ATOMS]
        batch['elements'],  # [batch, MAX_ATOMS]
        reduction='none'  # 保持维度 [batch, MAX_ATOMS]
    )
    check_finite("per_atom_loss", per_atom_loss)
    mask_sum = mask.sum()
    if mask_sum == 0:
        raise ValueError("⚠️ mask.sum() == 0，无法计算 masked loss")
    # 应用掩码并计算平均损失
    element_loss = (per_atom_loss * mask).sum() / mask.sum()
    print(f"✅ elem loss: {element_loss.item()}")
    #  修改坐标损失为周期性版本
    coord_pred = recon[..., ELEMENT_TYPES:]
    coord_loss = periodic_coord_loss(
        coord_pred * mask.unsqueeze(-1),
        batch['coords'] * mask.unsqueeze(-1)
    )
    print(f"✅ Coord loss: {coord_loss.item()}")
    # 晶胞参数损失
    lattice_loss = F.mse_loss(recon_lattice, batch['lattice'])
    print(f"✅ Lattice loss: {lattice_loss.item()}")
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(f"✅ KL loss: {kl_loss.item()}")
    # 化学计量约束
    comp_pred = element_pred.softmax(dim=-1).mean(dim=1)
    check_finite("comp_pred", comp_pred)
    comp_loss = F.mse_loss(comp_pred, batch['composition'])
    print(f"✅ Composition loss: {comp_loss.item()}")
    repel_loss = distance_loss(output['recon'][..., ELEMENT_TYPES:], mask)
    print(f"✅ Repel loss: {repel_loss.item()}")
    print(f"μ range: {mu.min().item():.4f} ~ {mu.max().item():.4f}")
    print(f"logvar range: {logvar.min().item():.4f} ~ {logvar.max().item():.4f}")
    total_loss = (
            1.0 * element_loss +
            0.3 * coord_loss +
            0.3 * lattice_loss +
            0.1 * kl_loss +
            0.2 * comp_loss +
            0.5 * repel_loss
    )
    return total_loss, element_loss, coord_loss, lattice_loss, kl_loss, comp_loss, repel_loss


def post_process(output, composition):
    """修改点6：新增后处理函数"""
    mask = output['mask']  # [batch_size, MAX_ATOMS]

    # 应用化学计量约束
    element_probs = output['recon'][..., :ELEMENT_TYPES].softmax(dim=-1)
    element_counts = (element_probs * composition.unsqueeze(1)).sum(dim=1)
    elements = torch.multinomial(element_probs.view(-1, ELEMENT_TYPES), 1)
    elements = elements.view(output['recon'].shape[0], -1)

    # 调整坐标
    coords = output['recon'][..., ELEMENT_TYPES:]
    # 按样本处理有效原子
    batch_valid = []
    for i in range(mask.size(0)):
        sample_mask = mask[i].bool()
        valid_elements = elements[i][sample_mask].to(DEVICE)
        valid_coords = coords[i][sample_mask].to(DEVICE)
        batch_valid.append((valid_elements, valid_coords))
        # 距离约束调整
        for j in range(len(valid_coords)):
            for k in range(j + 1, len(valid_coords)):
                dist = torch.norm(valid_coords[j] - valid_coords[k])
                if dist < 1.0:
                    adjustment = torch.randn(3, device=DEVICE) * 0.1
                    valid_coords[k] += adjustment

        batch_valid.append((valid_elements, valid_coords))

    return batch_valid


def collate_fn(batch):
    """修改点7：自定义批处理函数"""
    return {
        'lattice': torch.stack([x['lattice'] for x in batch]).to(DEVICE),  # 新增
        'elements': pad_sequence([x['elements'] for x in batch], batch_first=True).to(DEVICE),
        'coords': pad_sequence([x['coords'] for x in batch], batch_first=True).to(DEVICE),
        'composition': torch.stack([x['composition'] for x in batch]).to(DEVICE),
        'mask': pad_sequence([x['mask'] for x in batch], batch_first=True).to(DEVICE)
    }


def train_model(model, data_loader, plot_dir, epochs=100):
    """修改点8：改进训练循环"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            for key, value in batch.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    print(f"❗ 输入 batch 的 {key} 含有 NaN 或 Inf")
                    raise ValueError("输入数据无效，含 NaN 或 Inf")
            optimizer.zero_grad()
            output = model(batch)
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    if not torch.isfinite(value).all():
                        print(f"⚠️ 模型输出 `{key}` 包含 NaN 或 Inf")
                        print(f"→ 最小值: {value.min()}, 最大值: {value.max()}")
                        raise ValueError(f"模型输出 `{key}` 非法")
            '''
            if batch_idx == 0 and epoch == 0:
                edge_index = output['edge_index']
                print("Sample edge_index check:")
                print("First sample edges:", edge_index[:, :MAX_ATOMS**2])
                print("Second sample edges:", edge_index[:, MAX_ATOMS**2:2*MAX_ATOMS**2])
            '''
            loss, element_loss, coord_loss, lattice_loss, kl_loss, comp_loss, repel_loss = vae_loss(output, batch)

            '''
            if not torch.isfinite(loss):
              print("⚠️ Loss 出现 NaN 或 Inf！详细信息如下：")
              print(f"Element loss: {element_loss.item()}")
              print(f"Coord loss: {coord_loss.item()}")
              print(f"Lattice loss: {lattice_loss.item()}")
              print(f"KL loss: {kl_loss.item()}")
              print(f"Composition loss: {comp_loss.item()}")
              print(f"Repel loss: {repel_loss.item()}")
              raise ValueError("Loss is NaN or Inf. 停止训练。")
            '''
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()

            #  每100个batch打印距离信息
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    valid_coords = output['recon'][..., ELEMENT_TYPES:][batch['mask'].bool()]
                    if valid_coords.shape[0] > 1:
                        dists = torch.pdist(valid_coords)
                        print(f"[Batch {batch_idx}] Min dist: {dists.min().item():.2f} Å")
        scheduler.step()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")
        if epoch % 5 == 0:
            with torch.no_grad():
                sample = model(next(iter(data_loader)))
                coords = sample['recon'][..., ELEMENT_TYPES:].cpu()
                plt.figure(figsize=(10, 6))
                plt.hist(coords.flatten().numpy(), bins=50, range=(0, 1))
                plt.title(f'Epoch {epoch + 1} Coordinate Distribution')
                plt.xlabel('Fractional Coordinate')
                plt.ylabel('Count')
                '''
                # 添加距离分布可视化
                valid_coords = coords[sample['mask'].bool().cpu()]
                if len(valid_coords) > 1:
                    # 转换为NumPy数组
                       valid_coords_np = valid_coords.detach().cpu().numpy()
                       delta = np.abs(valid_coords_np[:, None] - valid_coords_np)
                       delta = np.minimum(delta, 1 - delta)
                       dist_matrix = np.sqrt(np.sum(delta**2, axis=-1))
                       dists = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
   
                       plt.subplot(1,2,2)
                       plt.hist(dists.flatten(), bins=50, range=(0,3))
                       plt.title('Interatomic Distance Distribution')
                       '''
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"epoch_{epoch + 1}_coords.png"))
                plt.close()  # 必须关闭图形

        # 每10个epoch保存并生成示例
        if (epoch + 1) % 10 == 0:
            save_dir = "C:/Users/19754/Desktop/checkpoints5"  # 定义保存目录
            os.makedirs(save_dir, exist_ok=True)  # 自动创建目录
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))
            with torch.no_grad():
                sample = model(batch)
                results = post_process(sample, batch['composition'])
                # 显示第一个样本的结果
                valid_elements, valid_coords = results[0]
                print(f"Generated structure with {len(valid_elements)} atoms")


# 修改点3：独立生成器类
class CrystalGenerator:
    def __init__(self, model_path, element_list):
        self.device = DEVICE
        self.element_list = element_list
        self.model = self._load_model(model_path)
        self.valid_structures = []
        self.struct_matcher = structure_matcher.StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)

    def _load_model(self, path):
        """加载模型并确保设备正确"""
        model = CrystalVAE().to(self.device)
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
        model.eval()
        return model

    def _denormalize(self, coords, lattice_params):
        """根据训练时的归一化方式反归一化"""
        # 假设训练时用平均晶格长度归一化
        scale_factor = 10
        return coords, lattice_params * scale_factor

    # 修改点4：改进的特征转换方法
    def _output_to_structure(self, output):
        """将模型输出转换为晶体结构（关键修改点1：适配新模型输出）"""
        # 获取模型输出
        atom_recon = output['recon'].detach()  # [batch, 30, 86]
        lattice_recon = output['recon_lattice'].detach()  # [batch, 9]
        mask = output['mask'].detach()  # [batch, 30]

        # 反归一化处理
        coords = atom_recon[0, :, ELEMENT_TYPES:].cpu().numpy()
        lattice_params = lattice_recon[0].cpu().numpy()
        coords, lattice_params = self._denormalize(coords, lattice_params)

        # 创建晶格
        lattice = Lattice(lattice_params.reshape(3, 3))

        # 采样元素类型（关键修改点2：应用掩码过滤）
        element_probs = F.softmax(atom_recon[0, :, :ELEMENT_TYPES], dim=-1)
        elements = torch.multinomial(element_probs, 1).view(-1).cpu().numpy()

        # 过滤无效原子
        valid_mask = (mask[0].bool().cpu().numpy())
        valid_elements = elements[valid_mask]
        valid_coords = coords[valid_mask]

        # 转换为元素符号
        try:
            element_symbols = [self.element_list[i] for i in valid_elements]  # 0-based索引
        except IndexError as e:
            raise ValueError(f"无效元素索引: {valid_elements}") from e

        print("模型输出键:", output.keys())
        print("recon形状:", output['recon'].shape)
        print("recon_lattice形状:", output['recon_lattice'].shape)
        return Structure(lattice, element_symbols, valid_coords, coords_are_cartesian=False)

    # 修改点5：增强的验证逻辑
    def _validate_structure(self, struct):
        """验证结构合理性（分数坐标版本）"""
        # 基本检查
        if len(struct) < 2:
            return False
        if 'Li' not in [e.symbol for e in struct.composition.elements]:
            return False

        try:
            # 创建晶格副本用于计算
            lattice = struct.lattice

            dist_matrix = struct.distance_matrix
            min_dist = np.min(dist_matrix[np.nonzero(dist_matrix)])  # 排除对角线的0值

            # 检查最近邻距离
            if min_dist < 0.5:  # 单位：Å
                return False

            # (修正点2) 精确计算分数坐标容差
            max_lattice_param = max(lattice.abc)
            merged_struct = struct.copy()
            merged_struct.merge_sites(
                tol=0.2 / max_lattice_param,  # 动态适应晶格参数
                mode="delete"
            )

            # 检查是否删除过多原子
            if len(merged_struct) < len(struct) * 0.5:
                return False

        except Exception as e:
            print(f"结构验证异常: {str(e)}")
            return False

        # 去重检查
        return not any(self.struct_matcher.fit(merged_struct, s) for s in self.valid_structures)

    def generate(self, batch_size=1):
        """核心生成方法（关键修改点4：适配VAE生成逻辑）"""
        with torch.no_grad():
            # 从先验分布采样
            z = torch.randn(batch_size, LATENT_DIM).to(self.device)

            # 生成随机成分向量（可根据需求修改）
            composition = F.softmax(torch.randn(batch_size, ELEMENT_TYPES), dim=-1).to(self.device)

            # 解码生成
            recon_atom, edge_index = self.model.decode(z, composition)
            recon_lattice = self.model.decoder_lattice(z)
            print("[DEBUG] recon_lattice shape:", recon_lattice.shape)  # 应为 [batch_size, 9]
            # 根据元素概率动态生成掩码
            element_probs = recon_atom[..., :ELEMENT_TYPES].softmax(dim=-1)
            mask = (element_probs.max(dim=-1).values > 0.4)  # 概率>10%视为有效

            return {
                'recon': recon_atom,
                'recon_lattice': recon_lattice,
                'mask': mask
            }

    # 修改点6：批量生成方法
    def generate_cif(self, num_structures=10, batch_size=5, output_dir="generated_cifs"):
        """批量生成晶体结构"""
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        for i in range(0, num_structures, batch_size):
            # 批量生成
            batch = min(batch_size, num_structures - i)
            output = self.generate(batch_size=batch)
            for j in range(batch):
                try:
                    struct = self._output_to_structure({
                        'recon': output['recon'][j:j + 1],
                        'recon_lattice': output['recon_lattice'][j:j + 1],
                        'mask': output['mask'][j:j + 1]  # 确保添加此行
                    })
                    print("[DEBUG] 生成的结构:", struct)
                    if self._validate_structure(struct):
                        cif_path = os.path.join(output_dir, f"gen_{i + j}.cif")
                        struct.to(filename=cif_path)
                        generated_files.append(cif_path)
                        self.valid_structures.append(struct)
                except Exception as e:
                    traceback.print_exc()  # 打印完整错误堆栈
                    print(f"生成失败: {str(e)}")
        print(f"成功生成 {len(generated_files)}/{num_structures} 个有效结构")
        return generated_files


# # 主程序
# if __name__ == "__main__":
#     # 初始化元素列表
#     global_elements = [Element.from_Z(z+1).symbol for z in range(ELEMENT_TYPES)]
#
#     # 初始化生成器
#     generator = CrystalGenerator(
#         model_path="C:/Users/19754/Desktop/checkpoints5/model_epoch_100.pth",
#         element_list=global_elements
#     )
#
#     # 生成结构
#     generated_files = generator.generate_cif(
#         num_structures=1000,  # 生成足够样本保证有效输出
#         batch_size=16,       # 根据GPU显存调整（T4建议16，V100建议32）
#         output_dir="C:/Users/19754/Desktop/generate_cifs2"
#     )
#
#     print(f"成功生成{len(generated_files)}个有效结构：")
#     for path in generated_files:
#         print(f"- {path}")

def main(model_path: str, num_structures: int, batch_size: int, output_dir: str):
    """批量生成晶体结构并保存为 CIF 文件"""
    # 初始化元素列表
    global_elements = [Element.from_Z(z + 1).symbol for z in range(ELEMENT_TYPES)]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化生成器
    generator = CrystalGenerator(
        model_path=model_path,
        element_list=global_elements
    )

    # 生成结构
    generated_files = generator.generate_cif(
        num_structures=num_structures,
        batch_size=batch_size,
        output_dir=output_dir
    )

    # 打印结果
    print(f"成功生成 {len(generated_files)} 个有效结构：")
    for path in generated_files:
        print(f"- {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 CrystalGenerator 批量生成 CIF 文件"
    )
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        default="C:/Users/19754/Desktop/checkpoints5/model_epoch_100.pth",
        help="PyTorch 模型文件路径，例如 C:/path/to/model.pth"
    )
    parser.add_argument(
        "--num_structures", "-n",
        type=int,
        default=1000,
        help="要生成的总结构数，默认为 1000"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="每次生成的批大小，默认为 16"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="CIF 文件输出目录"
    )

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        num_structures=args.num_structures,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )