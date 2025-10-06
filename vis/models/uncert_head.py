# -*- coding: utf-8 -*-
"""
2D uncertainty head for visual features.
Predicts per-axis log-variance (log σx², log σy²) from patch pairs + geometry.
"""
import torch
import torch.nn as nn

class UncertHead2D(nn.Module):
    """
    Lightweight CNN+MLP head for 2D uncertainty estimation.
    
    Args:
        in_ch: Input channels (2 for concatenated patch pair)
        geom_dim: Dimension of geometric features
        d: Hidden dimension for CNN
        h: Hidden dimension for MLP
        out_dim: Output dimension (2 for [log σx², log σy²])
    """
    def __init__(self, in_ch=2, geom_dim=4, d=64, h=128, out_dim=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch,  32, 3, 2, 1), nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(32,     64, 3, 2, 1), nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(64,    128, 3, 2, 1), nn.ReLU(inplace=True),  # 8->4
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(128 + geom_dim, h), nn.ReLU(inplace=True),
            nn.Linear(h, out_dim)  # -> [logσx², logσy²]
        )

    def forward(self, patch2, geom):
        """
        Args:
            patch2: [B,2,H,W] concatenated patches
            geom: [B,F] geometric features
        
        Returns:
            [B,2] predicted log-variances [log σx², log σy²]
        """
        f = self.cnn(patch2).flatten(1)  # [B,128]
        y = self.mlp(torch.cat([f, geom], dim=1))
        return y  # [B,2] (lvx, lvy)


class UncertHead_ResNet_CrossAttention(nn.Module):
    """
    终极版模型：使用预训练的ResNet-18作为图像主干，
    并结合交叉注意力来智能融合图像和几何特征。
    """
    def __init__(self, in_ch=2, geom_dim=11, d_model=128, n_heads=4, out_dim=2, pretrained=True):
        """
        Args:
            in_ch (int): 输入图像通道数，对于您的patch对是2。
            geom_dim (int): 几何特征向量的维度，是11。
            d_model (int): 模型的核心维度。ResNet-18的layer2输出是128，所以这里设为128。
            n_heads (int): 交叉注意力机制的头数。
            out_dim (int): 最终输出维度，是2 ([log_var_x, log_var_y])。
            pretrained (bool): 是否加载ImageNet预训练权重。
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        # 1. 图像主干网络 (Backbone) - 使用预训练的ResNet-18
        try:
            import torchvision.models as models
            
            if pretrained:
                # 使用推荐的默认权重
                weights = models.ResNet18_Weights.DEFAULT
                print("[model] Loading pretrained ResNet-18 weights.")
            else:
                weights = None
                print("[model] Training ResNet-18 from scratch.")
            
            resnet = models.resnet18(weights=weights)
        except Exception as e:
            print(f"[model] Warning: Failed to load ResNet-18: {e}")
            print("[model] Using ResNet-18 without pretrained weights")
            import torchvision.models as models
            resnet = models.resnet18(weights=None)

        # --- 关键修改：适配2通道输入 ---
        # 原始的第一个卷积层 resnet.conv1 是为3通道(RGB)设计的。
        # 我们需要创建一个新的卷积层来处理我们的2通道输入(patch1, patch2)。
        original_conv1 = resnet.conv1
        self.cnn_stem = nn.Conv2d(
            in_ch, original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        if pretrained:
            # 将预训练的权重迁移到新的卷积层。
            # 常用技巧：取原始3通道权重的均值，然后复制到2个新通道上。
            with torch.no_grad():
                self.cnn_stem.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)

        # 我们只需要ResNet的特征提取部分，去掉最后的全局池化和全连接分类头。
        # 对于32x32的输入，经过到layer2的处理后，特征图大小为4x4，通道数为128。
        self.cnn_body = nn.Sequential(
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2  # 输出特征图形状: [B, 128, 4, 4]
        )
        
        # 2. 几何特征归一化 + 投影器
        # LayerNorm用于归一化几何特征，避免量纲跨度大导致注意力偏科
        self.geom_norm = nn.LayerNorm(geom_dim)
        # 将归一化后的几何特征投影到与图像特征相同的维度 d_model
        self.geom_projector = nn.Linear(geom_dim, d_model)

        # 3. 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            batch_first=True  # 确保输入/输出形状为 (Batch, Sequence, Dim)
        )
        
        # 4. Layer Normalization 和 MLP 预测头（加入Dropout防止过拟合）
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # Dropout=0.4 配合数据增强，加强正则化
            nn.Linear(d_model * 2, out_dim)
        )

    def forward(self, patch2, geom):
        """
        前向传播过程
        
        Args:
            patch2: [B,2,H,W] concatenated patches
            geom: [B,F] geometric features
        
        Returns:
            [B,2] predicted log-variances [log σx², log σy²]
        """
        # 1. 提取高质量的图像特征图
        # patch2 [B, 2, 32, 32] -> f_img [B, 128, 4, 4]
        x = self.cnn_stem(patch2)
        f_img = self.cnn_body(x)
        
        # 2. 准备交叉注意力的输入 (Query, Key, Value)
        B, C, H, W = f_img.shape
        
        # a) 将特征图展平并重排，作为 Key 和 Value
        # f_img [B, 128, 4, 4] -> [B, 128, 16] -> [B, 16, 128]
        key_value = f_img.flatten(2).permute(0, 2, 1)

        # b) 将几何特征归一化、投影并重塑，作为 Query
        # geom [B, 11] -> normalize -> [B, 11] -> [B, 128] -> [B, 1, 128]
        geom_normalized = self.geom_norm(geom)
        query = self.geom_projector(geom_normalized).unsqueeze(1)
        
        # 3. 执行交叉注意力
        # Query: 几何特征, Key/Value: 图像特征
        attn_output, _ = self.cross_attn(query=query, key=key_value, value=key_value)
        
        # 4. 残差连接与归一化 (Transformer block的标准操作)
        h = self.norm1(query + attn_output)
        
        # 5. 通过MLP进行最终预测
        # h [B, 1, 128] -> [B, 128] -> [B, 2]
        y = self.mlp(h.squeeze(1))
        
        return y