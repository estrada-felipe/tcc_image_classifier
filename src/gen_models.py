import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import SwinModel, SwinConfig, ViTModel, ViTConfig
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    resnet18, ResNet18_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
)

from torchvision.models.resnet import ResNet, BasicBlock

class CoAtNet(nn.Module):
    def __init__(self, input_shape, num_classes, channels=[64, 128, 256, 512]):
        super().__init__()
        in_channels = input_shape[2]
        self.stage1 = nn.Sequential(
            CoAtNetBlock(in_channels, channels[0], kernel_size=3, stride=1),
            CoAtNetBlock(channels[0], channels[1], kernel_size=3, stride=2)
        )
        self.stage2 = nn.Sequential(
            CoAtNetBlock(channels[1], channels[2], kernel_size=3, stride=2),
            CoAtNetBlock(channels[2], channels[2], kernel_size=3, stride=1, attention=True)
        )
        self.stage3 = nn.Sequential(
            CoAtNetBlock(channels[2], channels[3], kernel_size=3, stride=2, attention=True),
            CoAtNetBlock(channels[3], channels[3], kernel_size=3, stride=1, attention=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CoAtNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, attention=False):
        super().__init__()
        self.attention = attention
        self.out_channels = out_channels

        if not attention:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.qkv = nn.Conv2d(in_channels, 3 * out_channels, kernel_size=1)
            self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.act = nn.ReLU()

    def forward(self, x):
        if not self.attention:
            x = self.conv(x)
            x = self.bn(x)
        else:
            batch_size, channels, height, width = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=1)
            q = q.flatten(2).transpose(1, 2)
            k = k.flatten(2)
            v = v.flatten(2).transpose(1, 2)
            attn = torch.bmm(q, k) / (k.shape[-1] ** 0.5)
            attn = F.softmax(attn, dim=-1)
            x = torch.bmm(attn, v)
            x = x.transpose(1, 2).reshape(batch_size, self.out_channels, height, width)
            x = self.proj(x)
        return self.act(x)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SeqPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention_pool = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attention_pool(x), dim=1)  # [batch_size, num_patches, 1]
        pooled = torch.sum(weights * x, dim=1)  # [batch_size, embed_dim]
        return pooled

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, attn_mask=None):
        # Pré-Norm + Residual na Atenção
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_output

        # Pré-Norm + Residual no Feed Forward
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class ViTLite(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, seq_pool=True):
        super().__init__()
        self.seq_pool = seq_pool
        self.embed_dim = embed_dim
        self.depth = depth   
        self.num_heads = num_heads
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Sempre inicializa o CLS Token e Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.encoder = nn.Sequential(*[TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)])
        self.pooling = SeqPooling(embed_dim) if seq_pool else None
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size, _, _ = x.size()

        # Sempre adiciona o CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        x = self.encoder(x)

        if self.seq_pool:
            x = x[:, 1:, :]  # Ignora CLS Token
            x = self.pooling(x)
        else:
            x = x[:, 0, :]  # Usa CLS Token

        return self.head(x)

class CVT(ViTLite):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, seq_pool=True):
        super().__init__(img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, seq_pool)


class CCT(nn.Module):
    def __init__(self, img_size, in_channels, num_classes, embed_dim, depth, num_heads, 
                 kernel_size=3, num_conv_layers=2, seq_pool=False, dropout=0.1):
        super().__init__()
        self.seq_pool = seq_pool
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout

        # Tokenizador aprimorado com convoluções
        self.tokenizer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else embed_dim, embed_dim, 
                              kernel_size=kernel_size, padding=kernel_size // 2, stride=1),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ) for i in range(num_conv_layers)
            ]
        )

        # Calcula o número de patches dinamicamente
        num_patches = (img_size // (2 ** num_conv_layers)) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )

        self.pooling = SeqPooling(embed_dim) if seq_pool else None
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        batch_size, _, _ = x.size()

        # Sempre adiciona o CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Adiciona embeddings posicionais
        x = x + self.pos_embed


        # Passa pelo Transformer Encoder
        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1)

        # Escolha entre Pooling ou CLS Token
        if self.seq_pool:
            x = x[:, 1:, :]  # Ignora CLS Token
            x = self.pooling(x)
        else:
            x = x[:, 0, :]  # Usa CLS Token


        # Classificação
        x = self.head(x)
        return x

class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class ResNet6(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Camada inicial
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Camadas intermediárias
        self.layer1 = self._make_layer(SingleConvBlock, 64, 1)
        self.layer2 = self._make_layer(SingleConvBlock, 128, 2)
        self.layer3 = self._make_layer(SingleConvBlock, 256, 2)
        self.layer4 = self._make_layer(SingleConvBlock, 512, 2)

        # Camada final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x    


class ResNet8(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Camada inicial
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Camadas intermediárias
        self.layer1 = self._make_layer(SingleConvBlock, 64, 2)
        self.layer2 = self._make_layer(SingleConvBlock, 128, 2)
        self.layer3 = self._make_layer(SingleConvBlock, 256, 2)
        # self.layer4 = self._make_layer(SingleConvBlock, 512, 2)

        # Camada final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block, out_channels, num_blocks):
        layers = []
        # Primeiro bloco pode ter stride 2 para downsampling
        layers.append(block(self.in_channels, out_channels, stride=2 if self.in_channels != out_channels else 1))
        self.in_channels = out_channels
        # Blocos adicionais
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

class ResNetTokenizer(nn.Module):
    def __init__(self, resnet_model, embed_dim):
        super().__init__()
        self.resnet = nn.Sequential(*list(resnet_model.children())[:-2])
        self.conv1x1 = nn.Conv2d(resnet_model.fc.in_features, embed_dim, kernel_size=1)
        self.num_patches = None  

    def forward(self, x):
        x = self.resnet(x)

        x = self.conv1x1(x) 

        self.num_patches = x.shape[-2] * x.shape[-1]
        x = x.flatten(2).transpose(1, 2)

        return x

class CCTWithResNetTokenizer(nn.Module):
    def __init__(self, resnet_model, img_size, num_classes, embed_dim, depth, num_heads, seq_pool=False, dropout=0.1):
        super().__init__()
        self.seq_pool = seq_pool
        self.embed_dim = embed_dim 
        self.depth = depth      
        self.num_heads = num_heads  
        self.dropout = dropout

        self.tokenizer = ResNetTokenizer(resnet_model, embed_dim)
        self.pos_embed = None
        if not seq_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        

        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )

        self.pooling = SeqPooling(embed_dim) if seq_pool else None

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.tokenizer(x) 
        batch_size, _, _ = x.size()

        if self.pos_embed is None:
            self.pos_embed = nn.Parameter(
        torch.zeros(1, x.shape[1] + (1 if not self.seq_pool else 0), x.shape[2])
        )


        if not self.seq_pool:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)


        x = x + self.pos_embed

        x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.transpose(0, 1) 

        if self.seq_pool:
            x = self.pooling(x)
        else:
            x = x[:, 0, :]

        return self.head(x)


class ModelFactory_Tiny:
    def __init__(self, num_classes, input_shape, pretrained=False,  model_params=None):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.IMAGE_SIZE_H, self.IMAGE_SIZE_W, self.CHANNELS = input_shape
        self.pretrained = pretrained
        self.model_params = model_params if model_params else {} 

    
    def create_model(self, architecture):
        architecture = architecture.lower()
        if architecture == "efficientnet":
            return self.create_efficientnet_tiny_model()
        
        elif architecture == "resnet18":
            return self.create_resnet_tiny_model()
        
        elif architecture == "resnet10":
            def resnet10(pretrained=False, **kwargs):
                """
                Implementação de ResNet10 personalizada.
                """
                model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)  # Um bloco em cada estágio
                if pretrained:
                    raise NotImplementedError("Pesos pré-treinados não estão disponíveis para ResNet10.")
                return model

            resnet10_model = resnet10()

            resnet10_model.fc = nn.Linear(resnet10_model.fc.in_features, self.num_classes)

            return resnet10_model

        elif architecture == "resnet8":
            resnet8_model = ResNet8()

            resnet8_model.fc = nn.Linear(resnet8_model.fc.in_features, self.num_classes)

            return resnet8_model

        elif architecture == "resnet6":
            resnet6_model = ResNet6()

            resnet6_model.fc = nn.Linear(resnet6_model.fc.in_features, self.num_classes)

            return resnet6_model
                
        elif architecture == "coatnet":
            return CoAtNet(self.input_shape, self.num_classes, channels=self.model_params.get("channels", [32, 64, 128, 256]))
        
        elif architecture == "vision transformer":
            return self.create_vit_tiny_model()

        elif architecture == "vitlite":
            return ViTLite(    
                img_size=self.IMAGE_SIZE_H,
                patch_size=self.model_params.get("patch_size", 4),
                in_channels=self.CHANNELS,
                num_classes=self.num_classes,
                embed_dim=self.model_params.get("embed_dim", 96),               # 64
                depth=self.model_params.get("depth", 3),                        # 3
                num_heads=self.model_params.get("num_heads", 2),                # 2
                seq_pool=self.model_params.get("seq_pool", True)                
            )
        
        elif architecture == "cvt":
            return CVT(
                img_size=self.IMAGE_SIZE_H,
                patch_size=self.model_params.get("patch_size", 4),
                in_channels=self.CHANNELS,
                num_classes=self.num_classes,
                embed_dim=self.model_params.get("embed_dim", 96),               # 64
                depth=self.model_params.get("depth", 4),                        # 3
                num_heads=self.model_params.get("num_heads", 2),                # 2
                seq_pool=self.model_params.get("seq_pool", True)
            )
        
        elif architecture == "cct":
            return CCT(
                img_size=self.IMAGE_SIZE_H,
                in_channels=self.CHANNELS,
                num_classes=self.num_classes,
                embed_dim=self.model_params.get("embed_dim", 96),               # 64
                depth=self.model_params.get("depth", 4),                        # 4
                num_heads=self.model_params.get("num_heads", 2),                # 2
                kernel_size=self.model_params.get("kernel_size", 3),
                num_conv_layers=self.model_params.get("num_conv_layers", 3),
                seq_pool=self.model_params.get("seq_pool", True),
                dropout=self.model_params.get("dropout", 0.3)
            )
        
        elif architecture == 'resnet18vit':
            # Initialize ResNet-18 with optional pretraining
            resnet18_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None)

            return CCTWithResNetTokenizer(
                resnet_model=resnet18_model,
                img_size=self.IMAGE_SIZE_H,
                num_classes=self.num_classes,
                embed_dim=self.model_params.get("embed_dim", 96),   # Default: 96
                depth=self.model_params.get("depth", 5),            # Default: 4 
                num_heads=self.model_params.get("num_heads", 3),    # Default: 2
                seq_pool=self.model_params.get("seq_pool", True),   # Default: True
                dropout=self.model_params.get("dropout", 0.25)      # Default: 0.3
            )
       
        elif architecture == 'resnet10vit':
            def resnet10(pretrained=False, **kwargs):
                """
                Implementação de ResNet10 personalizada.
                """
                model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)  # Um bloco em cada estágio
                if pretrained:
                    raise NotImplementedError("Pesos pré-treinados não estão disponíveis para ResNet10.")
                return model

            resnet10_model = resnet10()

            return CCTWithResNetTokenizer(
                resnet_model=resnet10_model,
                img_size=self.IMAGE_SIZE_H,
                num_classes=self.num_classes,
                embed_dim=self.model_params.get("embed_dim", 128),  # Default: 96
                depth=self.model_params.get("depth", 5),            # Default: 4 
                num_heads=self.model_params.get("num_heads", 4),    # Default: 2
                seq_pool=self.model_params.get("seq_pool", True),   # Default: True
                dropout=self.model_params.get("dropout", 0.25)      # Default: 0.3
            )
        elif architecture == 'resnet8vit':
            resnet08_model = ResNet8()

            return CCTWithResNetTokenizer(
                resnet_model=resnet08_model,
                img_size=self.IMAGE_SIZE_H,
                num_classes=self.num_classes,
                embed_dim=self.model_params.get("embed_dim", 128),  # Default: 96
                depth=self.model_params.get("depth", 5),            # Default: 4 
                num_heads=self.model_params.get("num_heads", 4),    # Default: 2
                seq_pool=self.model_params.get("seq_pool", True),   # Default: True
                dropout=self.model_params.get("dropout", 0.25)      # Default: 0.3
            )
        
        elif architecture == 'resnet6vit':

            def resnet06(pretrained=False, **kwargs):
                """
                Implementação de ResNet6 personalizada.
                """
                model = ResNet6(**kwargs)
                if pretrained:
                    raise NotImplementedError("Pesos pré-treinados não estão disponíveis para ResNet6.")
                return model
            
            resnet06_model = resnet06()

            return CCTWithResNetTokenizer(
                resnet_model=resnet06_model,
                img_size=self.IMAGE_SIZE_H,
                num_classes=self.num_classes,
                embed_dim=self.model_params.get("embed_dim", 96),  # Default: 96
                depth=self.model_params.get("depth", 3),            # Default: 4 
                num_heads=self.model_params.get("num_heads", 2),    # Default: 2
                seq_pool=self.model_params.get("seq_pool", True),   # Default: True
                dropout=self.model_params.get("dropout", 0.15)      # Default: 0.3
            )

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def create_vit_tiny_model(self):
        image_size = self.input_shape[0]  # Tamanho da imagem (ex: 64)
        patch_size = self.model_params.get("patch_size", 4)
        # Verifica se patch_size é divisor do tamanho da imagem
        if image_size % patch_size != 0:
            # Ajusta o patch_size para ser compatível
            for p in range(patch_size, 0, -1):
                if image_size % p == 0:
                    patch_size = p
                    break
            print(f"⚠️ patch_size ajustado para: {patch_size}")


        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=self.input_shape[2],
            hidden_size=self.model_params.get("hidden_size", 96),
            num_hidden_layers=self.model_params.get("depth", 4),
            num_attention_heads=self.model_params.get("num_heads", 2),
            intermediate_size=self.model_params.get("intermediate_size", 768),
            num_labels=self.num_classes,
        )

        vit_model = ViTModel(config)

        class ViTTinyClassifier(nn.Module):
            def __init__(self, vit_model, hidden_size, num_classes):
                super().__init__()
                self.vit_model = vit_model
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                vit_output = self.vit_model(x)           

                pooled_output = vit_output.pooler_output 

                logits = self.fc(pooled_output) 


        model = ViTTinyClassifier(vit_model, config.hidden_size, self.num_classes)
        return model

    def create_efficientnet_tiny_model(self):
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if self.pretrained else None
        model = efficientnet_b0(weights=weights)  # Usando EfficientNet-B0 (leve e eficiente)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        return model

    def create_resnet_tiny_model(self):
        weights = ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model
 