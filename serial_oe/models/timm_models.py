import timm
import torch
import torch.nn as nn
import torchaudio.transforms as T


class Backbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, in_chans=3):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)

        if "regnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "res" in name:
            self.out_features = self.net.fc.in_features
        elif "efficientnet" in name:
            self.out_features = self.net.classifier.in_features
        elif "senet" in name:
            self.out_features = self.net.fc.in_features
        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x


class ASDModel(nn.Module):
    def __init__(
        self,
        backbone,
        embedding_size=128,
        pretrained=False,
        n_fft=2048,
        hop_length=256,
        n_mels=224,
        f_min=50,
        f_max=7800,
        power=1,
        out_dim=7,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        self.backbone = Backbone(backbone, pretrained=pretrained, in_chans=1)
        self.melspectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=0,
            n_mels=n_mels,
            power=power,
            normalized=True,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU(),
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
        )
        self.machine_head = nn.Linear(1, 1, bias=True)
        self.product_head = nn.Linear(self.embedding_size, out_dim, bias=False)

    def forward(self, input):
        x = self.melspectrogram(input).unsqueeze(1)
        x = self.backbone(x)
        x = self.global_pool(x)[:, :, 0, 0]
        embedding = self.neck(x)
        machine = self.machine_head(
            torch.pow(embedding, 2).sum(dim=1).unsqueeze(1) / self.embedding_size
        )
        product = self.product_head(embedding)
        output_dict = {
            "embedding": embedding,
            "machine": machine,
            "product": product,
        }
        return output_dict
