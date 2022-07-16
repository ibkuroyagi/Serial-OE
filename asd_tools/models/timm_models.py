import timm
import torch
import torch.nn as nn
import torchaudio.transforms as T
import logging


class Backbone(nn.Module):
    def __init__(self, name="resnet18", pretrained=False, in_chans=3):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)

        if "regnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "res" in name:  # works also for resnest
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
        use_pos=False,
        in_chans=3,
        n_fft=2048,
        hop_length=256,
        n_mels=224,
        f_min=50,
        f_max=7800,
        power=1,
        out_dim=7,
        time_mask_param=0,
        freq_mask_param=0,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_chans = in_chans
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        self.use_pos = use_pos
        self.backbone = Backbone(backbone, pretrained=pretrained, in_chans=in_chans)
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
        if use_pos:
            self.amplitude2db = T.AmplitudeToDB(stype=power)
            self.melspectrogram1 = T.MelSpectrogram(
                sample_rate=16000,
                n_fft=n_fft // 2,
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
        if time_mask_param != 0:
            self.timemask = T.TimeMasking(time_mask_param=time_mask_param)
        else:
            self.timemask = None
        if freq_mask_param != 0:
            self.freqmask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        else:
            self.freqmask = None
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU(),
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
        )

        self.machine_head = nn.Linear(1, 1, bias=True)
        self.section_head = nn.Linear(self.embedding_size, out_dim, bias=False)

    def forward(self, input, specaug=False):
        x = self.melspectrogram(input)
        if specaug:
            if self.timemask is not None:
                x = self.timemask(x)
            if self.freqmask is not None:
                x = self.freqmask(x)
            # logging.info(f"specaug:{x.shape}")
        x = x.unsqueeze(1)
        # logging.info(f"unsqueeze:{x.shape}")
        if self.use_pos:
            x1 = self.amplitude2db(x)
            x2 = self.amplitude2db(self.melspectrogram1(input))
            x = torch.cat([x, x1, x2.unsqueeze(1)], dim=1)
        else:
            x = x.expand(-1, 3, -1, -1)
        # logging.info(f"before x:{x.shape}")
        x = self.backbone(x)
        x = self.global_pool(x)[:, :, 0, 0]
        embedding = self.neck(x)
        machine = self.machine_head(
            torch.pow(embedding, 2).sum(dim=1).unsqueeze(1) / self.embedding_size
        )
        section = self.section_head(embedding)
        output_dict = {
            "embedding": embedding,
            "machine": machine,
            "section": section,
        }

        return output_dict
