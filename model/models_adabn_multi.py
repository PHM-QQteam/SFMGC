import torch.nn as nn


class MultiDomainBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.bn_source = nn.BatchNorm1d(num_features)
        self.bn_target1 = nn.BatchNorm1d(num_features)
        self.bn_target2 = nn.BatchNorm1d(num_features)

        self.bn_target1.weight = self.bn_source.weight
        self.bn_target1.bias = self.bn_source.bias
        self.bn_target2.weight = self.bn_source.weight
        self.bn_target2.bias = self.bn_source.bias

        self.domain = "source"

    def forward(self, x):
        if self.domain == "source":
            return self.bn_source(x)

        if self.domain == "target1":
            return self.bn_target1(x)

        if self.domain == "target2":
            return self.bn_target2(x)

        raise ValueError(f"Unknown domain: {self.domain}")


class Gear1DCNN_MultiDomain(nn.Module):
    def __init__(self, num_classes=6, input_channels=1):
        super().__init__()

        self.layer1 = self._conv_block(
            input_channels,
            16,
            kernel_size=32,
            stride=2,
            padding=16,
        )
        self.layer2 = self._conv_block(16, 32, 3, 1, 1)
        self.layer3 = self._conv_block(32, 64, 3, 1, 1)
        self.layer4 = self._conv_block(64, 64, 3, 1, 1)
        self.layer5 = self._conv_block(64, 64, 3, 1, 1)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)

        self.fc = nn.Sequential(
            nn.Linear(64 * 4, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes),
        )

    @staticmethod
    def _conv_block(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            ),
            MultiDomainBatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def set_domain(self, domain):
        for module in self.modules():
            if isinstance(module, MultiDomainBatchNorm1d):
                module.domain = domain

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
