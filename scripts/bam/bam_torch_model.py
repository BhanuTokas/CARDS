"""PyTorch port of BAM's obj and scene models (TF official ResNet-50 v1.5, 10 classes).

Same input as the TF graph: NCHW float, raw 0-255 RGB minus CHANNEL_MEANS. The
weights come from export_bam_weights.py. Two details differ from torchvision's
ResNet-50: TF's SAME max-pool pads only the bottom and right, and the
batch-norm epsilon is 1.001e-5. With TF32 off (load() turns it off), logits
match the TF model to about 3e-5.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

CHANNEL_MEANS = (123.68, 116.78, 103.94)
BN_EPS = 1.001e-5


class BamResNet50(torchvision.models.ResNet):
    def __init__(self, num_classes: int = 10):
        super().__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                         norm_layer=lambda c: torch.nn.BatchNorm2d(c, eps=BN_EPS))

    def features(self, x: torch.Tensor, upto: str = "layer4") -> torch.Tensor:
        """Output of TF's block_layer1..4 (layer1..layer4)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float("-inf")), 3, 2)
        for name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(self, name)(x)
            if name == upto:
                return x
        raise ValueError(upto)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).mean((2, 3)))


def load(bam_root: Path, model: str) -> BamResNet50:
    """The obj or scene model in eval mode, from <bam-root>/models/<model>_weights.npz."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    w = np.load(Path(bam_root) / "models" / f"{model}_weights.npz")
    convs = sorted({k.split("/")[1] for k in w.files if k.split("/")[1].startswith("conv2d")},
                   key=lambda n: int(n.split("_")[1]) if "_" in n else 0)
    net = BamResNet50()
    order = [(net.conv1, net.bn1)]
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        for block in layer:
            if block.downsample is not None:  # TF creates the projection shortcut first
                order.append((block.downsample[0], block.downsample[1]))
            order += [(block.conv1, block.bn1), (block.conv2, block.bn2), (block.conv3, block.bn3)]
    assert len(order) == len(convs) == 53

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a))

    for (conv, bn), name in zip(order, convs):
        kernel = w[f"resnet_model/{name}/kernel"]  # (kh, kw, in, out)
        assert kernel.shape == (*conv.kernel_size, conv.in_channels, conv.out_channels), name
        conv.weight.data = t(kernel.transpose(3, 2, 0, 1))
        prefix = "resnet_model/" + name.replace("conv2d", "batch_normalization")
        bn.weight.data, bn.bias.data = t(w[f"{prefix}/gamma"]), t(w[f"{prefix}/beta"])
        bn.running_mean.data, bn.running_var.data = t(w[f"{prefix}/moving_mean"]), t(w[f"{prefix}/moving_variance"])
    net.fc.weight.data = t(w["resnet_model/dense/kernel"].T)
    net.fc.bias.data = t(w["resnet_model/dense/bias"])
    return net.eval()
