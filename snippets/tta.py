from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, FiveCrop, Lambda
import torchvision.transforms.functional as TF

ds_val = None
model = None
img_size = 456

model.eval()

labels = []
preds = []
for i in range(len(ds_val)):
    img, label = ds_val[i]
    labels.append(label)

    aug_transforms = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.hflip(TF.vflip(x)),
    ]

    # transform = Compose([
    #     Resize(img_size + 32, interpolation=Image.BICUBIC),
    #     CenterCrop(img_size),
    #     ToTensor(),
    #     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    transform = Compose([
        Resize(img_size + 32, interpolation=Image.BICUBIC),
        FiveCrop(img_size),
        Lambda(lambda xs: [ToTensor()(x) for x in xs]),
        Lambda(lambda xs: [Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x) for x in xs]),
        Lambda(lambda xs: torch.stack(xs, dim=0)),
    ])

    img_augs = []
    for t in aug_transforms:
        img_aug = t(img)
        img_aug = transform(img_aug)
        img_augs.append(img_aug)

    x = torch.stack(img_augs, dim=0)
    if x.ndim == 5:
        x = x.view(-1, x.size()[2:])
    x = x.cuda()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    pred = probs.mean(dim=0).argmax().item()
    preds.append(pred)


