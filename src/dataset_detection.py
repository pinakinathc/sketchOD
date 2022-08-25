import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import selectivesearch

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(
            self.opts.data_dir, '256x256', 'sketch', 'tx_000000000000'
        ))
        if mode == 'train':
            self.all_categories = list(set(self.all_categories) - set(unseen_classes))
        else:
            self.all_categories = unseen_classes

        self.all_sketches_path = {}
        self.all_photos_path = []

        for category in self.all_categories:
            self.all_photos_path.extend(glob.glob(os.path.join(
                self.opts.data_dir, 'EXTEND_image_sketchy', category, '*.jpg'
            )))
            self.all_sketches_path[category] = glob.glob(os.path.join(
                self.opts.data_dir, '256x256', 'sketch', 'tx_000000000000', category, '*.png'
            ))

    def __len__(self):
        return len(self.all_photos_path)

    def __getitem__(self, index):
        filepath = self.all_photos_path[index]
        img_path = filepath

        filepath, filename = os.path.split(filepath)
        category = os.path.split(filepath)[-1]

        list_sk_path = []
        for cat in [category] + list(np.random.choice(self.all_categories, self.opts.nclass-1)):
            list_sk_path.append(np.random.choice(self.all_sketches_path[cat]))
        
        img_data = Image.open(img_path).convert('RGB')
        list_sk_data = [Image.open(sk_path).convert('RGB') for sk_path in list_sk_path]

        img_data = ImageOps.pad(img_data, size=(self.opts.max_size, self.opts.max_size))
        _, proposals = selectivesearch.selective_search(
            np.array(img_data), scale=self.opts.bbox_scale,
            sigma=self.opts.bbox_sigma, min_size=self.opts.bbox_min_size)
        proposals = torch.tensor([item['rect'] for item in proposals], dtype=torch.float)
        proposals[:, 2:] = proposals[:, 2:] + proposals[:, :2]

        list_sk_data = [ \
            ImageOps.pad(sk_data, size=(self.opts.max_size, self.opts.max_size)) \
            for sk_data in list_sk_data]
        
        img_tensor = self.transform(img_data) # 3 x H x W
        sk_tensor = torch.stack([self.transform(sk_data) for sk_data in list_sk_data]) # nclass x 3 x H x W

        target = torch.zeros(len(list_sk_data))
        target[0] = 1

        # shuffle class order
        rand_idx = torch.randperm(len(list_sk_data))
        list_sk_data = [list_sk_data[idx] for idx in rand_idx]
        sk_tensor = sk_tensor[rand_idx]
        target = target[rand_idx]

        if self.return_orig:
            return (img_tensor, sk_tensor, proposals, target, img_data, list_sk_data)
        else:
            return (img_tensor, sk_tensor, proposals, target)

    @staticmethod
    def collate_fn(batch):
        img_tensor = torch.stack([item[0] for item in batch])
        sk_tensor = torch.stack([item[1] for item in batch])
        proposals = [item[2] for item in batch]
        target = torch.stack([item[3] for item in batch])
        return img_tensor, sk_tensor, proposals, target

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm
    from PIL import ImageDraw
    from torch.utils.data import DataLoader

    dataset_transforms = Sketchy.data_transform(opts)

    dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    dataloader = DataLoader(
        dataset=dataset, batch_size=opts.batch_size, num_workers=opts.workers, collate_fn=Sketchy.collate_fn)

    idx = 0
    for data in tqdm.tqdm(dataloader):
        
        img_tensor, sk_tensor, proposals, target = data
        print (img_tensor.shape, sk_tensor.shape, len(proposals), target.shape)
        # print (proposals)
        continue

        (img_tensor, sk_tensor, proposals,
            target, img_data, list_sk_data) = data

        print (target, proposals.shape)

        # Draw boxes
        draw = ImageDraw.Draw(img_data)
        for bbox in proposals.numpy():
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], fill=None, outline='red', width=1)

        canvas = Image.new('RGB', (224*len(list_sk_data), 224))
        canvas.paste(img_data, (0, 0))
        offset = img_data.size[0]
        for sk_data in list_sk_data:
            canvas.paste(sk_data, (offset, 0))
            offset += sk_data.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1