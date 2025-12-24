from PIL import Image
import open_clip
import argparse
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import torch

class text_image_pair(torch.utils.data.Dataset):
    def __init__(self, dir_path, csv_path, resolution=1):
        self.dir_path = dir_path
        df = pd.read_csv(csv_path)
        self.text_description = df['caption']
        if resolution != 1:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k', force_image_size=resolution)
        else:
            _, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')

    def __len__(self):
        return len(self.text_description)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir_path, f'{idx}.png')
        raw_image = Image.open(img_path)
        image = self.preprocess(raw_image).squeeze().float()
        text = self.text_description[idx]
        return image, text

parser = argparse.ArgumentParser(description='Evaluation of CLIP Score')
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--text_path', type=str, default='./generated_images/subset.csv')
parser.add_argument('--img_path', type=str, default='./generated_images/subset')
parser.add_argument('--num_eval', type=int, default=5000)
args = parser.parse_args()

# define dataset / data_loader
text2img_dataset = text_image_pair(dir_path=args.img_path, csv_path=args.text_path)
text2img_loader = torch.utils.data.DataLoader(dataset=text2img_dataset, batch_size=args.bs, shuffle=False)
print("total length:", len(text2img_dataset))

model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
model = model.cuda().eval()
tokenizer = open_clip.get_tokenizer('ViT-g-14')

cnt = 0.
total_clip_score = 0.

scores = np.zeros((args.num_eval,), dtype=np.float32)
with torch.no_grad(), torch.cuda.amp.autocast():
    for idx, (image, text) in tqdm(enumerate(text2img_loader)):
        image = image.cuda().float()
        text = list(text)
        text = tokenizer(text).cuda()
        image_features = model.encode_image(image).float()
        text_features = model.encode_text(text).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        total_clip_score += (image_features * text_features).sum()
        scores[args.bs*idx:args.bs*idx+len(image)] = (image_features * text_features).sum(dim=-1).cpu().numpy()

        cnt += len(image)
        if cnt >= args.num_eval:
            print(f'Evaluation complete! NUM EVAL: {cnt}')
            break

np.savez(f'{args.img_path}/clip_scores.npz', scores=scores)
print("Average CLIP score:", total_clip_score.item() / cnt)