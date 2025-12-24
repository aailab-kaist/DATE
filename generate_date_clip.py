# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionDATECLIPPipeline, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
import torch
import os
import pandas as pd
import argparse
from torch_utils import distributed as dist
import numpy as np
import tqdm
import gc
from transformers import CLIPModel, CLIPTokenizer

parser = argparse.ArgumentParser(description='Generate images with stable diffusion')
parser.add_argument('--steps', type=int, default=50, help='number of inference steps during sampling')
parser.add_argument('--generate_seed', type=int, default=6)
parser.add_argument('--w', type=float, default=7.5)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--max_cnt', type=int, default=5000, help='number of maximum generated samples')
parser.add_argument('--save_path', type=str, default='./generated_images')
parser.add_argument('--scheduler', type=str, default='DDIM')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--text_path', type=str, default='subset.csv')
parser.add_argument('--num_skip', type=int, default=0)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--ver_sd', type=str, default='1.5')

parser.add_argument('--num_upt_prompt', type=int, default=1)
parser.add_argument('--lr_upt_prompt', type=float, default=0.1)
parser.add_argument('--skip_freq', type=int, default=1)
parser.add_argument('--skip_itrs', type=str, default='')
parser.add_argument('--skip_intervals', type=str, default='')
parser.add_argument('--weight_prior', type=float, default=0.)
parser.add_argument('--init_org', action='store_true', default=False)
parser.add_argument('--clip_type', type=str, default='clip-vit-large-patch14')
parser.add_argument('--fp16_clip', action='store_true', default=False)

args = parser.parse_args()

dist.init()
dist.print0('Args:')
for k, v in sorted(vars(args).items()):
    dist.print0('\t{}: {}'.format(k, v))

df = pd.read_csv(args.text_path)
all_text = list(df['caption'])
all_text = all_text[: args.max_cnt]

num_batches = ((len(all_text) - 1) // (args.bs * dist.get_world_size()) + 1) * dist.get_world_size()
all_batches = np.array_split(np.array(all_text), num_batches)
rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

index_list = np.arange(len(all_text))
all_batches_index = np.array_split(index_list, num_batches)
rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]

if args.ver_sd == '1.5':
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
elif args.ver_sd == '1.4':
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
else:
    raise NotImplementedError(f"No defined pretrained model name {args.ver_sd}")
if args.fp16:
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32
pipe = StableDiffusionDATECLIPPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=weight_dtype)
dist.print0("default scheduler config:")
dist.print0(pipe.scheduler.config)
pipe = pipe.to("cuda")

clip_tokenizer = CLIPTokenizer.from_pretrained(f"openai/{args.clip_type}")
if args.fp16_clip:
    clip_model = CLIPModel.from_pretrained(f"openai/{args.clip_type}", torch_dtype=torch.float16)
else:
    clip_model = CLIPModel.from_pretrained(f"openai/{args.clip_type}")
clip_model = clip_model.to("cuda")

if args.scheduler == 'DDPM':
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == 'DDIM':
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == 'SDE':
    pipe.scheduler = SDEScheduler.from_config(pipe.scheduler.config)
elif args.scheduler == 'ODE':
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.use_karras_sigmas = False
else:
    raise NotImplementedError

generator = torch.Generator(device="cuda").manual_seed(args.generate_seed)

save_name = f'DATE_CLIP_scheduler_{args.scheduler}_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}'
if args.name is not None:
    save_name += f'_name_{args.name}'
save_dir = os.path.join(args.save_path, save_name)
if dist.get_rank() == 0 and not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
if dist.get_rank() == 0 and not os.path.exists(save_dir):
    os.mkdir(save_dir)
dist.print0("save images to {}".format(save_dir))

if args.skip_itrs != '':
    skip_itrs = list(map(int, args.skip_itrs.split('-')))
elif args.skip_intervals != '':
    skip_itrs = list(range(*map(int, args.skip_intervals.split('-'))))
else:
    skip_itrs = None

for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    torch.distributed.barrier()
    text = list(mini_batch)

    if rank_batches_index[cnt][-1] < args.num_skip:
        skip_images = True
    else:
        skip_images = False

    SD_out = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, dist=dist, skip_images=skip_images,
                  clip_model=clip_model, num_upt_prompt=args.num_upt_prompt, lr_upt_prompt=args.lr_upt_prompt,
                  skip_freq=args.skip_freq, weight_prior=args.weight_prior, skip_itrs=skip_itrs,
                  init_org=args.init_org, mixed_precision=(args.fp16 and not args.fp16_clip),)

    if not skip_images:
        image = SD_out.images
        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            image[text_idx].save(os.path.join(save_dir, f'{global_idx}.png'))
    gc.collect()
    torch.cuda.empty_cache()

torch.distributed.barrier()
if dist.get_rank() == 0:
    d = {'caption': all_text}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(save_dir, 'subset.csv'))