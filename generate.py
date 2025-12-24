# make sure you're logged in with \`huggingface-cli login\`
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, SDEScheduler, EulerDiscreteScheduler
import torch
import os
import pandas as pd
import argparse
from torch_utils import distributed as dist
import numpy as np
import tqdm
import gc

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
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=weight_dtype)
dist.print0("default scheduler config:")
dist.print0(pipe.scheduler.config)
pipe = pipe.to("cuda")

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

save_name = f'scheduler_{args.scheduler}_steps_{args.steps}_w_{args.w}_seed_{args.generate_seed}'
if args.name is not None:
    save_name += f'_name_{args.name}'
save_dir = os.path.join(args.save_path, save_name)
if dist.get_rank() == 0 and not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
if dist.get_rank() == 0 and not os.path.exists(save_dir):
    os.mkdir(save_dir)
dist.print0("save images to {}".format(save_dir))

for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
    torch.distributed.barrier()
    text = list(mini_batch)

    if rank_batches_index[cnt][-1] < args.num_skip:
        skip_images = True
    else:
        skip_images = False

    SD_out = pipe(text, generator=generator, num_inference_steps=args.steps, guidance_scale=args.w, dist=dist, skip_images=skip_images)

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