import os
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from cheap.proteins import LatentToSequence

device = torch.device("cuda")
max_len = 254



def decode(encodings, masks, pipeline, dim):
    pipe_decode = LatentToSequence()

    sequences = []
    batch_size = 10

    for ind in tqdm(range(0, len(encodings), batch_size)):
        enc = torch.Tensor(encodings[ind: ind + batch_size]).cuda()
        enc = enc.reshape(enc.shape[0], -1, dim)
        mask = torch.Tensor(masks[ind: ind + batch_size]).cuda()

        seqs = pipeline.decode(enc, mask)
        seqs = pipe_decode.to_sequence(seqs, mask)[2]

        for s, m in zip(seqs, mask):
            ind = int(sum(m).item())
            sequences.append(s[:ind])
        
        torch.cuda.empty_cache()

    return sequences


parser = argparse.ArgumentParser()
parser.add_argument("--cheap_model", type=str)
parser.add_argument("--path", type=str)
parser.add_argument("--output_path", type=str)

args = parser.parse_args()



if args.cheap_model == "CHEAP_shorten_1_dim_512":
    from cheap.pretrained import CHEAP_shorten_1_dim_512
    pipeline = CHEAP_shorten_1_dim_512(return_pipeline=True)

if args.cheap_model == "CHEAP_shorten_2_dim_512":
    from cheap.pretrained import CHEAP_shorten_2_dim_512
    pipeline = CHEAP_shorten_2_dim_512(return_pipeline=True)



if args.cheap_model == "CHEAP_shorten_1_dim_256":
    from cheap.pretrained import CHEAP_shorten_1_dim_256
    pipeline = CHEAP_shorten_1_dim_256(return_pipeline=True)

if args.cheap_model == "CHEAP_shorten_2_dim_256":
    from cheap.pretrained import CHEAP_shorten_2_dim_256
    pipeline = CHEAP_shorten_2_dim_256(return_pipeline=True)



if args.cheap_model == "CHEAP_shorten_1_dim_128":
    from cheap.pretrained import CHEAP_shorten_1_dim_128
    pipeline = CHEAP_shorten_1_dim_128(return_pipeline=True)

if args.cheap_model == "CHEAP_shorten_2_dim_128":
    from cheap.pretrained import CHEAP_shorten_2_dim_128
    pipeline = CHEAP_shorten_2_dim_128(return_pipeline=True)



if args.cheap_model == "CHEAP_shorten_1_dim_64":
    from cheap.pretrained import CHEAP_shorten_1_dim_64
    pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True)

if args.cheap_model == "CHEAP_shorten_2_dim_64":
    from cheap.pretrained import CHEAP_shorten_2_dim_64
    pipeline = CHEAP_shorten_2_dim_64(return_pipeline=True)


model_name = args.cheap_model
print(model_name)

dim = int(model_name.split("_")[-1])

data = json.load(open(args.path, "r"))

encodings = [d[0] for d in data]
masks = [d[1] for d in data]

sequences = decode(encodings, masks, pipeline, dim)

print(f"Total length: {len(sequences)}")

dir_ = "."
for p in args.output_path.split("/")[:-1]:
    dir_ += f"/{p}"
    os.makedirs(dir_, exist_ok=True)

json.dump(sequences, open(args.output_path, "w"), indent=4)

# CUDA_VISIBLE_DEVICES=4 python decode.py \
# --cheap_model="CHEAP_shorten_1_dim_64" \
# --path="../DiMA/generated_seqs/DiMA2.0-CHEAP_shorten_1_dim_64-CHEAP_shorten_1_dim_64-64-320-updated/100-N=250-len=10-t_min=0.05.json" \
# --output_path="generated_seqs/DiMA2.0-CHEAP_shorten_1_dim_64-CHEAP_shorten_1_dim_64-64-320-updated/100-N=250-len=10-t_min=0.05.json" 