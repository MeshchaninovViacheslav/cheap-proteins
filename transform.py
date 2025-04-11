import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

device = torch.device("cuda")
max_len = 254



def transform(data, pipeline):
    sequences = []
    encodings = []
    masks = []

    for batch in tqdm(DataLoader(data, batch_size=100)):
        seqs = [s[:max_len] for s in batch["sequence"]]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            emb, mask = pipeline(seqs)

        emb = emb.detach().cpu().tolist()
        mask = mask.detach().cpu().tolist()
        for s, e, m in zip(seqs, emb, mask):
            sequences.append(s)
            encodings.append(e)
            masks.append(m)
        
        torch.cuda.empty_cache()

    dt = Dataset.from_dict({
        "sequence": sequences,
        "encoding": encodings,
        "mask": masks,
    })
    return dt


parser = argparse.ArgumentParser()
parser.add_argument("--cheap_model", type=str)
parser.add_argument("--save_dir", type=str)

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

save_dir = f"{args.save_dir}/data/{model_name}"
print(save_dir)

dt_train = transform(load_from_disk("../DiMA/data/swissprot/train"), pipeline)
dt_train.save_to_disk(f"{save_dir}/train")

dt_test = transform(load_from_disk("../DiMA/data/swissprot/test"), pipeline)
dt_test.save_to_disk(f"{save_dir}/test")