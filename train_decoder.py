import os
import torch
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from datasets import load_from_disk
from cheap.pretrained import CHEAP_shorten_1_dim_64

from config import create_config
from encoders import Decoder, ESM2EncoderModel
from utils import load_fasta_file
from diffusion_utils.dynamic import DynamicSDE
from encoders.enc_normalizer import EncNormalizer

from utils.util import CheapDataset


def reconstruction_loss(target, prediction_scores, mask):
    if mask is None:
        return cross_entropy(
            input=prediction_scores.view(-1, prediction_scores.shape[-1]),
            target=target.view(-1),
        )

    ce_losses = cross_entropy(
        input=prediction_scores.view(-1, prediction_scores.shape[-1]),
        target=target.view(-1),
        reduce=False,
    )
    ce_losses = ce_losses * mask.reshape(-1)
    ce_loss = torch.sum(ce_losses) / torch.sum(mask)
    return ce_loss


def get_loaders(config, batch_size):
    train_dataset = load_from_disk(config.data.train_dataset_path)
    train_dataset = CheapDataset(config, train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True
    )

    valid_dataset = load_from_disk(config.data.test_dataset_path)
    valid_dataset = CheapDataset(config, valid_dataset)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=20,
        pin_memory=True
    )

    return train_loader, valid_loader


def loss_step(sequences, mask, latent, pipeline, decoder, enc_normalizer, config, eval=False):
    tokenized_X = encoder.tokenizer(
            sequences, 
            return_attention_mask=True, 
            return_tensors="pt", 
            truncation=True,                       
            padding=True, 
            max_length=config.data.max_sequence_len,
            return_special_tokens_mask=True,
        )
    tokenized_X = tokenized_X.to(f"cuda:0")

    # Normalization
    latent = enc_normalizer.normalize(latent)

    # Noising
    if not eval:
        dynamic = DynamicSDE(config=config)
        if config.model.decoder_mode == "noisy":
            T = 1.0
        else:
            T = 0.25
        eps = 0.001
        t = torch.cuda.FloatTensor(latent.shape[0]).uniform_() * (T - eps) + eps
        x_t = dynamic.marginal(latent, t)["x_t"]
        latent = x_t
    
    # Get Latent
    latent = enc_normalizer.denormalize(latent)
    latent = pipeline.decode(latent, mask)

    targets = tokenized_X["input_ids"]
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = decoder(latent, mask)

    # check shape
    print(targets.shape, logits.shape)
    loss = reconstruction_loss(targets, logits, mask=mask)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def train(config, pipeline, decoder, enc_normalizer, exp_name):
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")
    
    batch_size = 256
    dim = int(config.model.encoder_name.split("_")[-1])

    train_loader, valid_loader = get_loaders(
        config=config,
        batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=1e-4,
        weight_decay=0.001,
        betas=(0.9, 0.98),
    )

    step = 0
    epochs = 1
    for _ in range(epochs):
        decoder.train()

        for batch in tqdm(train_loader):
            latent = batch["encoding"].reshape(batch_size, -1, train_dataset.dim).cuda()
            mask = batch["mask"].cuda()
            sequence = batch["sequence"]

            loss, acc = loss_step(
                sequences=sequences,
                mask=mask,
                latent=latent,
                pipeline=pipeline,
                decoder=decoder,
                enc_normalizer=enc_normalizer,
                config=config,
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1

            
        decoder.eval()
        for sequences in tqdm(valid_loader):
            with torch.no_grad():
                loss, acc = loss_step(
                    sequences=sequences,
                    encoder=encoder,
                    decoder=decoder,
                    config=config,
                    eval=True
                )

        wandb.log({f'valid loss': loss.item()}, step=step)
        wandb.log({f'valid accuracy': acc.item()}, step=step)

    os.makedirs(config.training.checkpoints_folder, exist_ok=True)
    
    decoder.eval()
    torch.save(
        {
            "decoder": decoder.state_dict(),
        },
        config.model.decoder_path
    )
    print(f"Save model to: {config.model.decoder_path}")


if __name__ == "__main__":
    config = create_config()
    
    pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True)
    enc_normalizer = EncNormalizer(
        enc_mean_path=config.data.enc_mean,
        enc_std_path=config.data.enc_std,
    )
    decoder = Decoder(config=config).cuda().train()

    exp_name = config.model.decoder_path.split("/")[-1].replace(".pth", "")
    wandb.init(project=config.project_name, name=exp_name, mode="online")
    train(config, pipeline, decoder, enc_normalizer, exp_name=exp_name)
