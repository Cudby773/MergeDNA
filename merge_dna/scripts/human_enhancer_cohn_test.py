import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from merge_dna.models import MergeDNAModel, MergeDNAEncoderModel, LatentClassifier
from merge_dna.data import human_enhancers_cohn_train_loader, human_enhancers_cohn_test_loader  
from merge_dna.training.trainer import load_checkpoint, ensure_mask_token_in_embedding
from merge_dna.scripts.pretrain import make_model_small 


def evaluate(model: LatentClassifier, dataloader, device):
    model.eval()
    preds = []
    labels = []
    logits_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            inputs, labs = batch

            inputs = inputs.to(device)
            labs = labs.to(device)
            
            out = model(inputs)  
           
            if isinstance(out, tuple) or isinstance(out, list):
                logits = out[0]
            else:
                logits = out

            if logits.dim() == 1 or logits.size(1) == 1:  # binary single-logit
                prob = torch.sigmoid(logits.view(-1))
                pred = (prob > 0.5).long().cpu().numpy()
            else:
                pred = torch.argmax(logits, dim=1).cpu().numpy()

            preds.append(pred)
            labels.append(labs.cpu().numpy())
            logits_list.append(logits.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    logits = np.concatenate(logits_list)

    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro" if len(np.unique(labels)) > 2 else "binary"))
    try:
        mcc = float(matthews_corrcoef(labels, preds))
    except Exception:
        mcc = float("nan")

    return {"accuracy": acc, "f1": f1, "mcc": mcc, "n": len(labels)}


def train_one_epoch(model: nn.Module, opt, dataloader, device):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        inputs, labels = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        logits: torch.Tensor = model(inputs)
        if logits.dim() == 2 and logits.size(1) > 1:
            loss = F.cross_entropy(logits, labels.long())
        else:
            loss = F.binary_cross_entropy_with_logits(logits.view(labels.shape), labels.float())

        loss.backward()
        opt.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        pbar.set_postfix({"loss": total_loss / n})
    return total_loss / n


def save_ckpt(out_dir, model, optimizer, step, epoch, config=None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "epoch": epoch,
        "config": config or {},
    }
    fname = os.path.join(out_dir, f"finetune_step{step}.pt")
    torch.save(ckpt, fname)
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["test", "finetune"], default="test")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    device = torch.device(args.device)

    train_dl, test_dl = human_enhancers_cohn_train_loader, human_enhancers_cohn_test_loader
    vocab_size = 6
    model: MergeDNAModel = make_model_small(vocab_size=vocab_size).to(device)  
    ensure_mask_token_in_embedding(model.local_encoder, vocab_size)
    
    step, start_epoch, ckpt_config = load_checkpoint(args.ckpt_path, model, None, device=device)
    print(f"Loaded checkpoint at epoch {start_epoch}, step {step}, config={ckpt_config}")
    
    encoder_model = MergeDNAEncoderModel(model.local_encoder, model.latent_encoder)
    classifier = LatentClassifier(encoder_model, num_classes=1, use_cls_token=True)
    classifier.to(device)

    if args.mode == 'test':
        metrics = evaluate(classifier, test_dl, device)
        print("TEST RESULTS:", metrics)
        return
    
    for p in classifier.encoder_model.local_encoder.parameters():
        p.requires_grad = False
    for p in classifier.encoder_model.latent_encoder.parameters():
        p.requires_grad = False
        
    classifier.eval()
    with torch.no_grad():
        batch0 = next(iter(train_dl))
        inputs, _ = batch0
        inputs = inputs.to(device)
        _ = classifier(inputs)   # lazy inits and registers the head on the correct device
        
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        print(f"[INFO] Head training epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(classifier, optimizer, train_dl, device)
        print(f"[INFO] Head epoch {epoch} train_loss: {train_loss:.4f}")
    final_metrics = evaluate(classifier, test_dl, device)
    print(final_metrics)


if __name__ == "__main__":
    main()
