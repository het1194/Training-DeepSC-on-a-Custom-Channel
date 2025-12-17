import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import EurDataset, collate_data
from transceiver import DeepSC
from utils import (
    BleuScore, SNR_to_noise, greedy_decode, SeqtoText,
    Channels
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# SHORT MARKOV FOR TESTING (compressed cycle)
# ------------------------------------------------------------
LOS_len = 5        # originally 100
SH_len  = 2        # originally 20
LOS2_len = 5       # newly added for LOS-after-shadow

STATE_LOS = { "dist": 550,  "K": 1000 }
STATE_SHADOW = { "dist": 2000, "K": 0 }

# TRAINING SNR RANGE
SNR_MIN = 5
SNR_MAX = 15

# ------------------------------------------------------------
# PERFORMANCE (short Markov cycle)
# ------------------------------------------------------------
def evaluate_markov(args, net, pad_idx, start_idx, end_idx, token_to_idx):

    bleu_1 = BleuScore(1, 0, 0, 0)
    test_set = EurDataset('test', data_dir=args.data_dir)

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        collate_fn=collate_data,
        shuffle=False
    )

    StoT = SeqtoText(token_to_idx, end_idx)

    net.eval()

    # buckets matching test cycle
    los1_pred, los1_true = [], []
    shadow_pred, shadow_true = [], []
    los2_pred, los2_true = [], []

    # State machine
    current_state = "LOS1"
    counter = 0

    total_batches = 0
    max_batches = LOS_len + SH_len + LOS2_len

    # ---- Run until we finish LOS1 → SHADOW → LOS2 ----
    with torch.no_grad():
        while total_batches < max_batches:

            for sents in test_loader:

                total_batches += 1
                counter += 1

                # ---- Markov switching ----
                if current_state == "LOS1" and counter > LOS_len:
                    current_state = "SHADOW"
                    counter = 1

                elif current_state == "SHADOW" and counter > SH_len:
                    current_state = "LOS2"
                    counter = 1

                # Stop when cycle is complete
                if total_batches > max_batches:
                    break

                # ---- Apply channel settings ----
                if current_state == "LOS1":
                    Channels._sat_dist_km = STATE_LOS["dist"]
                    Channels._sat_K_factor = STATE_LOS["K"]

                elif current_state == "SHADOW":
                    Channels._sat_dist_km = STATE_SHADOW["dist"]
                    Channels._sat_K_factor = STATE_SHADOW["K"]

                else:  # LOS2
                    Channels._sat_dist_km = STATE_LOS["dist"]
                    Channels._sat_K_factor = STATE_LOS["K"]

                # ---- Sample SNR like training ----
                noise_std = SNR_to_noise(np.random.uniform(SNR_MIN, SNR_MAX))

                # ---- Decode ----
                sents = sents.to(device)
                decoded = greedy_decode(
                    net, sents, noise_std, args.MAX_LENGTH,
                    pad_idx, start_idx, args.channel
                )

                decoded = decoded.cpu().numpy().tolist()
                pred_text = list(map(StoT.sequence_to_text, decoded))

                target_np = sents.cpu().numpy().tolist()
                tgt_text = list(map(StoT.sequence_to_text, target_np))

                # ---- Bucket assignment ----
                if current_state == "LOS1":
                    los1_pred += pred_text
                    los1_true += tgt_text

                elif current_state == "SHADOW":
                    shadow_pred += pred_text
                    shadow_true += tgt_text

                else:  # LOS2
                    los2_pred += pred_text
                    los2_true += tgt_text

    # ---------------------------------------------------------
    # Compute BLEU for each region
    # ---------------------------------------------------------
    def safe_bleu(true_list, pred_list):
        if len(true_list) == 0:
            return float("nan")
        score = bleu_1.compute_blue_score(true_list, pred_list)
        score = np.array(score).flatten()[0]
        return score

    bleu_los1 = safe_bleu(los1_true, los1_pred)
    bleu_shadow = safe_bleu(shadow_true, shadow_pred)
    bleu_los2 = safe_bleu(los2_true, los2_pred)

    return bleu_los1, bleu_shadow, bleu_los2

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.', type=str)
    parser.add_argument('--vocab-file', default='./snli_vocab.json', type=str)
    parser.add_argument('--checkpoint-path', default='./checkpoint_500.pth', type=str)
    parser.add_argument('--channel', default='LEO_LMS', type=str)
    parser.add_argument('--MAX-LENGTH', default=30, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--d-model', default=128, type=int)
    parser.add_argument('--dff', default=512, type=int)
    parser.add_argument('--num-layers', default=4, type=int)
    parser.add_argument('--num-heads', default=8, type=int)
    args = parser.parse_args()

    # load vocab
    vocab = json.load(open(os.path.join(args.data_dir, args.vocab_file), 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)

    pad_idx   = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx   = token_to_idx["<END>"]

    print("Initializing DeepSC...")
    net = DeepSC(
        args.num_layers, num_vocab, num_vocab,
        num_vocab, num_vocab,
        args.d_model, args.num_heads, args.dff, 0.1
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint)

    print("\n===== Evaluating Short Markov Simulation =====\n")

    bleu_los1, bleu_shadow, bleu_los2 = evaluate_markov(
        args, net, pad_idx, start_idx, end_idx, token_to_idx
    )

    print("\n=========== FINAL RESULTS (Short Markov) ===========")
    print(f"BLEU (LOS before shadow):   {bleu_los1:.4f}")
    print(f"BLEU (Shadow region):       {bleu_shadow:.4f}")
    print(f"BLEU (LOS after shadow):    {bleu_los2:.4f}")
    print("====================================================\n")
