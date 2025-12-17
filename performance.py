# -*- coding: utf-8 -*-
"""
Evaluation & BLEU testing script for DeepSC + Satellite Channel
(MODIFIED FOR FULL TEST MATRIX)

This script will loop through every combination of specified
Distances, K-Factors, and SNRs.
"""

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
# ARGUMENT PARSER
# ------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--data-dir', default='.', type=str,
                    help="Directory containing test.pkl and vocab file")
parser.add_argument('--vocab-file', default='./snli_vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='./checkpoints/deepsc-Generalist/checkpoint_500.pth', type=str,
                    help="Path to the trained 'Generalist' .pth model file")

parser.add_argument('--channel', default='LEO_LMS', type=str,
                    help='This should always be LEO_LMS for the new unified channel.')

# --- Base satellite physics params (used as defaults) ---
parser.add_argument('--sat-ref-dist-km', type=float, default=550.0,
                    help="Reference distance (must match training)")
parser.add_argument('--sat-freq-hz', type=float, default=2.4e9)
parser.add_argument('--sat-dtx-m', type=float, default=0.6)
parser.add_argument('--sat-drx-m', type=float, default=0.6)
parser.add_argument('--sat-eff', type=float, default=0.6)
parser.add_argument('--sat-extra-db', type=float, default=1.0)

# Model params (must match the trained model)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)

parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=1, type=int,
                    help="Number of times to run each test (1 is fine)")

# ------------------------------------------------------------
# PERFORMANCE FUNCTION (Unchanged)
# This function runs one full SNR sweep for a given
# set of global channel parameters.
# ------------------------------------------------------------
def performance(args, SNR_list, net, pad_idx, start_idx, end_idx, token_to_idx):

    bleu_1gram = BleuScore(1, 0, 0, 0)

    test_set = EurDataset('test', data_dir=args.data_dir)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_data
    )

    StoT = SeqtoText(token_to_idx, end_idx)

    net.eval()
    final_scores = []

    with torch.no_grad():
        for ep in range(args.epochs):
            Tx_all = []
            Rx_all = []

            for snr in tqdm(SNR_list, desc=f"SNR Loop (K={Channels._sat_K_factor}, Dist={Channels._sat_dist_km}km)"):

                Tx_sentences = []
                Rx_sentences = []
                noise_std = SNR_to_noise(snr)

                for sents in test_loader:
                    sents = sents.to(device)
                    target_sents = sents

                    decoded = greedy_decode(
                        net, sents, noise_std, args.MAX_LENGTH,
                        pad_idx, start_idx, args.channel
                    )

                    decoded = decoded.cpu().numpy().tolist()
                    decoded_text = list(map(StoT.sequence_to_text, decoded))
                    Tx_sentences += decoded_text

                    target_np = target_sents.cpu().numpy().tolist()
                    target_text = list(map(StoT.sequence_to_text, target_np))
                    Rx_sentences += target_text

                Tx_all.append(Tx_sentences)
                Rx_all.append(Rx_sentences)

            # Compute BLEU score
            bleu_scores = []
            for pred_list, real_list in zip(Tx_all, Rx_all):
                bleu_scores.append(
                    bleu_1gram.compute_blue_score(real_list, pred_list)
                )

            bleu_scores = np.array(bleu_scores)
            bleu_scores = np.mean(bleu_scores, axis=1)
            final_scores.append(bleu_scores)

    return np.mean(np.array(final_scores), axis=0)

# ------------------------------------------------------------
# MAIN (MODIFIED FOR TEST MATRIX)
# ------------------------------------------------------------
if __name__ == '__main__':

    args = parser.parse_args()

    # --------------------------------------------------------
    # BIND CONSTANT SATELLITE PARAMETERS
    # These parameters do not change during the test loops.
    # --------------------------------------------------------
    Channels._sat_ref_dist_km = args.sat_ref_dist_km
    Channels._sat_freq_hz = args.sat_freq_hz
    Channels._sat_d_tx_m = args.sat_dtx_m
    Channels._sat_d_rx_m = args.sat_drx_m
    Channels._sat_eff = args.sat_eff
    Channels._sat_extra_db = args.sat_extra_db

    # --------------------------------------------------------
    # LOAD VOCAB
    # --------------------------------------------------------
    vocab_path = os.path.join(args.data_dir, args.vocab_file)
    print(f"Loading vocab from: {vocab_path}\n")

    vocab = json.load(open(vocab_path, "rb"))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)

    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    # --------------------------------------------------------
    # CREATE AND LOAD MODEL (ONCE)
    # --------------------------------------------------------
    print("Initializing DeepSC model...\n")
    net = DeepSC(
        args.num_layers, num_vocab, num_vocab,
        num_vocab, num_vocab,
        args.d_model, args.num_heads,
        args.dff, 0.1
    ).to(device)

    print(f"Loading model checkpoint: {args.checkpoint_path}\n")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        print("Please specify the correct path using --checkpoint-path")
    else:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully.\n")

        # --------------------------------------------------------
        # ***** TEST MATRIX DEFINITION *****
        # --------------------------------------------------------
        DISTANCES_TO_TEST = [550, 1000, 1500, 2000, 2500, 3000]
        K_FACTORS_TO_TEST = [0, 10, 15, 500, 1000]
        SNR_LIST_TO_TEST = [0, 3, 6, 9, 12, 15, 18]
        # --------------------------------------------------------

        print("--- Starting Full Test Matrix Evaluation ---")
        print(f"Model: {args.checkpoint_path}")
        print(f"Distances: {DISTANCES_TO_TEST}")
        print(f"K-Factors: {K_FACTORS_TO_TEST} (0=Rayleigh, 10/15=Rician, >500=AWGN/LOS)")
        print(f"SNRs: {SNR_LIST_TO_TEST}")
        print("=====================================================================")

        # Create nested loops to iterate through all combinations
        for dist in DISTANCES_TO_TEST:
            for k_factor in K_FACTORS_TO_TEST:

                # 1. SET the global "knobs" for this specific test run
                Channels._sat_dist_km = dist
                Channels._sat_K_factor = k_factor

                # 2. RUN the performance test for this one combination
                # The performance() function will handle the SNR_LIST loop
                bleu = performance(
                    args, SNR_LIST_TO_TEST, net, pad_idx, 
                    start_idx, end_idx, token_to_idx
                )

                # 3. PRINT the result for this combination
                print("\n" + "="*50)
                print(f"     RESULTS FOR: Dist={dist} km, K-Factor={k_factor}")
                print("="*50)
                print(f"SNRs        : {SNR_LIST_TO_TEST}")
                print(f"BLEU Scores : {bleu}")
                print("="*50 + "\n")

        print("=====================================================================")
        print("               Full Test Matrix Evaluation Complete                  ")
        print("=====================================================================")