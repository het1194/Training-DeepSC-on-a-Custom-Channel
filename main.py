# -*- coding: utf-8 -*-
"""
Main training script for DeepSC + Satellite channel
(MODIFIED for 2-State Markov Model)

This version trains a model on a simplified 2-state Markov chain,
simulating transitions between a "Good" (LOS) state and a "Bad" (Shadowed) state.
This is a more realistic training method than pure randomization.
"""

import os
import argparse
import time
import json
import torch
import random  # Still used, but less
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step
from dataset import EurDataset, collate_data
from transceiver import DeepSC  # Assumes transceiver.py is present
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Channels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# ARGUMENT PARSER
# ------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--vocab-file', default='./snli_vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Markov', type=str,
                    help="Directory to save the new Markov model")

parser.add_argument('--channel', default='LEO_LMS', type=str,
                    help='Only LEO_LMS is supported.')

# Satellite parameters (used to define the states)
parser.add_argument('--sat-dist-km', type=float, default=2000.0,
                    help="Max distance (for 'Bad' state)")
parser.add_argument('--sat-ref-dist-km', type=float, default=550.0,
                    help="Reference distance (for 'Good' state)")
parser.add_argument('--sat-K-factor', type=float, default=15.0,
                    help='Default Rician K-factor for validation.')
parser.add_argument('--sat-freq-hz', type=float, default=2.4e9)
parser.add_argument('--sat-dtx-m', type=float, default=0.6)
parser.add_argument('--sat-drx-m', type=float, default=0.6)
parser.add_argument('--sat-eff', type=float, default=0.6)
parser.add_argument('--sat-extra-db', type=float, default=1.0)

# Model parameters
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)

# Training
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--resume-from', type=str, default=None)

# ------------------------------------------------------------
# MARKOV MODEL DEFINITION
# (This is the new, simple 2-state logic)
# ------------------------------------------------------------

# State 1: "Good State" [cite: 47] - Clear LOS, low FSPL
# (Corresponds to State 1 in the paper [cite: 28])
STATE_LOS = {
    "K_factor": 1000.0,  # AWGN/LOS
    "dist_km": 550.0,    # Baseline distance
    "name": "LOS/AWGN"
}

# State 2: "Bad State" [cite: 47] - Shadowed, high FSPL
# (Corresponds to State 3 in the paper [cite: 45])
STATE_SHADOWED = {
    "K_factor": 0.0,     # Rayleigh
    "dist_km": 2000.0,   # Max distance
    "name": "Shadowed/Rayleigh"
}

# Define how long to stay in each state (in batches)
# This simulates the "correlation length" or "state frame" [cite: 228, 229, 668]
STATE_LENGTHS = {
    "LOS_len": 100,  # Stay in good state for 100 batches
    "SH_len": 20     # Stay in bad state for 20 batches
}

# Global variables to track our position in the Markov chain
# (This is a simplified implementation of a Markov chain [cite: 105, 121, 165])
state_counter = 0
current_state = STATE_LOS  # Start in the "Good" state


# ------------------------------------------------------------
# VALIDATION (Unchanged)
# ------------------------------------------------------------
def validate(epoch_print, args, net, pad_idx, criterion):
    """
    Validates on the DEFAULT channel specified by the args (e.g., K=15, Dist=550).
    Validation is kept simple for a consistent benchmark.
    """
    test_eur = EurDataset('test')
    test_loader = DataLoader(test_eur, batch_size=args.batch_size,
                             num_workers=0, pin_memory=True,
                             collate_fn=collate_data)

    net.eval()
    total = 0
    pbar = tqdm(test_loader)
    
    # Set validation parameters (uses the defaults from args)
    Channels._sat_dist_km = args.sat_ref_dist_km 
    Channels._sat_K_factor = args.sat_K_factor   

    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            val_noise_std = SNR_to_noise(10.0) 
            loss = val_step(net, sents, sents, val_noise_std, pad_idx, criterion, args.channel)
            total += loss
            pbar.set_description(f"Epoch: {epoch_print}; Type: VAL; Loss: {loss:.5f}")

    return total / len(test_loader)


# ------------------------------------------------------------
# TRAINING (MODIFIED with Markov Logic)
# ------------------------------------------------------------
def train(epoch_print, args, net, pad_idx, criterion, optimizer):
    global state_counter, current_state # Use the global state variables

    train_eur = EurDataset('train')
    train_loader = DataLoader(train_eur, batch_size=args.batch_size,
                              num_workers=0, pin_memory=True,
                              collate_fn=collate_data)

    # Train on a range of noise levels (e.g., 5dB to 15dB)
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(15))
    pbar = tqdm(train_loader)

    for sents in pbar:
        sents = sents.to(device)
        
        # --- "MARKOV STATE SWITCHER" LOGIC ---
        
        # 1. Check if we need to transition
        state_counter += 1
        if current_state["name"] == "LOS/AWGN" and state_counter > STATE_LENGTHS["LOS_len"]:
            # Transition from LOS to Shadowed [cite: 174, 207]
            current_state = STATE_SHADOWED
            state_counter = 0  # Reset counter for the new state
        elif current_state["name"] == "Shadowed/Rayleigh" and state_counter > STATE_LENGTHS["SH_len"]:
            # Transition from Shadowed back to LOS [cite: 174, 207]
            current_state = STATE_LOS
            state_counter = 0  # Reset counter
            
        # 2. Set the channel parameters for this batch
        Channels._sat_K_factor = current_state["K_factor"]
        Channels._sat_dist_km = current_state["dist_km"]
        
        # --- END OF "MARKOV" LOGIC ---

        # train_step will now use these new stable state parameters
        loss = train_step(net, sents, sents, noise_std, pad_idx,
                          optimizer, criterion, args.channel)
        
        # Update the progress bar to show what's happening
        pbar.set_description(f"Epoch: {epoch_print}; Train; State: {current_state['name']}; Loss: {loss:.5f}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == '__main__':

    args = parser.parse_args()

    # --- Set GLOBAL Channel Parameters ---
    # These are used by train_step/val_step, which read from the Channels class
    # Training will OVERWRITE these every batch based on the Markov state.
    # Validation will USE these defaults.
    Channels._sat_dist_km = args.sat_ref_dist_km # Default for validation
    Channels._sat_ref_dist_km = args.sat_ref_dist_km
    Channels._sat_K_factor = args.sat_K_factor # Default for validation
    Channels._sat_freq_hz = args.sat_freq_hz
    Channels._sat_d_tx_m = args.sat_dtx_m
    Channels._sat_d_rx_m = args.sat_drx_m
    Channels._sat_eff = args.sat_eff
    Channels._sat_extra_db = args.sat_extra_db
    # --- End Channel Params ---

    # Vocabulary
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]

    # Model
    net = DeepSC(args.num_layers, num_vocab, num_vocab,
                 num_vocab, num_vocab,
                 args.d_model, args.num_heads,
                 args.dff, 0.1).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4,
                                 betas=(0.9, 0.98), eps=1e-8,
                                 weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # auto-resume from checkpoint
    start_epoch = 1
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming training from: {args.resume_from}")
            net.load_state_dict(torch.load(args.resume_from, map_location=device))
            # Try to infer epoch number from filename
            try:
                start_epoch = int(args.resume_from.split('_')[-1].split('.')[0]) + 1
                print(f"Starting from epoch {start_epoch}")
            except:
                print("Could not infer epoch, starting from epoch 1 of new run.")
                start_epoch = 1 
        else:
            print(f"Warning: --resume-from file not found: {args.resume_from}")
            print("Starting new training from scratch.")
            initNetParams(net)
    else:
        print("Starting new training from scratch.")
        initNetParams(net)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------
    print(f"Starting training for {args.epochs} epochs.")
    print(f"Models will be saved to: {args.checkpoint-path}")
    print(f"Training with 2-State Markov Model: {STATE_LENGTHS['LOS_len']} batches (LOS) -> {STATE_LENGTHS['SH_len']} batches (Shadowed)")
    
    for epoch in range(start_epoch, start_epoch + args.epochs):

        # Train one epoch with the stateful Markov model
        train(epoch, args, net, pad_idx, criterion, optimizer)
        
        # Validate one epoch with the default (good) channel
        val_loss = validate(epoch, args, net, pad_idx, criterion)

        # Save new epoch checkpoint
        os.makedirs(args.checkpoint_path, exist_ok=True)
        ckpt_name = f"checkpoint_{epoch}.pth"
        torch.save(net.state_dict(), os.path.join(args.checkpoint_path, ckpt_name))
        print(f"Epoch {epoch} complete. Val Loss: {val_loss:.5f}. Checkpoint saved.")

    print("Training complete.")