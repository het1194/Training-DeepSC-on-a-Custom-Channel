# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
(This file is already perfect for the Markov model.
It is controlled by global parameters, which main.py will set.)
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from mutual_info import sample_batch, mutual_information

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        true_dist[:, self.padding_idx] = 0 
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        weight_decay = 0 # Disabled by default in original file
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 


class Channels():
    """
    Unified LEO Land Mobile Satellite Channel Model.
    This class now only contains the 'LEO_LMS' superclass channel.
    The parameters are set globally from main.py
    """
    
    # Global parameters set by main.py
    _sat_dist_km = 550.0
    _sat_ref_dist_km = 550.0
    _sat_freq_hz = 2.4e9
    _sat_d_tx_m = 0.6
    _sat_d_rx_m = 0.6
    _sat_eff = 0.6
    _sat_extra_db = 1.0
    _sat_K_factor = 15.0 # Default K-factor

    def LEO_LMS(self, Tx_sig, n_var):
        """
        Unified LEO Land Mobile Satellite Channel Model.
        - Handles FSPL via noise scaling relative to a reference distance.
        - Handles fading via a Rician K-factor:
            - K_factor = 0    -> Rayleigh
            - K_factor > 0    -> Rician
            - K_factor >> 10  -> AWGN/LOS-Only
        
        This model assumes perfect CSI (non-blind) and inverts the channel.
        """
        
        # --- 1. FSPL / Link Budget Noise Scaling (Your Satellite code) ---
        c = 299792458.0
        lam = c / float(self._sat_freq_hz)
        def dish_gain_linear(D, eta):
            return eta * (math.pi * D / lam) ** 2

        Gtx_dBi = 10.0 * math.log10(dish_gain_linear(self._sat_d_tx_m, self._sat_eff))
        Grx_dBi = 10.0 * math.log10(dish_gain_linear(self._sat_d_rx_m, self._sat_eff))

        def fspl_db(d_km, f_hz):
            f_MHz = f_hz / 1e6
            return 32.44 + 20.0 * math.log10(max(d_km, 1e-6)) + 20.0 * math.log10(f_MHz)

        def net_loss_db(d_km):
            return fspl_db(d_km, self._sat_freq_hz) - Gtx_dBi - Grx_dBi - float(self._sat_extra_db)

        delta_loss_db = net_loss_db(self._sat_dist_km) - net_loss_db(self._sat_ref_dist_km)
        
        # This is the new noise standard deviation, scaled by FSPL [cite: 3]
        scaled_noise_std = float(n_var) * (10.0 ** (delta_loss_db / 20.0))

        # --- 2. Fading Model (Huiqiang's Rician code) ---
        
        # Check if K is so high that we can treat it as pure AWGN [cite: 2]
        if self._sat_K_factor > 1000:
            # This is a pure LOS/AWGN channel. No fading matrix H.
            noise = torch.normal(0.0, scaled_noise_std, size=Tx_sig.shape, device=Tx_sig.device)
            Rx_sig = Tx_sig + noise
            return Rx_sig
        
        # This is a Fading Channel (Rician or Rayleigh)
        shape = Tx_sig.shape
        K = self._sat_K_factor
        
        # K=0 makes mean=0 (Rayleigh), K>0 makes mean>0 (Rician) [cite: 1, 2]
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (2 * (K + 1))) # 1/2 power for real, 1/2 for imag
        
        # Generate the 2x2 channel matrix H
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(0, std, size=[1]).to(device) # Imaginary part has 0 mean
        
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)

        # Apply the fading to the signal
        Tx_faded = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)

        # Apply AWGN noise (using the FSPL-scaled noise)
        noise = torch.normal(0.0, scaled_noise_std, size=Tx_faded.shape, device=Tx_faded.device)
        Rx_sig_faded_noisy = Tx_faded + noise

        # --- 3. Invert Channel (Assuming Perfect CSI) ---
        # This removes the fading effect, as in Huiqiang's original code [cite: 4]
        Rx_sig = torch.matmul(Rx_sig_faded_noisy, torch.inverse(H)).view(shape)

        return Rx_sig


def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    loss *= mask
    return loss.mean()

def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    model.train()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    # --- SIMPLIFIED CHANNEL CALL ---
    if channel == 'LEO_LMS':
        # This single function now handles AWGN, Rayleigh, and Rician
        # using the parameters set globally in main.py
        Rx_sig = channels.LEO_LMS(Tx_sig, n_var)
    else:
        raise ValueError("Unknown channel. Only 'LEO_LMS' is supported.")
    # --- END ---

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    
    ntokens = pred.size(-1)
    
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.0009 * loss_mine

    loss.backward()
    opt.step()

    return loss.item()


def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    # This function is not used by the new main.py, but we update it
    # for completeness.
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'LEO_LMS':
        Rx_sig = channels.LEO_LMS(Tx_sig, n_var)
    else:
        raise ValueError("Unknown channel. Only 'LEO_LMS' is supported.")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()

def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    # --- SIMPLIFIED CHANNEL CALL ---
    if channel == 'LEO_LMS':
        Rx_sig = channels.LEO_LMS(Tx_sig, n_var)
    else:
        raise ValueError("Unknown channel. Only 'LEO_LMS' is supported.")
    # --- END ---

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    
    return loss.item()
    
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """ 
    Uses the single LEO_LMS channel.
    The channel's behavior (FSPL, K-factor) is set globally from
    the performance.py script before this is called.
    """
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    # --- SIMPLIFIED CHANNEL CALL ---
    if channel == 'LEO_LMS':
        Rx_sig = channels.LEO_LMS(Tx_sig, n_var)
    else:
        raise ValueError("Unknown channel. Only 'LEO_LMS' is supported.")
    # --- END ---
          
    memory = model.channel_decoder(Rx_sig)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        
        prob = pred[: ,-1:, :] 
        _, next_word = torch.max(prob, dim = -1)
        
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs