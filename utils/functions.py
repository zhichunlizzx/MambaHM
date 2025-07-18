#!/usr/bin/env python
# Copyright 2025 Z Zhang

# MambaHM, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pysam
import math
import pyBigWig


'''
load_data
'''


def check_if_out_of_bounds(samples, chrom_size):
    """
    Check for out of bounds samples

    Args:
        samples: a data frame of samples
        chrom_size: chromosize of the reference genome of samples
    
    Return:
        The chromosome of the false sample or None
    """
    samples = samples[np.argsort(samples, axis=0)[:, 0]]

    for chr in np.unique(samples[:, 0]):
        chr_idx = np.argwhere(samples[:, 0] == chr).squeeze(-1)
        chr_samples = samples[chr_idx]
        chr_samples = chr_samples[np.argsort(chr_samples[:, -1].astype(int))]
        if int(chr_samples[0][1]) < 0 or int(chr_samples[-1][2]) > chrom_size[chr][0][-1]:
            return chr
        
    return None


def bw_2_chromosome_size(bw_file):
    """Read chromosome size from .bw file"""
    try:
        bw_open = pyBigWig.open(bw_file)
    except:
        raise Exception('Error: bw_file must be a bigwig file')
    chromsize = bw_open.chroms()
    for chr in chromsize:
        chromsize[chr] = [(0, chromsize[chr])]
    return chromsize


def fai_2_choromosome_size(fai_file):
    """Read chromosome size from fai file"""
    with open(fai_file, 'r') as r_obj:
        lines = r_obj.readlines()
    sections = [section.split() for section in lines]

    chrom_size = {}
    for section in sections:
        chrom_size[section[0]] = [(0, int(section[1]))]
    
    return chrom_size


def load_chromosomes(genome_file):
    """ Load genome segments from either a FASTA file or chromosome length table. """
    # is genome_file FASTA or (chrom,start,end) table?
    file_fasta = (open(genome_file).readline()[0] == '>')

    chrom_segments = {}
    try:
        if file_fasta:
            fasta_open = pysam.Fastafile(genome_file)
            for i in range(len(fasta_open.references)):
                chrom_segments[fasta_open.references[i]] = [(0, fasta_open.lengths[i])]
            fasta_open.close()
        else:
            # (chrom,start,end) table
            for line in open(genome_file):
                a = line.split()
                chrom_segments[a[0]] = [(0, int(a[1]))]
    except:
        raise Exception('Error: reference genome file errore')

    return chrom_segments


def split_based_chr(samples, divide_chr=['chr22']):
    '''
    split samples to training, validation and test set

    Args:
        samples: [num_samples, 3]
        divide_chr: select the samples of chromosomes in divide_chr

    Return:
        samples_divided: the samples of chromosomes in divide_chr
        samples_reserved: the rest of the samples
    '''
    divided_idx = [sample in divide_chr for sample in samples[:, 0]]

    reserved_idx = (np.asarray(divided_idx) == False)

    samples_reserved = samples[reserved_idx]
    samples_divided = samples[divided_idx]
    
    return samples_divided, samples_reserved


def split_based_percent(samples, chose_sample_percent=1.):
    '''
    split samples to two part based on appointed percent

    Args:
        samples: [num_samples, 3]
        chose_sample_percent: division ratio(float)

    Return:
        chose_samples: the sample of the chosen_sample_percent ratio in samples
        reserved_samples: the sample of the (1 - chosen_sample_percent) ratio in samples
    '''

    if chose_sample_percent > 1:
        raise Exception('Error: chose_sample_percent must be an integer less than 1')

    num_chose_sample = math.floor(chose_sample_percent * len(samples))

    chose_sample_idx = list(np.random.choice(list(range(len(samples))), num_chose_sample, replace=False))

    reserved_sample_idx = list(set(list(range(len(samples)))).difference(set(chose_sample_idx)))

    chose_samples = samples[chose_sample_idx]
    reserved_samples = samples[reserved_sample_idx]

    return chose_samples, reserved_samples


def split_based_num(samples, chose_num=1):
    '''
    split samples to two part based on num of samples

    Args:
        samples: [num_samples, 3]
        chose_num: chose num

    Return:
        chose_samples: the sample of the chosen_sample_percent ratio in samples
        reserved_samples: the sample of the (1 - chosen_sample_percent) ratio in samples
    '''
    # select a part of the sample
    # train and valid
    if chose_num < 1 or not(type(chose_num)==int):
        raise Exception('Error: chose_sample_num must be an integer greater than 0')
    
    if len(samples) < chose_num:
        raise Exception('Error: chose_num exceeds the maximum num of samples')

    num_chose_sample = chose_num

    chose_sample_idx = list(np.random.choice(list(range(len(samples))), num_chose_sample, replace=False))

    reserved_sample_idx = list(set(list(range(len(samples)))).difference(set(chose_sample_idx)))

    chose_samples = samples[chose_sample_idx]
    reserved_samples = samples[reserved_sample_idx]

    return chose_samples, reserved_samples
