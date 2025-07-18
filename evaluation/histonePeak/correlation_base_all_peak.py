#!/usr/bin/env python
# Copyright 2025 Z Zhang

# MambaHM, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
from scipy.stats import pearsonr, spearmanr
import pyBigWig
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import h5py


class CovFace:
  def __init__(self, cov_file):
    self.cov_file = cov_file
    self.bigwig = False
    self.bed = False

    cov_ext = os.path.splitext(self.cov_file)[1].lower()
    if cov_ext == '.gz':
      cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

    if cov_ext in ['.bed', '.narrowpeak']:
      self.bed = True
      self.preprocess_bed()

    elif cov_ext in ['.bw','.bigwig']:
      self.cov_open = pyBigWig.open(self.cov_file, 'r')
      self.bigwig = True

    elif cov_ext in ['.h5', '.hdf5', '.w5', '.wdf5']:
      self.cov_open = h5py.File(self.cov_file, 'r')

    else:
      print('Cannot identify coverage file extension "%s".' % cov_ext,
            file=sys.stderr)
      exit(1)

  def preprocess_bed(self):
    # read BED
    bed_df = pd.read_csv(self.cov_file, sep='\t',
      usecols=range(3), names=['chr','start','end'])

    # for each chromosome
    self.cov_open = {}
    for chrm in bed_df.chr.unique():
      bed_chr_df = bed_df[bed_df.chr==chrm]

      # find max pos
      pos_max = bed_chr_df.end.max()

      # initialize array
      self.cov_open[chrm] = np.zeros(pos_max, dtype='bool')

      # set peaks
      for peak in bed_chr_df.itertuples():
        self.cov_open[peak.chr][peak.start:peak.end] = 1


  def read(self, chrm, start, end):
    if self.bigwig:
      # print(chrm, start, end)
      cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')
      # print(cov)

    else:
      if chrm in self.cov_open:
        cov = self.cov_open[chrm][start:end]
        pad_zeros = end-start-len(cov)
        if pad_zeros > 0:
          cov_pad = np.zeros(pad_zeros, dtype='bool')
          cov = np.concatenate([cov, cov_pad])
      else:
        print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % \
          (self.cov_file, chrm, start, end), file=sys.stderr)
        cov = np.zeros(end-start, dtype='float16')

    return cov

  def close(self):
    if not self.bed:
      self.cov_open.close()


def read_peak(path, include_chr):
    """read peak regions from file (chr, start, end)"""
    sections = np.loadtxt(path, dtype='str')[:, :3]
    section_dict = {}

    for chr in include_chr:
        section_dict[chr] = []

    for section in sections:
        if section[0] in include_chr:
            section_dict[section[0]].append({'start': section[1], 'end': section[2]})

    return section_dict

def corr_resolution(
                    predicted_file,
                    experiment_file,
                    path_peak,
                    length=1280,
                    window_size=128,
                    include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8','chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17','chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
                    ):
    """
    compute pearson correlation and spearman correlation in the region contains all peaks
    Args:
        length: specifies the width of peak
        window_size: bin size used in evaluation
    """
    predicted_open = CovFace(predicted_file)
    experiment_open = CovFace(experiment_file)
    
    peaks = read_peak(path_peak, include_chr)
    pre_all = np.asarray([])
    exper_all = np.asarray([])

    for chr in peaks:
        if chr not in include_chr:
           continue
        chr_peaks = peaks[chr]
        for peak in chr_peaks:
            start = int(peak['start']) // window_size * window_size
            end = int(peak['end']) // window_size * window_size
            if end == start:
                end = start + 1 * window_size

            mid = (start + end) // 2
            start_extend = mid - int(length/2)
            end_extend = mid + int(length/2)

            try:
                predicted = predicted_open.read(chr, start_extend, end_extend)
                exper = experiment_open.read(chr, start_extend, end_extend)
            except:
                predicted = predicted_open.read(chr, start, end)
                exper = experiment_open.read(chr, start, end)
                
            predicted[np.where(np.isnan(predicted))] = 1e-3
            predicted[np.where(np.isinf(predicted))] = 1e-3

            exper[np.where(np.isnan(exper))] = 1e-3
            exper[np.where(np.isinf(exper))] = 1e-3

            predicted = np.mean(predicted.reshape(-1, window_size), axis=-1)
            exper = np.mean(exper.reshape(-1, window_size), axis=-1)

            if (predicted == predicted[0]).all() or (exper == exper[0]).all():
                continue

            pre_all = np.append(pre_all, predicted)
            exper_all = np.append(exper_all, exper)

    pre_all = np.asarray(pre_all, dtype='float32')
    exper_all = np.asarray(exper_all, dtype='float32')

    correlation = round(pearsonr(pre_all, exper_all)[0], 4)
    spe = round(spearmanr(pre_all, exper_all)[0], 4)

    predicted_open.close()
    experiment_open.close()

    return correlation, spe


def call_correlation(histone_list,
                     peak_file_dir,
                     experiment_file_dir,
                     predicted_file_dir,
                     include_chr=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                                  'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                                  'chr18', 'chr19', 'chr20', 'chr21', 'chr22'],
                     window_size=128):
    """
    Compute correlation between predicted and experimental histone modification signals.
    Evaluated in regions: TSS±1k, ±10k, ±30k, and >30k.

    Args:
        histone_list: list of histone modification types (e.g., ['H3k27ac']).
        peak_file_dir: directory containing peak .bed files.
        experiment_file_dir: directory containing ground-truth .bigWig files.
        predicted_file_dir: directory containing predicted .bw files.
        include_chr: list of chromosomes to include.
        window_size: resolution for correlation (default: 128 bp).
    """
    pearson_all = []
    spearman_all = []

    for his in histone_list:
        his = his.split('.')[0]
        print(f"Processing: {his}")

        experiment_file = f'{experiment_file_dir}/{his}.bigWig'
        predicted_file = f'{predicted_file_dir}/{his}.bw'
        path_peak = os.path.join(peak_file_dir, f'{his}.bed')

        try:
            correlation, spe = corr_resolution(predicted_file, experiment_file, path_peak,
                                               include_chr=include_chr, window_size=window_size)
        except Exception as e:
            print(f"Error processing {his}: {e}")
            correlation, spe = -1, -1

        print(f"Pearson: {correlation}, Spearman: {spe}")
        pearson_all.append(correlation)
        spearman_all.append(spe)

        print('-' * 50)

    print('All Pearson correlations:')
    for item in pearson_all:
        print(item)
    print('All Spearman correlations:')
    for item in spearman_all:
        print(item)

    return pearson_all, spearman_all


def main():
    # 固定的路径，无需 cellline 循环
    histone_list = ['H3k9me3']

    peak_file_dir = 'MambaHM/genome_regions/HistonePeak/GM12878/'
    experiment_file_dir = 'exper_dir'
    predicted_file_dir = 'pred_dir'
    window_size = 16

    print("Running correlation analysis for histone modifications...")
    call_correlation(histone_list,
                     peak_file_dir,
                     experiment_file_dir,
                     predicted_file_dir,
                     window_size=window_size)

    
if __name__ == '__main__':
    main()
