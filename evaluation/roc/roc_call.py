#!/usr/bin/env python
# Copyright 2025 Z Zhang

# MambaHM, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import os
import pyBigWig
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
import numpy as np
import pyBigWig
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import subprocess
import json
import gc


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


def read_values(bw_file, label_file, regions, window_size, max=192, min=0.5):
    predicted_open = pyBigWig.open(bw_file, 'r')
    label_open = pyBigWig.open(label_file, 'r')

    max = predicted_open.header()['maxVal']

    predicted_data = np.asarray([])
    label_data = np.asarray([])
    for region in regions:
        pre = predicted_open.values(region[0], int(region[1]), int(region[2]), numpy=True).astype('float16').reshape((-1, window_size))
        label = label_open.values(region[0], int(region[1]), int(region[2]), numpy=True).astype('float16').reshape((-1, window_size))
        label[np.isnan(label)] = 0
        label[np.isinf(label)] = 0

        pre[np.isnan(pre)] = 0
        pre[np.isinf(pre)] = 0
        
        pre = np.max(pre, axis=-1) / max
        pre[pre>1] = 1

        label = (np.max(label, axis=-1) > min).astype('int')
        predicted_data = np.append(predicted_data, pre)
        label_data = np.append(label_data, label)

    predicted_open.close()
    label_open.close()

    return predicted_data, label_data

def prc_plt(predicted_values, label_values, out_file):
    precision, recall, thresholds = precision_recall_curve(label_values, predicted_values)
    ap = average_precision_score(label_values, predicted_values, average='macro', sample_weight=None)
    plt.plot(recall, precision, label='PRC (area = {0:.2f})'.format(ap), lw=2)
    plt.plot([0,1], linestyle='--')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("PRC curve")  
    plt.legend(loc="lower right")
    plt.savefig(out_file)
    plt.clf()

    return ap


def roc_plt(predicted_values, label_values, out_file):
    fpr, tpr, thersholds = roc_curve(label_values, predicted_values, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
    plt.plot([0,1], linestyle='--')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("ROC curve")  
    plt.legend(loc="lower right")
    plt.savefig(out_file)
    plt.clf()

    return roc_auc

def linechart(x, y, out_file):
    item = os.path.basename(out_file).split('.')[0]
    plt.plot(x, y, lw=2)
    plt.plot([0,1], linestyle='--')
    plt.xlabel("threshold")
    plt.ylabel(item)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(item)  
    plt.legend(loc="lower right")
    plt.savefig(out_file)
    plt.clf()


def bw_eva(bw_file, label_file, outpath, item, include_chr=['chr22'], window_size=128, reference_genome_idx = '/local/zzx/hg19_data/chrom_size.bed'):
    whole_genome_size = bw_2_chromosome_size(bw_file=bw_file)
    regions = []
    for chr in whole_genome_size:
        if chr in include_chr:
            regions.append([chr, 0, whole_genome_size[chr][0][-1] // window_size * window_size])
    
    # sort label 
    sort_label = os.path.join(outpath, item+'.sort.bed')
    cmd_bedSort = 'sort-bed ' + label_file + ' > ' + sort_label
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()
    # merge label
    merge_label = os.path.join(outpath, item+'.merge.bed')
    cmd_merge = 'bedtools merge -i ' + sort_label + ' > ' + merge_label
    p = subprocess.Popen(cmd_merge, shell=True)
    p.wait()
    # sort label
    sort_label = os.path.join(outpath, item+'.sort.bed')
    cmd_bedSort = 'sort-bed ' + merge_label + ' > ' + sort_label
    p = subprocess.Popen(cmd_bedSort, shell=True)
    p.wait()
    
    bedGraph_label = os.path.join(outpath, item+'.bedgraph')

    labels = np.loadtxt(sort_label, dtype='str', delimiter='\t')

    # (chr start, end) --> (chr, start, end, 1)
    with open(bedGraph_label, 'w') as w_obj:
        for label in labels:
            if label[0] in include_chr:
                w_obj.write('\t'.join(label)+'\t1\n')

    bw_label_file = os.path.join(outpath, item+'.bw')

    cmd = ['bedGraphToBigWig', bedGraph_label, reference_genome_idx, bw_label_file]
    subprocess.call(cmd)
    
    predicted_values, label_values = read_values(bw_file, bw_label_file, regions, window_size)

    cmd_rm = ['rm', '-f', merge_label, sort_label, bedGraph_label, bw_label_file]
    subprocess.call(cmd_rm)

    return predicted_values, label_values

 

    
if __name__ == '__main__':

    restart = False

    label_dir = 'MambaHM/genome_regions/HistonePeak/'

    predicted_bw_dir = 'pred_dir'
    outdir = 'outdir'

    window_size=16

    reference_genome_idx = 'MambaHM/genome_regions/chrome_size/hg19/chrom_size.bed'
        
    fdr_outpath = os.path.join(outdir, 'prc.txt')
    if restart:
        with open(fdr_outpath, 'r') as r_obj:
            prc_dict = json.load(r_obj)
    # histone items

    items = ['H3k4me1', 'H3k4me2', 'H3k4me3', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k9ac', 'H4k20me1']
    # include chr
    include_chr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
                'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                'chr18', 'chr19', 'chr20', 'chr21', 'chr22']

    aucs = []
    prcs = []
    for item in items:
        bw_file = os.path.join(predicted_bw_dir, '%s.bw' % (item))
        label_file = os.path.join(label_dir, '%s.bed' % (item))
        outpath = os.path.join(outdir, '%d/%s' % (window_size, item))
        fpr_out = os.path.join(outpath, 'fpr.txt')
        tpr_out = os.path.join(outpath, 'tpr.txt')
        precision_out = os.path.join(outpath, 'precision.txt')
        recall_out = os.path.join(outpath, 'recall.txt')
        prc_out = os.path.join(outpath, 'prc.txt')

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        predicted_values, label_values = bw_eva(bw_file, label_file, outpath, item, include_chr=include_chr, window_size=window_size, reference_genome_idx=reference_genome_idx)
        
        # roc parameters
        fpr, tpr, thresholds = roc_curve(label_values, predicted_values, pos_label=1)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # prc parameters
        precision, recall, _ = precision_recall_curve(label_values, predicted_values)
        prc = average_precision_score(label_values, predicted_values, average='macro', sample_weight=None)

        prcs.append(prc)

        np.savetxt(fpr_out, fpr, fmt='%f', delimiter=',')
        np.savetxt(tpr_out, tpr, fmt='%f', delimiter=',')
        np.savetxt(precision_out, precision, fmt='%f', delimiter=',')
        np.savetxt(recall_out, recall, fmt='%f', delimiter=',')
        np.savetxt(prc_out, [prc], fmt='%f', delimiter=',')

        # 删除单个变量
        del predicted_values, label_values, fpr, tpr, thresholds, precision, recall

        # 强制触发垃圾回收
        gc.collect()


    print('AUC: ')
    for auc_item in aucs:
        print(round(auc_item, 4))
    print('AUPR: ')
    for prc_item in prcs:
        print(round(prc_item, 4))
    print('-'*50)
    






            