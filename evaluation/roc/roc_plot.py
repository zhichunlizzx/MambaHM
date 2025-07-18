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
from sklearn.metrics import auc
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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


if __name__ == '__main__':
    outdir ='outdir'

    window_size=16

    cellline_list = ['HeLa']

    models = ['MambaHM', 'dHICA', 'dHIT', 'Enformer']

    color_list = ['#a1a0a5', '#d65316', '#ecb21d', '#0075ba']

    # histone item
    item = 'H3k4me1'
               
    fpr_list = []
    tpr_list = []
    auc_list = []
    pre_list = []
    recall_list = []
    prc_list = []
    roc_out = os.path.join(outdir, 'roc_comparision/roc_%s.pdf' % (item))
    aupr_out = os.path.join(outdir, 'roc_comparision/prc_%s.pdf' % (item))

    for model in models:
        outpath = os.path.join(outdir, 'roc/%s/%d/%s' % (model, window_size, item))

        fpr_out = os.path.join(outpath, 'fpr.txt')
        tpr_out = os.path.join(outpath, 'tpr.txt')
        precision_out = os.path.join(outpath, 'precision.txt')
        recall_out = os.path.join(outpath, 'recall.txt')
        prc_out = os.path.join(outpath, 'prc.txt')


        # auc parameters
        fpr = np.loadtxt(fpr_out, delimiter=',')
        tpr = np.loadtxt(tpr_out, delimiter=',')
        roc_auc = auc(fpr, tpr)
        
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)

        # prc parameters
        precision = np.loadtxt(precision_out, delimiter=',')
        recall = np.loadtxt(recall_out, delimiter=',')
        prc = np.loadtxt(prc_out, delimiter=',')

        pre_list.append(precision)
        recall_list.append(recall)
        prc_list.append(prc)


    # 设置全局字体为 Arial，字体大小为 7
    plt.rc('font', family='Arial', size=5)
    curve_width = 1

    # Calculating the required size in inches for the image
    width_mm = 52 # width in mm
    height_mm = 48 # height in mm

    # Conversion factor from mm to inches
    mm_to_inch = 25.4

    # Convert to inches
    width_inch = width_mm / mm_to_inch
    height_inch = height_mm / mm_to_inch
    
    border_linewidth = 0.8  # 设置边框宽度

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(width_inch, height_inch))
    #################
    # ROC
    ##########

    for i in range(len(models)):
        fpr = fpr_list[i]
        tpr = tpr_list[i]
        roc_auc = auc_list[i]
        plt.plot(fpr, tpr, lw=curve_width, color=color_list[i], label=f'{models[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='black', lw=border_linewidth, linestyle='--')
    plt.xlim([0.0, 1.])
    plt.ylim([0.0, 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (%s)' % item)
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    
    ax.spines['top'].set_linewidth(border_linewidth)    # 设置上边框宽度
    ax.spines['right'].set_linewidth(border_linewidth)  # 设置右边框宽度
    ax.spines['left'].set_linewidth(border_linewidth)   # 设置左边框宽度
    ax.spines['bottom'].set_linewidth(border_linewidth)  # 设置下边框宽度  

    plt.savefig(roc_out)
    plt.clf()


    #################
    # PRC
    ##########

    for i in range(len(models)):
        pre = pre_list[i]
        recall = recall_list[i]
        prc = prc_list[i]
        plt.plot(recall, pre, lw=curve_width, color=color_list[i], label=f'{models[i]} (AUPR = {prc:.4f})')

    
    plt.plot([0, 1], [1, 0], color='black', lw=border_linewidth, linestyle='--')
    plt.xlim([0.0, 1.])
    plt.xlabel("Recall")
    plt.ylabel("Precision (%s)" % item)
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    ax.spines['top'].set_linewidth(border_linewidth)    # 设置上边框宽度
    ax.spines['right'].set_linewidth(border_linewidth)  # 设置右边框宽度
    ax.spines['left'].set_linewidth(border_linewidth)   # 设置左边框宽度
    ax.spines['bottom'].set_linewidth(border_linewidth)  # 设置下边框宽度

    plt.savefig(aupr_out)
    plt.clf()
            