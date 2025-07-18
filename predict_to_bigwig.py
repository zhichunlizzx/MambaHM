#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
from re import I
import numpy as np
from model_biMamba import Dhit2
from functions import bw_2_chromosome_size
from dataloader import RD_dataloader
from torch.utils.data import DataLoader
from numpy.core.multiarray import scalar
from tqdm import tqdm
from typing import Optional
from torchmetrics import Metric
from einops import rearrange
from operator import itemgetter
import subprocess

import torch
DEVICE=torch.device("cuda:0")

print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])
print("Torch sees", torch.cuda.device_count(), "GPUs")
print("Device 0 name:", torch.cuda.get_device_name(0))


def write_predicted_result(
                        results,
                        out_path,
                        chr_length,
                        target_list,
                        reference_genome_idx,
                        seq_length=114688,
                        window_size=128,
                        ):
    """ 
    Write result to bigwig file

    Args:
        results: predicted result, {chr:[{start:xx, end:xx, result:xx}]}
        out_path: output path
        chr_length: chromosome length
        target_list: target sequencing data list
        reference_genome_idx: reference genome idx

    Return:
        None
    """
    seq_length = seq_length
    target_length = seq_length // window_size
    
    for j in range(len(target_list)):
            if os.path.isfile(os.path.join(out_path, target_list[j] + '.bedgraph')):
                os.remove(os.path.join(out_path, target_list[j] + '.bedgraph'))

    for chr in results:
        chr_result = results[chr]
        chr_result = sorted(chr_result, key=itemgetter('start'))
        for j in range(len(target_list)):
            with open(os.path.join(out_path, target_list[j] + '.bedgraph'), 'a') as w_obj:
                # assign 0 to the area not covered by the sample
                if chr_result[0]['start'] > 0:
                    w_obj.write(chr + '\t' + str(0) + '\t' + str(chr_result[0]['start']) + '\t' + str(0) + '\n')

                # write predict result
                last_end = 0
                for item in chr_result:
                    if item['start'] >= last_end: 
                        for i in range(target_length):
                            start = item['start'] + i * window_size
                            end = start + window_size
                            w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                    else:
                        print(item)
                        gap_h = last_end - item['start']
                        h_start = gap_h // window_size
                        w_obj.write(chr + '\t' + str(last_end) + '\t' + str(item['start'] + window_size * (h_start+1)) + '\t' + str(item['predicted'][h_start][j]) + '\n')
                        for i in range(h_start+1, target_length):
                            start = item['start'] + i * window_size
                            end = start + window_size 
                            w_obj.write(chr + '\t' + str(start) + '\t' + str(end) + '\t' + str(item['predicted'][i][j]) + '\n')
                last_end = item['end']

                # assign 0 to the area not covered by the sample
                if chr_result[-1]['end'] < chr_length[chr]:
                    w_obj.write(chr + '\t' + str(chr_result[-1]['end']) + '\t' + str(chr_length[chr]) + '\t' + str(0) + '\n')

    # bedgraph to bigwig
    for j in range(len(target_list)):
        bed_path = os.path.join(out_path, target_list[j] + '.bedgraph')
        bedgraph_path_sorted = os.path.join(out_path, target_list[j] + '_sorted.bedgraph')
        cmd_bedSort = 'sort-bed ' + bed_path + ' > ' + bedgraph_path_sorted
        p = subprocess.Popen(cmd_bedSort, shell=True)
        p.wait()

        bw_path = os.path.join(out_path, target_list[j] + '.bw')

        cmd = ['bedGraphToBigWig', bedgraph_path_sorted, reference_genome_idx, bw_path]
        subprocess.call(cmd)

        cmd_rm = ['rm', '-f', bed_path]
        subprocess.call(cmd_rm)

        cmd_rm = ['rm', '-f', bedgraph_path_sorted]
        subprocess.call(cmd_rm)

    return True


'''
evaluation
'''
class MeanPearsonCorrCoefPerChannel(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    def __init__(self, n_channels:int, dist_sync_on_step=False):
        """Calculates the mean pearson correlation across channels aggregated over regions"""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_dims=(0, 1)
        self.add_state("product", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("true_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("pred_squared", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum", )
        self.add_state("count", default=torch.zeros(n_channels, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.product += torch.sum(preds * target, dim=self.reduce_dims)
        self.true += torch.sum(target, dim=self.reduce_dims)
        self.true_squared += torch.sum(torch.square(target), dim=self.reduce_dims)
        self.pred += torch.sum(preds, dim=self.reduce_dims)
        self.pred_squared += torch.sum(torch.square(preds), dim=self.reduce_dims)
        self.count += torch.sum(torch.ones_like(target), dim=self.reduce_dims)

    def compute(self):
        true_mean = self.true / self.count
        pred_mean = self.pred / self.count

        covariance = (self.product
                    - true_mean * self.pred
                    - pred_mean * self.true
                    + self.count * true_mean * pred_mean)

        true_var = self.true_squared - self.count * torch.square(true_mean)
        pred_var = self.pred_squared - self.count * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation
    

def convert_resolution(target, window_width, aim_resolution):
    k = aim_resolution // window_width
    target = rearrange(target, 'b (r n) d -> b r n d', n = k)
    target = torch.mean(target, dim=2)
    return target


reference_genome_file = '/local1/zzx/data/hg19/hg19.fa'
# reference_genome_file = '/local1/zzx/data/mm10/mm10.fa'

sequence_data_file = [
                        [
                        # ['/local1/zzx/hg19_data/atac-seq/k562_ENCFF092ESJ.bw'], # Extremely low read depth
                        # ['/local1/zzx/hg19_data/atac-seq/k562_ENCFF451IKJ.bw'],
                        # ['/local1/zzx/hg19_data/atac-seq/k562_ENCFF156WZT.bw'],
                        # ['/local1/zzx/hg19_data/atac-seq/k562_ENCFF093IIW.bw'],

                        # GM12878
                        # ['/local1/zzx/hg19_data/atac-seq/gm_ENCFF316PVI.bw']
                        # HCT116
                        # ['/local1/zzx/hg19_data/atac-seq/hct_ENCFF725NBM.bw']
                        # Hela 有点假
                        # ['/local1/zzx/hg19_data/atac-seq/hela_Naive3hCatATAC2trimBowtie2FilterMarkDuplicatesnormbigWig.bw']
                        # CD4
                        ['/local1/zzx/hg19_data/atac-seq/cd4_ENCFF343VRK.bw']
                        # IMR90
                        # ['/local1/zzx/hg19_data/atac-seq/imr90_ENCFF921WHS.bw']
                        # MCF7
                        # ['/local1/zzx/hg19_data/atac-seq/mcf7.hg19.ENCFF782BVX.bw']
                        # HepG2
                        # ['/local1/zzx/hg19_data/atac-seq/hepG2_ENCFF024GLW.bw']
                        # A549
                        # ['/local1/zzx/hg19_data/atac-seq/a549_ENCFF272ATG.bw']
                        # Heart
                        # ['/local1/zzx/hg19_data/atac-seq/heart_hg19_ENCFF840TYL.bw']
                        # Spleen
                        # ['/local1/zzx/hg19_data/atac-seq/spleen_hg19_ENCFF869NXS.bw']
                        # Heart mm10
                        # ['/local1/zzx/hg19_data/atac-seq/heart_mm10_ENCFF890UDQ.bw']
                        # mm10_Hindbrain
                        # ['/local1/zzx/hg19_data/atac-seq/mm10_Hindbrain_ENCFF752FVM.bigWig']
                        # G1E
                        # ['/local1/zzx/hg19_data/atac-seq/G1E_mm10_ENCFF852SDC_0.55.bigWig'],

                         ]
                        
                        ]

target_seq_file = [
            # K562
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local1/zzx/hg19_data/histone/histone/H3k4me1.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H3k4me2.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H3k4me3.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H3k27ac.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H3k27me3.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H3k36me3.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H3k9ac.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H3k9me3.bigWig', 
            # '/local1/zzx/hg19_data/histone/histone/H4k20me1.bigWig', ],

            # GM12878
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_gm/H4K20me1.bigWig',],

            # # HCT116
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hct/H4K20me1.bigWig',]

            # # HeLa
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hela/H4K20me1.bigWig',]

            # # CD4
            ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            '/local33/ww/data/cell_bw/his_bw/his_cd4/H3K4me1.bigWig',
            '/local1/zzx/hg19_data/histone/histone/H3k4me2.bigWig',
            '/local33/ww/data/cell_bw/his_bw/his_cd4/H3K4me3.bigWig',
            '/local1/zzx/hg19_data/histone/histone/H3k27ac.bigWig',
            '/local33/ww/data/cell_bw/his_bw/his_cd4/H3K27me3.bigWig',
            '/local33/ww/data/cell_bw/his_bw/his_cd4/H3K36me3.bigWig',
            '/local33/ww/data/cell_bw/his_bw/his_cd4/H3K9ac.bigWig',
            '/local33/ww/data/cell_bw/his_bw/his_cd4/H3K9me3.bigWig',
            '/local1/zzx/hg19_data/histone/histone/H4k20me1.bigWig',]

            # IMR-90
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_imr90/H4K20me1.bigWig',]


            # MCF7
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_mcf/H4K20me1.bigWig',]

            # # HepG2
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_hepG2/H4K20me1.bigWig',]

            # A549
            # ['/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/his_bw/his_a549/H4K20me1.bigWig',]

            # Heart hg19
            # ['/local33/ww/data/cell_bw/tissue/heart_hg19/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_hg19/H3K9me3.bigWig',]

            # Spleen hg19
            # ['/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/spleen_hg19/H3K9me3.bigWig',]

            # Heart mm10
            # ['/local33/ww/data/cell_bw/tissue/heart_mm10/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/heart_mm10/H3K9me3.bigWig',]

            # mm10_Hindbrain
            # ['/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K4me2.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K27ac.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K9ac.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/hindbrain_mm10/H3K9me3.bigWig',]

            # G1E mm10
            # ['/local33/ww/data/cell_bw/tissue/G1E/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K4me1.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K4me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K27me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K36me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K9me3.bigWig',
            # '/local33/ww/data/cell_bw/tissue/G1E/H3K9me3.bigWig',]
    ]

print(sequence_data_file)

# 由/local1/zzx/code/LongSeq/ConvMamba/histone_test_samples.bed对区间做出调整得到，长度更改为114688
samples = np.loadtxt('/local1/zzx/code/LongSeq/ConvMamba/ATAC_predict_all_hg19.bed', dtype=str, delimiter='\t')
# samples = samples[samples[:, 0] == 'chr22']
# print(samples)
# samples = np.loadtxt('/local1/zzx/code/LongSeq/ConvMamba/ATAC_predict_all_mm10.bed', dtype=str, delimiter='\t')
model_dir = '/local1/zzx/code/LongSeq/ConvMamba_1.1/ATAC_model_save/2Mamba_lr_1e-4_dynamic_2/model'

# 添加numpy的scalar到安全全局变量
torch.serialization.add_safe_globals([scalar])

checkpoint_path = os.path.join(model_dir, 'model (2).pth')
checkpoint = torch.load(checkpoint_path, map_location=DEVICE    )

model = Dhit2(channels=384,
            num_heads=8,
            num_transformer_layers=11,
            pooling_type='max',
            output_channels=len(target_seq_file[0]),
            target_length=7168,
            device=DEVICE
            ).to(DEVICE)
# model = torch.nn.DataParallel(model, device_ids=[0])
model.load_state_dict(checkpoint['model_state_dict'])

window_width = 16

# evaluation
eva_data_loader = RD_dataloader(samples,
                        reference_genome_file,
                        sequence_data_file,
                        target_seq_file,
                        window_width=window_width,
                        extend=40960,  
                        nan=0,
                        valid=True,
                        rc=False,
                        )
eva_dataset = DataLoader(dataset=eva_data_loader, batch_size=1, shuffle=False)

target_list = [
            'H3k122ac',
            'H3k4me1',
            'H3k4me2',
            'H3k4me3',
            'H3k27ac',
            'H3k27me3',
            'H3k36me3',
            'H3k9ac',
            'H3k9me3',
            'H4k20me1',
            ]
cellline = 'CD4'
out_path = '/local1/zzx/code/LongSeq/ConvMamba_1.1/ATAC_model_save/2Mamba_lr_1e-4_dynamic_2/results'
out_path = os.path.join(out_path, cellline)
results = {}
print(target_list)

chrom_size = bw_2_chromosome_size(sequence_data_file[0][0][0])

chr_length = {}
for chr in np.unique(samples[:, 0]):
    results[chr] = []
    chr_length[chr] = chrom_size[chr][0][1]

if not os.path.isdir(out_path):
    os.mkdir(out_path)

# chromosome length file
reference_genome_idx = os.path.join(out_path, 'idx.fai')
with open(reference_genome_idx, 'w') as w_obj:
    for chr in chrom_size:
        w_obj.write(chr + '\t' + str(chrom_size[chr][0][1]) + '\n')

metric = MeanPearsonCorrCoefPerChannel(n_channels=10)
model.eval()
with torch.no_grad():
    for i, eva_data_item in tqdm(enumerate(eva_dataset), ascii=True):
        result = {}
        if i > len(samples):
            break
        dna = eva_data_item[0].to(DEVICE)
        seq = eva_data_item[1].to(DEVICE)
        target = eva_data_item[-1].to(DEVICE)
        predicted = model(dna, seq).to(DEVICE)
        # print(target.shape)
        # print(predicted.shape)
        target_resolution = convert_resolution(target, window_width, 1024).detach().cpu()
        predicted_resolution = convert_resolution(predicted, window_width, 1024).detach().cpu()

        target_t = target_resolution
        predicted_t = predicted_resolution
        metric.update(target_t, predicted_t)

        predict_cpu = predicted.detach().cpu()
        result['chr'] = samples[i][0]
        result['start'] = int(samples[i][1])
        result['end'] = int(samples[i][2])
        result['predicted'] = predict_cpu[0].numpy()
        results[result['chr']].append(result)

write_down = write_predicted_result(results, out_path, chr_length, target_list, reference_genome_idx, seq_length=114688, window_size=16)

for pearson in metric.compute().numpy():
    print(round(pearson, 4))
