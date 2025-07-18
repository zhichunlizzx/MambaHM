#!/usr/bin/env python
# Copyright 2023 Z Zhang

# BioSeq2Seq, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .get_feature import get_input_seq_feature, get_target_feature, get_dna_seq_onehot_encoding

class RD_dataloader(Dataset):
    def __init__(self,
                 samples,
                 reference_genome_file,
                 sequencing_data_file,
                 target_sequencing_file=None,
                 window_width=128,
                 extend=40960,
                 rc=True,
                 nan=None,
                 valid=False,
                 test=False) -> None:
        super().__init__()
        self.samples = samples
        self.reference_genome_file = reference_genome_file
        self.sequencing_data_file = sequencing_data_file
        self.target_sequencing_file = target_sequencing_file
        self.window_width = window_width
        self.extend = extend
        self.nan = nan
        self.valid = valid
        self.test = test
        self.rc = rc

        self.seq_file_num = len(sequencing_data_file[0])

    def __getitem__(self, index):
        # seq_id = int(self.samples[index][3])
        seq_id = 0
        sample = [[self.samples[index][0], int(self.samples[index][1]), int(self.samples[index][2])]]
        # print(sample, seq_id)
        # print(sample)
        # if self.valid:
        #     seq_id = 0
        # else:
        #     seq_id = np.random.randint(0, self.seq_file_num)
        # print(self.sequencing_data_file[0][seq_id])
        dna_encoding = np.squeeze(get_dna_seq_onehot_encoding(sample, self.reference_genome_file, extend=self.extend, rc=self.rc), 0)
        seq_feature = np.squeeze(get_input_seq_feature(sample, self.sequencing_data_file[0][seq_id], extend=self.extend, nan=self.nan), 0)
        dna_encoding = torch.from_numpy(dna_encoding).to(torch.float32)
        seq_feature = torch.from_numpy(seq_feature).to(torch.float32)
        
        if not self.test:
            # print(self.target_sequencing_file[seq_id])
            target = np.squeeze(get_target_feature(sample, self.target_sequencing_file[seq_id], window_width=self.window_width, mean_method='mean', nan=self.nan), 0)
            target = torch.from_numpy(target).to(torch.float32)
            return dna_encoding, seq_feature, target
        else:
            return dna_encoding, seq_feature

    def __len__(self):
        """
        return number of sample.
        """
        return len(self.samples)
    

if __name__ == '__main__':
    reference_genome_file = '/local1/zzx/data/hg19/hg19.fa'
    input_data_peak_path = ['/local1/zzx/za/Ro-seq-3']
    output_data_peak_path = ['/local1/zzx/hg19_data/new_classified_data/histone-3']

    sequence_data_file = [
                            [
                            ['/local1/zzx/hg19_data/proseq/G1_minus.bw', '/local1/zzx/hg19_data/proseq/G1_plus.bw'],
                            ['/local1/zzx/hg19_data/proseq/G2_minus.bw', '/local1/zzx/hg19_data/proseq/G2_plus.bw'],
                            ['/local1/zzx/hg19_data/proseq/G3_minus.bw', '/local1/zzx/hg19_data/proseq/G3_plus.bw'],
                            ['/local1/zzx/hg19_data/proseq/G4_minus.bw', '/local1/zzx/hg19_data/proseq/G4_plus.bw'],
                            ['/local1/zzx/hg19_data/proseq/G5_minus.bw', '/local1/zzx/hg19_data/proseq/G5_plus.bw'],
                            ['/local1/zzx/hg19_data/proseq/G6_minus.bw', '/local1/zzx/hg19_data/proseq/G6_plus.bw'],
                            ['/local1/zzx/hg19_data/proseq/G7_minus.bw', '/local1/zzx/hg19_data/proseq/G7_plus.bw'],
                            ]
                            
                            ]

    target_seq_file = [
        '/local1/zzx/hg19_data/histone/histone/H3k122ac.bigWig',
        '/local1/zzx/hg19_data/histone/histone/H3k4me1.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H3k4me2.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H3k4me3.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H3k27ac.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H3k27me3.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H3k36me3.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H3k9ac.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H3k9me3.bigWig', 
        '/local1/zzx/hg19_data/histone/histone/H4k20me1.bigWig', 
        # TREs
        # '/local/zzx/hg19_data/new_classified_data/new_promoter_enhancer/all_promoter_enhancer.bw', 
        # '/local2/zzx/hg19_data/dhit2_data/peak_file/TRE/test_genebody/ncbi/expression_stat/genebody_7_lines_only_gene/stat_percent_max_128_only_great_0.4/G1/genebody.bw',
        # '/local2/zzx/hg19_data/dhit2_data/peak_file/TRE/test_genebody/ncbi/expression_stat/genebody_7_lines_only_gene/stat_percent_max_128_only_great_0.4/G1/polya.bw', 
        # '/local/zzx/hg19_data/new_classified_data/bw/insulator.bw', 
        ]
    
    samples = [['chr20', '48571488', '48686176'],
                ['chr2', '70811736', '70926424'],
                ['chr3', '63951571', '64066259']]
    dataset = RD_dataloader(
                    samples,
                    reference_genome_file,
                    sequence_data_file,
                    target_seq_file
                )
    
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    data_iter = iter(data_loader)
    print(next(data_iter))

    # for i, data in enumerate(data_loader):
    #     print(data[0].shape)
    #     print(data[1].shape)
    #     print(data[2].shape)
    #     print('----' * 30)