#!/usr/bin/env python
# Copyright 2025 Z Zhang

# MambaHM, Version 1.0;
# you may not use this file except in compliance with the License.
# Use of this code requires following originality guidelines
# and declaring the source of the code.
# email:zhichunli@mail.dlut.edu.cn
# =========================================================================
import os
import subprocess

def plotHeatMap(outdir, point_type='TSS', max_dict=100):
    png_out = os.path.join(outdir, 'png')
    if not os.path.exists(png_out):
        os.makedirs(png_out)
    files = os.listdir(outdir)
    for file in files:
        if os.path.splitext(file)[1].lower() in ['.gz']:
            histone = os.path.splitext(file)[0][4:].lower()
            try:
                max = max_dict[histone]
            except:
                continue
            gz_file = os.path.join(outdir, file)
            out_png = os.path.join(png_out, os.path.splitext(file)[0]+'.pdf')

            cmd_plot = [
                'plotHeatmap',
                # '--colorMap', 'RdYlBu_r',
                '--colorList', '#2369AC, #92C3D7, #FFFFFF, #FCBC98, #ED4B3A',
                # '--whatToShow', 'heatmap and colorbar',
                '--heatmapHeight', str(15),
                '--zMax', str(max),
                '-m',  gz_file,
                '-out', out_png
            ]
            subprocess.call(cmd_plot)



def computeMatrix(bigwig_dir, outdir, gene_file, extend=1000, process=15, point_type='TSS'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    bw_files = os.listdir(bigwig_dir)

    for file in bw_files:
        if os.path.splitext(file)[1].lower() in ['.bw','.bigwig']:
            print(file)
            bw_file = os.path.join(bigwig_dir, file)
            out_gz = os.path.join(outdir, point_type + '_' + os.path.splitext(file)[0] + '.gz')
            out_bed = os.path.join(outdir, point_type + '_' + os.path.splitext(file)[0] + '.bed')
            
            # computeMatrix
            cmd_matrix = ['computeMatrix',
                          'reference-point',
                          '--referencePoint', point_type,
                          '--skipZeros',
                          '-p', str(process),
                          '-b', str(extend), '-a', str(extend),
                          '-R', gene_file,
                          '-S', bw_file,
                          '-o', out_gz,
                          '--outFileSortedRegions', out_bed]
            subprocess.call(cmd_matrix)


if __name__ == '__main__':
    extend = 5000

    gene_file = 'MambaHM/genome_regions/FunctionalElements/GM12878/tss_length_1.bed'

    bigwig_dir = 'bigwig_dir'
    outdir = 'outdir'

    computeMatrix(bigwig_dir, outdir, gene_file, extend)




