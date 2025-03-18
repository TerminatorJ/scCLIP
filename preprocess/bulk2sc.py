
from torch.utils.data import Dataset
import scanpy as sc
import pandas as pd
import numpy as np
import zarr
from gcell.rna.gencode import Gencode
from pyranges import PyRanges as pr
from typing import Dict, List, Union
import sys
import os
import shutil
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'get_model', 'tutorials')))

from preprocess_utils import (
    add_exp,
    create_peak_motif,
    get_motif,
    query_motif,
)
print("loading all the packages")


class Multiomics(Dataset):
    """
    This is a class for the multiomics dataset.
    
    """
    def __init__(
            self,
            ad: sc.AnnData,
            assembly='hg38',
            version='40',
            extend_bp=300,
            motif_bed_path="/scratch/project_465001820/scCLIP/dataset/motif/hg38.archetype_motifs.v1.0.bed.gz",
            tmp_path="/scratch/project_465001820/scCLIP/dataset/processed_pbmc_dataset",
            keep_zarr=True
    ):
        """
        Args:
            ad:  Full AnnData object with all cells
            assembly (str): Genome assembly (e.g., 'hg38', 'mm10').
            version (int): Version of the genome assembly.
            extend_bp (int): Number of base pairs to extend.
            motif_bed_path (str): Path to the motif bed file.
            tmp_path (str): Temporary path for saving bed files.
            keep_zarr (bool): Flag to keep zarr data.
        """
        super().__init__()
        self.ad = ad
        self.tmp_path = tmp_path
        self.motif_bed_path = motif_bed_path
        self.keep_zarr = keep_zarr
        self.extend_bp = extend_bp
        #get the atac data from the scanpy object
        self.atac = self._get_atac()
        #get the rna data from the multiomics scanpy object
        self.rna = self._get_rna()
        #filter the data
        self.rna, self.atac = self.data_filter()
        #initialize the Genecode object
        self.gencode = Gencode(assembly=assembly, version=version)  # The gencode include the promotor information

    def __repr__(self) -> str:
        return f"-------Multiomics dataset with cell barcode {self.cell_barcode}, containing {self._get_gene_num} genes, {self._get_peak_num} peaks, and a library size of {self._get_libsize}-------"
    


    @property
    def _get_gene_num(self) -> int:
        import pdb; pdb.set_trace()
        gene_num = np.sum(self.rna[self.rna.obs.index == self.cell_barcode, :].X > 0)
        return gene_num

    @property
    def _get_peak_num(self) -> int:
        import pdb; pdb.set_trace()
        peak_num = np.sum(self.atac[self.atac.obs.index == self.cell_barcode, :].X > 0)
        return peak_num
    
    @property
    def _get_libsize(self) -> int:
        import pdb; pdb.set_trace()
        libsize = int(self.atac[self.atac.obs.index == self.cell_barcode, :].X.sum())
        return libsize
    
    def data_filter(self, min_genes=1000, min_cells=5, libsize=10000):
        """
        Filter the data by gene number, peak number, and library size.

        Args:
            min_genes (int): Minimum number of genes for filtering.
            peak_num (int): Number of peaks.
            libsize (int): Library size.
        """
        # Filter cells based on ATAC data
        atac_filter = np.array(np.asarray(self.atac.X.sum(axis=1)).flatten() >= libsize)

        # Filter cells based on RNA data
        rna_filter = np.array(np.asarray((self.rna.X > 0).sum(axis=1)).flatten() >= min_genes)

        # Combine filters to get cells that meet both criteria
        combined_filter = atac_filter & rna_filter

        # Subset the AnnData object to include only the filtered cells
        self.rna = self.rna[combined_filter]
        self.atac = self.atac[combined_filter]
        sc.pp.filter_genes(self.rna, min_cells=min_cells)
        sc.pp.filter_genes(self.atac, min_cells=min_cells)
        #filter the MT genes
        mt_genes_filter = self.rna.var_names.str.startswith("MT-")
        self.rna = self.rna[:,~mt_genes_filter]
        

        return self.rna, self.atac
    
    def _get_barcode(self, index: int) -> str:
        """
        Get the cell barcode from the index.

        Args:
            index (int): Index of the item.

        Returns:
            str: Cell barcode.
        """
        self.cell_barcode = self.atac.obs.index[index]
        return self.cell_barcode
    
    def __getitem__(self, index: int) -> Dict[str, Dict[str, Union[float, List]]]:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            Dict[str, Dict[str, Union[str, List]]], 
            {"gene_name": {"TPM": float, "motifs": ["motif1", "motif2"], "motifs_value": [0.1, 0.2]}}
        """



        self.cell_barcode = self._get_barcode(index)
        
        self.peak_names = self._get_peak_for_sc()
        #save the bed file
        self._save_bed() #self.bed_file will be activated
        #get the peak x motif matrix
        sc_peak_motif = self.get_sc_peak_motif()
        #get the peak x gene expression
        sc_peak_exp_pro = self.get_sc_peak_exp(extend_bp=self.extend_bp)
        #get the motif x gene expression
        gene_motif_exp = self.get_motif_exp(sc_peak_motif, sc_peak_exp_pro)
        import pdb; pdb.set_trace()
        return gene_motif_exp
    

    def _save_bed(self):
        """
        Save the dataframe to a bed file.
        """
        assert self.peak_names is not None, "Please run the _get_peak_for_sc method first"
        peaks = pr(self.peak_names[['Chromosome', 'Start', 'End', 'aCPM']], int64=True).sort().df
        self.bed_file = f"{self.tmp_path}/{self.cell_barcode}.atac.bed"
        peaks.to_csv(self.bed_file, sep="\t", header=False, index=False)
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        #the number of cells
        return self.atac.shape[0]

    
    def _get_atac(self):
        """
        Getting the atac data from the multiomics scanpy object
        """
        atac = self.ad[:, np.where(self.ad.var.feature_types == "Peaks")[0]]
        return atac
    
    def _get_rna(self):
        """
        Getting the rna data from the multiomics scanpy object
        """
        rna = self.ad[:, np.where(self.ad.var.feature_types == "Gene Expression")[0]]
        return rna
    
    def to_tpm(self, X):
        """
        Convert the raw count to aCPM.
        """
        acpm = np.log10(X / X.sum() * 1e5 + 1) #normalization
        acpm = acpm/acpm.max()
        return acpm
    
    
    def _get_peak_for_sc(self, aCPM_cutoff=0.5):
        """
        Get the peak names from the scanpy object with only single cell.
        Args:
            aCPM_cutoff (float): Cutoff for aCPM filtering.

        Returns:
            peak_names: pandas DataFrame with the peak names. 'Chromosome', 'Start', 'End', 'aCPM', 'Name'
        """
        #getting the single cell anndata object
        sc_atac = self.atac[self.atac.obs.index == self.cell_barcode, :]
        peak_names = pd.DataFrame(sc_atac.var.index.str.split('[:-]').tolist(), columns=['Chromosome', 'Start', 'End'])
        peak_names['Start'] = peak_names['Start'].astype(int)
        peak_names['End'] = peak_names['End'].astype(int)
        
        peak_names['aCPM'] = self.to_tpm(np.asarray(sc_atac.X.todense()).flatten())#to one dimension
        # only the peak names with the aCPM > aCPM_cutoff percentile
        peak_names = peak_names[peak_names["aCPM"] > 0]
        peak_names = peak_names[peak_names['aCPM'] > peak_names['aCPM'].quantile(aCPM_cutoff)]
        # filter out the peaks with chrM, chrY, chrUn
        peak_names = peak_names.query('Chromosome.str.startswith("chr") & ~Chromosome.str.endswith("M") & ~Chromosome.str.endswith("Y") & ~Chromosome.str.startswith("chrUn")')
        #create the peak name for retrieving
        peak_names["Name"] = peak_names.apply(
            lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
        )
        return peak_names
    
    
    def get_sc_peak_motif(self):
        """
        Get the single cell peak x motif matrix.

        Returns:
            sc_peak_motif: pandas DataFrame with the single cell peak x motif matrix
        """
        #get the peak x motif matrix by the following command
        zarr_path = f"{self.tmp_path}/{self.cell_barcode}.zarr"
        
        if not os.path.exists(zarr_path):
            print("running query motif...(~35s)")
            peaks_motif = query_motif(self.bed_file, self.motif_bed_path, self.tmp_path, self.cell_barcode) #filtering the motifs that are overlap with the peak 
            print("running get motif...(~216s)")
            peak_motif_rawpath = get_motif(self.bed_file, peaks_motif, self.tmp_path, self.cell_barcode) #overlapping the motif according to the chromosomes
            print("saving to the zarr file...")
            create_peak_motif(peak_motif_rawpath, zarr_path, self.bed_file) # all cell will later be added to the same zarr file as we use the same peak set.
            #delete the tmp files
            if os.path.exists(peaks_motif):
                os.remove(peaks_motif)
            if os.path.exists(peak_motif_rawpath):
                os.remove(peak_motif_rawpath)
            if os.path.exists(self.bed_file):
                os.remove(self.bed_file)
        else:
            print(f"Using existing peak x motif matrix for {self.cell_barcode}")

        
        
        # os.remove(f"{self.tmp_path}/tmp.zarr")  # Clean up the temporary zarr file
        # query the peaks with the reference peak x motif matrix
        # fetch the subset peaks x motifs according to the sc_peaks
        self.zarr_file = zarr.open(zarr_path, mode="r")
        sc_peak_motif = self.zarr_file["data"][:]
        sc_peak_names = self.zarr_file["peak_names"][:]
        sc_motif_names = self.zarr_file["motif_names"][:]
        #get the peak x motif dataframe
        self.sc_peak_motif_df = pd.DataFrame(sc_peak_motif, index=sc_peak_names, columns=sc_motif_names)
        if not self.keep_zarr:
            shutil.rmtree(zarr_path)
        return self.sc_peak_motif_df

    
    def get_sc_peak_exp(self, extend_bp=300, exp_cutoff=0.25):
        """
        Get the single cell peak x motif matrix.

        Returns:
            sc_peak_exp_pro: pandas DataFrame with the single cell peak x expression within the promoter regions
        """
        # Read RNA data
        sc_rna = self.rna[self.rna.obs.index == self.cell_barcode, ]
        # normalization the gene expression
        sc.pp.normalize_total(sc_rna, target_sum=1e4)
        sc.pp.log1p(sc_rna)
        #to dataframe
        gene_exp = pd.DataFrame({"gene_name": np.asarray(sc_rna.var.index), "Exp": np.asarray(sc_rna.X.todense()).flatten()})
        #only select the gene with Exp > Exp quantile(0.25) in default
        
        gene_exp = gene_exp[gene_exp['Exp'] > 0]
        gene_exp = gene_exp[gene_exp['Exp'] > gene_exp['Exp'].quantile(exp_cutoff)]
        #assign expression level to each gene
        promoter_exp = pd.merge(
            self.gencode.gtf, gene_exp, left_on="gene_name", right_on="gene_name"
        )
        #prepare for join
        self.peak_names = pr(self.peak_names.reset_index(), int64=True)
        
        #join the peak with gtf+exp
        peak_exp = self.peak_names.join(pr(promoter_exp, int64=True).extend(extend_bp), how="left").as_df() 
        #only select the peaks in the promoter regions
        self.sc_peak_exp_pro = peak_exp.query('index_b!=-1')[['index', 'gene_name', 'Strand', 'Name', 'Exp']]
        # Process expression data
        return self.sc_peak_exp_pro
        
    
    def get_motif_exp(self, sc_peak_motif, sc_peak_exp_pro):
        """
        Add the single cell gene expression to the promoter regions via Genecode gft file 
        before intersecting the peaks to get the motifs.

        Returns:
            desired_dict: dict with gene_name as key and {"TPM": float, "motifs": ["motif1", "motif2"], "motifs_value": [0.1, 0.2]} as value
        """
        #merge two df on "Name"
        merged_df = sc_peak_motif.merge(sc_peak_exp_pro, left_index=True, right_on="Name")

        # Group by `gene_name`
        grouped = merged_df.groupby(['gene_name', "Strand"], as_index=False)

        # Create a function to filter out columns with all zeros and calculate the mean of `Exp`
        def process_group(group):
            # Drop columns with all zeros
            group = group.loc[:, (group != 0).any(axis=0)]
            # Calculate mean of `Exp`
            mean_exp = group['Exp'].mean()
            # Create a new DataFrame with one row and the necessary columns
            result = group.iloc[[0]].copy()
            result['Mean_exp'] = mean_exp
            motifs_clusters = group.columns[(group != 0).any(axis=0) & np.isin(group.columns, sc_peak_motif.columns)]
            result['motifs_clusters'] = "|".join(list(motifs_clusters))
            #get the motifs number
            result["cluster_number"] = len(motifs_clusters)
            result = result[["gene_name", "Mean_exp", "motifs_clusters", "cluster_number", "Strand", "Name"]]
            return result
        # Apply the function to each group and reset the index
        self.gene_motif_exp = grouped.apply(process_group).reset_index(drop=True)

        return self.gene_motif_exp
    

if __name__ == "__main__":
    #load the multiomics data
    ad = sc.read_10x_h5("/scratch/project_465001820/scCLIP/dataset/pbmc_multiomics/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5", gex_only=False)
    #initialize the Multiomics object
    multiomics = Multiomics(ad)
    #get the length of the dataset
    print(len(multiomics))
    #get the first item
    print(multiomics[1])
    #get the representation of the dataset
    print(multiomics)
    



