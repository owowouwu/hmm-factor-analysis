renv::activate()
library(sparsefactor)
library(tidyverse)
library(data.table)
args <- commandArgs(trailingOnly = TRUE)

stage <- args[1]
if (!(stage %in% c("ips", "defendo", "mesendo"))){
  stop("Invalid Cell Stage")
}


X <- fread('data/iPS/log_normalised_counts.csv.gz')
genes <- X$V1
X <- X[,-1]
cells <- names(X)
rownames(X) <- genes
cell_diff <- read.csv('data/iPS/cell_diff_stages.tsv', sep='\t')

mat <- X %>%
  subset(select = filter(cell_diff, cell_differentiation == stage)$cell) %>%
  as.matrix()

S <- 5 # sparse factors

for(seed in 11:15) {
  samples <- gibbs(3000, mat, rep(0.1, S),
                   0.001, 0.001, 0.001, 0.001,
                   thin=10, burn_in=2000, seed=seed)
  # undo sign-switching and label-switching within chain
  samples <- relabel_samples(samples)
  # save chain
  saveRDS(samples, sprintf("output/gibbs_%s_s_%s.rds", stage, seed))
}



