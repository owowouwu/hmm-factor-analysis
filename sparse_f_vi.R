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

S <- 10 # sparse factors



save.freq <- 100 # how often are variational factors saved
best.elbo <- -Inf
for(seed in 21:30) {
  vi.res <- cavi(mat, rep(0.1, S), 0.001, 0.001, 0.001, 0.001,
                 save=save.freq, max_iter=10000, seed=seed)
  
  n.iter <- tail(vi.res$iter, 1)
  idx <- n.iter / save.freq
  elbo <- vi.res$elbo[idx]
  elbo.change <- elbo - vi.res$elbo[idx-1]
  print(paste0('Trial ', seed-20, ': ', n.iter, ' iterations, ELBO = ', elbo))
  print(paste0('ELBO increased by ', elbo.change, ' in the last ', save.freq, ' iterations'))
  if(elbo > best.elbo) {
    best.elbo <- elbo
    best.seed <- seed
  }
  
  saveRDS(vi.res, sprintf("output/cavi_%s_s_%s.rds", stage, seed))
}
print(paste0('Trial with highest ELBO is trial ', best.seed-20))