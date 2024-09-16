source('scripts/utils.R')
library(viridis)
library(lattice)
library(rhdf5)
library(dplyr)
library(abind)
set.seed(2)
# simulate data according to Section 5.1 of report
S <- 5 # sparse factors
D <- 1 # dense factors
K <- S + D
N <- 100
G <- 800
T <- 5
output_dir <- 'data/synthetic/no_change'



ymat_list <- list()
lmat_list <- list()
zmat_list <- list()
tauvec_list <- list()
alphavec_list <- list()

for (t in 1:T) {
  # Run the simulation for each t
  data_t <- sim.sfm(K=K, N=N, G=G,
                    zmat = matrix(c(rep(0, 30 * G / 40), rep(1, 1 * G / 40), rep(0, 5 * G / 40), rep(1, 1 * G / 40), rep(0, 2 * G / 40), rep(1, 1 * G / 40),
                                    rep(0, 6 * G / 40), rep(1, 1 * G / 40), rep(0, 13 * G / 40), rep(1, 1 * G / 40), rep(0, 1 * G / 40), rep(1, 1 * G / 40), rep(0, 13 * G / 40), rep(1, 3 * G / 40), rep(0, 1 * G / 40),
                                    rep(1, 1 * G / 40), rep(0, 4 * G / 40), rep(1, 1 * G / 40), rep(0, 14 * G / 40), rep(1, 2 * G / 40), rep(0, 8 * G / 40), rep(1, 6 * G / 40), rep(0, 4 * G / 40),
                                    rep(1, 5 * G / 40), rep(0, 15 * G / 40), rep(1, 10 * G / 40), rep(0, 10 * G / 40),
                                    rep(1, 20 * G / 40), rep(0, 20 * G / 40),
                                    rep(1, G)),
                                  nrow=G, ncol=K),
                    alphavec=rep(1, K), snr=5)
  
  # Concatenate results over t
  ymat_list[[t]] <- data_t$ymat
  lmat_list[[t]] <- data_t$lmat
  zmat_list[[t]] <- data_t$zmat
  tauvec_list[[t]] <- data_t$tauvec
  alphavec_list[[t]] <- data_t$alphavec
}

# Concatenate all matrices
ymat_3d <- abind::abind(ymat_list, along = 3)
lmat_3d <- abind::abind(lmat_list, along = 3)
zmat_3d <- abind::abind(zmat_list, along = 3)
tauvec_combined <- abind::abind(tauvec_list, along = 2)  # assuming tauvec is a vector
alphavec_combined <- abind::abind(alphavec_list, along = 2)  # assuming alphavec is a vector


# Save results to the specified directory in Arrow format
output_dir <- normalizePath(output_dir, mustWork = FALSE)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}


h5file <- file.path(output_dir, "arrays.h5")
h5createFile(h5file)

# Write each matrix to the HDF5 file
h5write(ymat_3d, h5file, "ymat")
h5write(lmat_3d, h5file, "lmat")
h5write(zmat_3d, h5file, "zmat")

# Write each vector to the HDF5 file
h5write(tauvec_combined, h5file, "tauvec")
h5write(alphavec_combined, h5file, "alphavec")  

H5close()

png(filename = file.path(output_dir, "connectivities.png"), 
    width = 1500, 
    height = 1000, 
    res = 250)
plot_faceted_zmat(zmat_list, G, K)
dev.off()
