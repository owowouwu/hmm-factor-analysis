# from ysfoo

sim.sfm <- function(lmat=NULL, tauvec=NULL, fmat=NULL, zmat=NULL, alphavec=NULL,
                    G=20, N=10, K=4, pivec=NULL, snr=NULL,
                    taushape=100, taurate=1, alphashape=1, alpharate=1) {
  # infer dimensions
  if(!is.null(lmat)) {
    ldim <- dim(lmat)
    G <- ldim[1]
    K <- ldim[2]
  }
  if(!is.null(tauvec)) G <- length(tauvec)
  if(!is.null(fmat)) {
    fdim <- dim(fmat)
    K <- fdim[1]
    N <- fdim[2]
  }
  if(!is.null(zmat)) {
    zdim <- dim(zmat)
    G <- zdim[1]
    K <- zdim[2]
  }
  if(!is.null(alphavec)) K <- length(alphavec)
  if(!is.null(pivec)) K <- length(pivec)
  
  # simulate parameters
  if(is.null(lmat)) {
    if(is.null(zmat)) {
      # first half of factors are dense, second half are sparse
      if(is.null(pivec)) pivec <- c(rep(0.9, K %/% 2), rep(0.1, K - K %/% 2))
      zmat <- sapply(pivec, simulate.z.col, G=G)
    }
    if(is.null(alphavec)) alphavec <- rgamma(K, alphashape, alpharate)
    lmat <- zmat * sapply(alphavec, function(alpha) rnorm(G, 0, 1 / sqrt(alpha)))
  }
  if(is.null(fmat)) fmat <- matrix(rnorm(K * N), nrow=K, ncol=N)
  
  lf <- lmat %*% fmat
  
  # simulate tau
  if(is.null(tauvec)) {
    if(is.null(snr)) tauvec <- rgamma(G, taushape, taurate)
    else {
      tauvec <- snr / matrixStats::rowVars(lf)
      tau.na <- !is.finite(tauvec)
      tauvec[tau.na] <- rgamma(sum(tau.na), taushape, taurate)
    }
  }
  
  # simulate y
  ymat <- lf + t(sapply(tauvec, function(tau) rnorm(N, 0, 1 / sqrt(tau))))
  
  return (list(ymat=ymat, lmat=lmat, fmat=fmat, zmat=zmat, tauvec=tauvec, alphavec=alphavec))
}

plot_faceted_zmat <- function(zmat_list, G, K) {
  # Create a combined data frame with an additional column for faceting (t)
  zmat_combined <- do.call(rbind, lapply(seq_along(zmat_list), function(t) {
    zmat_t <- zmat_list[[t]]
    # Convert each matrix into a long-format data frame
    expand.grid(factor = 1:K, feature = 1:G) %>%
      transform(value = as.vector(t(zmat_t)), t = as.factor(t))
  }))
  
  # Generate the faceted level plot
  levelplot(value ~ factor * feature | t, data=zmat_combined,
            at=seq(0, 1, 0.05), aspect=2, col.regions=viridis(100),
            ylim=0.5 + c(G, 0), scales=list(y=list(at=c())),
            colorkey=list(width=1),
            main=list(label="Connectivity matrix (truth)\n", cex=1),
            xlab="Factors", ylab="Features",
            par.settings=list(layout.heights=list(axis.top=0.5)),
            layout=c(5, ceiling(length(zmat_list)/5)))  # Customize the layout as needed
}
