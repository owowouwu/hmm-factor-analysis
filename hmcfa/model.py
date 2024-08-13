import numpy as np
import scipy as sp
from typing import Dict, List, Union

class HiddenMarkovFA:

    def __init__(self, data: np.ndarray, n_factors: int,
                 **kwargs
                 ):
        """
        :param data: data array of shape (genes x samples x time)
        :param n_factors: number of factors to use
        """
        self.Y = data
        self.K = n_factors
        self.T, self.G, self.N = data.shape
        self._parse_hyperparameters(**kwargs)
        self._init_variational_params()
        self.mask = ~np.eye(self.K, dtype = bool)


    def _parse_hyperparameters(self, **kwargs):
        # TODO
        self.a_alpha_prior = 1
        self.b_alpha_prior = 1
        self.a_tau_prior = 1
        self.b_tau_prior = 1
        self.pi = np.repeat(0.5, self.K)
        self.dirchlet_prior_0 = 1
        self.dirchlet_prior_1 = 1
    def _init_variational_params(self):
        self.mu_F = sp.stats.norm.rvs(size=(self.T, self.K, self.N))
        self.sigma2_F = sp.stats.gamma.rvs(a=1,scale=1,size=(self.T, self.K, self.N))
        self.mu_L = sp.stats.norm.rvs(size=(self.T, self.G, self.K))
        self.sigma2_L = sp.stats.gamma.rvs(a=1,scale=1,size=(self.T, self.G, self.K))
        self.a_tau = sp.stats.gamma.rvs(a=1,scale=1,size=(self.T, self.G))
        self.b_tau = sp.stats.gamma.rvs(a=1,scale=1,size=(self.T, self.G))
        self.a_alpha = sp.stats.gamma.rvs(a=1,scale=1,size=(self.T, self.K))
        self.b_alpha = sp.stats.gamma.rvs(a=1,scale=1,size=(self.T, self.K))
        self.dirichlet = sp.stats.gamma.rvs(a=1,scale=1,size=(self.K, 2))
        self.eta = sp.stats.beta.rvs(a=1,b=1,size=(self.T, self.G, self.K))
        self.pairwise = sp.stats.beta.rvs(a=1,b=1,size=(self.T - 1, self.G, self.K, 2))


    def variational_likelihood(self, z, g, k, t):

        pass

    def update_L(self):
        # first update sigma
        second_moment = self.mu_F ** 2 + self.sigma2_F
        sum_second_moment = second_moment.sum(axis = 2)
        a_over_b_alpha = self.a_alpha / self.b_alpha
        a_over_b_tau = self.a_tau / self.b_tau

        sigma2_L_new = a_over_b_alpha[:, np.newaxis, :] + np.einsum('td,tk->tdk', a_over_b_tau, sum_second_moment)
        sigma2_L_new = 1 / self.sigma2_L

        # update mu
        Y_mu = np.einsum('tgn,tkn->tgk', self.Y, self.mu_F)
        # cursed outer product operations
        outer_product_mu_eta = np.einsum('tdj,tdj,tin,tjn->tdijn', self.eta, self.eta, self.mu_F, self.mu_F)
        outer_product_sum_offdiag = np.einsum('tdijn,ij->tdin', outer_product_mu_eta, self.mask)
        outer_product_summed_n = outer_product_sum_offdiag.sum(axis = -1)

        mu_L_new = np.einsum('td,tdk->tdk', a_over_b_tau, sigma2_L_new*(Y_mu - outer_product_summed_n))
        return mu_L_new, sigma2_L_new

    def update_F(self):

        second_moment_L = self.mu_L**2 + self.sigma2_L
        a_over_b_tau = self.a_tau / self.b_tau
        sigma2_F_new = 1 / (np.einsum('td,tdk->tk', a_over_b_tau, self.eta * second_moment_L) + 1)
        # broadcast this
        sigma2_F_new = np.repeat(sigma2_F_new[:, :, np.newaxis], self.N, axis = 2)

        # more cursed outer product operations
        outer_product_mu_eta = np.einsum('tdj,tdi,tdi,tdj->tdji', self.eta, self.eta, self.mu_L, self.mu_L)
        outer_prod_weighted = np.einsum('td,tdij->tdij', a_over_b_tau, outer_product_mu_eta)
        outer_prod_weighted_sum_genes = outer_prod_weighted.sum(axis=1)
        sum_fk_Dtau_lk_bar = np.einsum('tnki,ki->tnk', outer_prod_weighted_sum_genes, self.mask)[0, 0, :]

        y_Dtau


        mu_F_new = sigma2_F_new

        return mu_F_new, sigma2_F_new