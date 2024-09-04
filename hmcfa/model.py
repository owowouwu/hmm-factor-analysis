import numpy as np
import scipy as sp
import scipy.stats
import warnings
from tqdm import tqdm
from scipy.special import digamma, gamma
from typing import Dict, List, Union
from functools import cached_property
from . import get_logger

# useful constants to precompute
LOG_TWOPI = np.log(2 * np.pi)


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
        self._updated_a_tau = False
        self.mask = ~np.eye(self.K, dtype=bool)
        self.I = np.eye(self.K, dtype=bool)
        self.logger = get_logger()

    def _parse_hyperparameters(self, **kwargs):
        # TODO properly
        self.a_alpha_prior = np.full(shape=(self.T, self.K), fill_value=1)
        self.b_alpha_prior = np.full(shape=(self.T, self.K), fill_value=1)
        self.a_tau_prior = np.full(shape=(self.T, self.G), fill_value=1)
        self.b_tau_prior = np.full(shape=(self.T, self.G), fill_value=1)
        self.pi = np.log(np.full(shape=(self.K, 2), fill_value=0.5))
        self.pi_extended = self.pi[np.newaxis, :, :]  # for later
        self.dirchlet_A_prior = np.full(shape=(self.K, 2), fill_value=1)

    def _init_variational_params(self):
        self.mu_F = sp.stats.norm.rvs(size=(self.T, self.K, self.N))
        self.sigma2_F = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.K, self.N))
        self.mu_L = sp.stats.norm.rvs(size=(self.T, self.G, self.K))
        self.sigma2_L = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.G, self.K))
        self.a_tau = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.G))
        self.b_tau = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.G))
        self.a_alpha = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.K))
        self.b_alpha = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.K))
        self.dirchlet_A = sp.stats.gamma.rvs(a=1, scale=1, size=(self.K, 2, 2))
        self.A_variational = np.full(shape=(self.K, 2, 2), fill_value=0.5)
        self.A_variational_to_add = self.A_variational[np.newaxis, :, :, :]
        self.variational_likelihoods = np.zeros(shape=(2, self.T, self.G, self.K))
        self.eta = sp.stats.beta.rvs(a=1, b=1, size=(self.T, self.G, self.K))
        self.pairwise = sp.stats.beta.rvs(a=1, b=1, size=(self.T - 1, self.G, self.K, 2, 2))
        self.normalisations = np.zeros(shape = (self.T, self.G, self.K))

    # cached intermediate terms

    @cached_property
    def weighted_l(self):
        return self.eta * self.mu_L

    @cached_property
    def a_over_b_alpha(self):
        return self.a_alpha / self.b_alpha

    @cached_property
    def a_over_b_tau(self):
        return self.a_tau / self.b_tau

    @cached_property
    def second_moment_F(self):
        return self.mu_F ** 2 + self.sigma2_F
    @cached_property
    def second_moment_L(self):
        return self.mu_L ** 2 + self.sigma2_L
    @cached_property
    def mixed_moment_F(self):
        outer_prod_mu_f = np.einsum('tkn,tin->tkin', self.mu_F, self.mu_F)
        masked_sigma2_f = np.einsum('tkn,ki->tkin', self.sigma2_F, self.I)
        mixed_moment_f = outer_prod_mu_f + masked_sigma2_f
        return mixed_moment_f
    @cached_property
    def mixed_moment_L(self):
        outer_prod_mu_l = np.einsum('tgk,tgi->tgki', self.mu_L, self.mu_L)
        masked_sigma2_l = np.einsum('tgk,ki->tgki', self.sigma2_L, self.I)
        mixed_moment_l = outer_prod_mu_l + masked_sigma2_l
        return mixed_moment_l


    # Baum Welsh steps
    def V_step(self):
        """
        Variational HMM E step - compute 'expected' transition probs and pairwise probs
        :return:
        """

        # 'expected' transition probs
        # keep in logspace
        A_variational_new = digamma(self.dirchlet_A) - digamma(self.dirchlet_A.sum(axis=2))[:, :, np.newaxis]

        # likelihoods

        # conditional mean
        mu_bar = np.einsum('tgk,ki->tgk', self.weighted_l, self.mask)
        mu_yd_conditional_z = self.mu_L + mu_bar
        mu_yd_conditional_z = np.stack([mu_bar, mu_yd_conditional_z], axis=0)
        mu_yd_conditional_z = np.einsum('ztgk,tkn->ztgn', mu_yd_conditional_z, self.mu_F)  # shape (2, T, G, N)

        # just repeat
        covar = np.repeat(self.a_over_b_tau[:, :, np.newaxis], axis=2, repeats=self.N)
        sigmas = np.sqrt(covar)

        # conditional likelihood z = 0
        cond_likelihood_0 = sp.stats.norm.logpdf(self.Y, loc=mu_yd_conditional_z[0], scale=sigmas)
        # conditional likelihood z = 1
        cond_likelihood_1 = sp.stats.norm.logpdf(self.Y, loc=mu_yd_conditional_z[1], scale=sigmas)

        cond_ll = np.stack([cond_likelihood_0, cond_likelihood_1], axis=0)

        return A_variational_new, cond_ll

    def M_step(self):
        forward, backward = self.forward_messages(), self.backward_messages()
        q_z = np.exp(forward + backward)
        pairwise_new = np.zeros(self.pairwise.shape)
        for t in range(1, self.T):
            pairwise_new[t - 1] = (
                    forward[t - 1, :, :, np.newaxis, np.newaxis] +
                    self.A_variational_to_add[np.newaxis, :, :, :, :] +
                    self.variational_likelihoods[np.newaxis, :, t - 1, :, :, np.newaxis]
            )

        # normalise to ensure these are probabilities
        normalisation_constants = q_z.sum(axis = -1)
        q_z = q_z / normalisation_constants[:,:,:,np.newaxis]

        # eta is q_z = 1 / expected value over hidden states
        eta_new = q_z[:,:,:,1]
        pairwise_normalisations = pairwise_new.sum(axis = (-1,-2))
        pairwise_new = pairwise_new / pairwise_normalisations[:,:,:,np.newaxis,np.newaxis]

        return eta_new, pairwise_new, normalisation_constants

    def forward_messages(self):
        forward = np.ones(shape=(self.T, self.G, self.K, 2))
        forward[0, :, :, :] = self.pi_extended
        for t in range(1, self.T):
            forward[t, :, :, 0] = np.logaddexp(
                forward[t - 1, :, :, 0] + self.A_variational_to_add[:, :, 0, 0] + self.variational_likelihoods[0, t, :,
                                                                                  :],
                forward[t - 1, :, :, 1] + self.A_variational_to_add[:, :, 1, 0] + self.variational_likelihoods[0, t, :,
                                                                                  :]
                )
            forward[t, :, :, 1] = np.logaddexp(
                forward[t - 1, :, :, 0] + self.A_variational_to_add[:, :, 0, 1] + self.variational_likelihoods[1, t, :,
                                                                                  :],
                forward[t - 1, :, :, 1] + self.A_variational_to_add[:, :, 1, 1] + self.variational_likelihoods[1, t, :,
                                                                                  :]
                )

        return forward

    def backward_messages(self):
        backward = np.ones(shape=(self.T, self.G, self.K, 2))
        backward[self.T, :, :, :] = 1
        # iterate backwards
        for t in range(self.T - 1, -1, -1):
            backward[t, :, :, 0] = np.logaddexp(
                backward[t + 1, :, :, 0] + self.A_variational_to_add[:, :, 0, 0] + self.variational_likelihoods[0,
                                                                                   t + 1, :, :],
                backward[t + 1, :, :, 1] + self.A_variational_to_add[:, :, 0, 1] + self.variational_likelihoods[1,
                                                                                   t + 1, :, :]
                )
            backward[t, :, :, 1] = np.logaddexp(
                backward[t + 1, :, :, 0] + self.A_variational_to_add[:, :, 1, 0] + self.variational_likelihoods[0,
                                                                                   t + 1, :, :],
                backward[t + 1, :, :, 1] + self.A_variational_to_add[:, :, 1, 1] + self.variational_likelihoods[1,
                                                                                   t + 1, :, :]
                )

        return backward

    # VI updates

    def update_L(self):
        # first update sigma
        sum_second_moment_F = self.second_moment_F.sum(axis=2)

        sigma2_L_new = self.a_over_b_alpha[:, np.newaxis, :] + np.einsum('td,tk->tdk', self.a_over_b_tau,
                                                                         sum_second_moment_F
                                                                         )
        sigma2_L_new = 1 / sigma2_L_new

        # update mu
        Y_mu = np.einsum('tgn,tkn->tgk', self.Y, self.mu_F)
        # cursed outer product operations
        outer_product_mu_eta = np.einsum('tdj,tdj,tin,tjn->tdijn', self.eta, self.eta, self.mu_F, self.mu_F)
        outer_product_sum_offdiag = np.einsum('tdijn,ij->tdin', outer_product_mu_eta, self.mask)
        outer_product_summed_n = outer_product_sum_offdiag.sum(axis=-1)

        mu_L_new = np.einsum('td,tdk->tdk', self.a_over_b_tau, sigma2_L_new * (Y_mu - outer_product_summed_n))
        return mu_L_new, sigma2_L_new

    def update_F(self):
        sigma2_F_new = 1 / (np.einsum('td,tdk->tk', self.a_over_b_tau, self.eta * self.second_moment_L) + 1)
        # broadcast this
        sigma2_F_new = np.repeat(sigma2_F_new[:, :, np.newaxis], self.N, axis=2)

        # more cursed outer product operations
        outer_product_mu_eta = np.einsum('tdj,tdi,tdi,tdj->tdji', self.eta, self.eta, self.mu_L, self.mu_L)
        outer_prod_weighted = np.einsum('td,tdij->tdij', self.a_over_b_tau, outer_product_mu_eta)
        outer_prod_weighted_sum_genes = outer_prod_weighted.sum(axis=1)
        sum_fk_Dtau_lk_bar = np.einsum('tnki,ki->tkn', outer_prod_weighted_sum_genes, self.mask)[0, 0, :]

        y_Dtau_l = np.einsum('tgn,tg,tgk->tkn', self.Y, self.a_over_b_tau, self.weighted_l)

        mu_F_new = sigma2_F_new * (y_Dtau_l - sum_fk_Dtau_lk_bar)

        return mu_F_new, sigma2_F_new

    def update_tau(self):
        if not self._updated_a_tau:
            a_tau_new = self.a_tau_prior + (self.N / 2.)
        else:
            a_tau_new = self.a_tau

        y_dot_y = np.einsum('tgn,tgn->tg', self.Y, self.Y)
        lbar_F_y = np.einsum('tgk,tkn,tgn->tg', self.weighted_l, self.mu_F, self.Y)

        outer_prod_mu_l = np.einsum('tgk,tgi->tgki', self.mu_L, self.mu_L)
        masked_sigma2_l = np.einsum('tgk,ki->tgki', self.sigma2_L, self.I)
        mixed_moment_l = outer_prod_mu_l + masked_sigma2_l
        mixed_eta = np.power(self.eta[:, :, :, np.newaxis], self.I) * self.eta[:, :, np.newaxis, :]
        weighted_mix_moment_l = mixed_moment_l * mixed_eta

        outer_prod_mu_f = np.einsum('tkn,tin->tkin', self.mu_F, self.mu_F)
        masked_sigma2_f = np.einsum('tkn,ki->tkin', self.sigma2_F, self.I)
        mixed_moment_f = outer_prod_mu_f + masked_sigma2_f
        mixed_moment_f_sum_n = mixed_moment_f.sum(axis=-1)

        l_FF_l_bar = weighted_mix_moment_l * mixed_moment_f_sum_n[:, np.newaxis, :, :]
        # sum over K
        l_FF_l_bar = l_FF_l_bar.sum(axis=(-2, -1))

        b_tau_new = y_dot_y - 2 * lbar_F_y + l_FF_l_bar

        return a_tau_new, b_tau_new

    def update_alpha(self):

        a_alpha_new = self.a_alpha_prior + 0.5 * self.eta.sum(axis=1)
        b_alpha_new = self.b_alpha_prior + 0.5 * np.einsum('tgk,tgk->tk', self.eta, self.second_moment_L)

        return a_alpha_new, b_alpha_new

    def update_A(self):
        dirchlet_A_new = self.dirchlet_A_prior + self.pairwise.sum(axis=(0, 1))  # sums over time and genes

        return dirchlet_A_new

    # elbo
    def elbo(self):
        """
        god help me
        :return:
        """
        digam_a_tau = digamma(self.a_tau)
        digam_a_alpha = digamma(self.a_alpha)
        log_b_tau = np.log(self.b_tau)
        log_b_alpha = np.log(self.b_alpha)
        p_F = -0.5 * ((self.mu_F + self.sigma2_F).sum() + (self.K * self.N * self.T) * LOG_TWOPI)
        p_tau = (
                (self.a_tau_prior - 1) * (digam_a_tau - np.log(self.b_tau)) -
                 self.a_over_b_tau * self.b_tau_prior +
                self.a_tau_prior * np.log(self.b_tau_prior) -
                np.log(gamma(self.a_tau_prior))
                 ).sum()
        p_alpha = (
                (self.a_alpha_prior - 1) * (digam_a_alpha - log_b_alpha) -
                self.a_over_b_alpha * self.b_alpha_prior +
                self.a_alpha_prior * np.log(self.b_alpha_prior) -
                np.log(gamma(self.a_alpha_prior))
        ).sum()
        mixed_eta = np.power(self.eta[:, :, :, np.newaxis], self.I) * self.eta[:, :, np.newaxis, :]
        weighted_mix_moment_l = mixed_eta * self.mixed_moment_L
        p_L = 0.5 * (
            self.eta * (
                digam_a_alpha[:, np.newaxis, :] -
                LOG_TWOPI - log_b_alpha[:, np.newaxis, :] -
                self.a_over_b_alpha[:, np.newaxis, :] * self.second_moment_L
                )
            )

        q_F = 0.5 * (self.T * self.K * self.N * (1 + LOG_TWOPI) + np.log(self.sigma2_F.sum()))
        q_tau = (
            self.a_tau - log_b_tau + np.log(gamma(self.a_tau)) + (1 - self.a_tau) * digam_a_tau
        ).sum()

        q_alpha = (
            self.a_alpha - log_b_alpha + np.log(gamma(self.a_alpha)) + (1 - self.a_alpha) * digam_a_alpha
        ).sum()

        q_L = 0.5 * (self.eta * np.log(2 * np.pi * self.sigma2_L) + self.eta).sum()
        q_Z = np.log(self.normalisations).sum()

        # simplified expression so it's easier
        p_q_A = ((self.dirchlet_A - self.dirchlet_A_prior) * self.A_variational).sum()


        return (
            p_F + p_tau + p_alpha + p_L + q_F + q_alpha + q_tau + q_L + q_Z + p_q_A
        )

    # full algorithm

    def run(self, eps: float = 1e-3, max_it: int = 1000, progress_bar : bool = False):
        elbo_converged = False
        current_elbo = -np.inf
        elbos = np.zeros(max_it)
        converged = False
        for i in tqdm(range(max_it), disable = not progress_bar):

            # update 'emission' parameters
            self.mu_L, self.sigma2_L = self.update_L()
            self.mu_F, self.sigma2_F = self.update_F()
            self.a_tau, self.b_tau = self.update_tau()
            self.a_alpha, self.b_alpha = self.update_alpha()

            # local updates
            self.dirchlet_A = self.update_A()
            self.A_variational, self.variational_likelihoods = self.V_step()
            self.eta, self.pairwise, self.normalisations = self.M_step()

            # compute elbo
            new_elbo = self.elbo()
            elbos[i] = new_elbo
            delta = np.abs(current_elbo - new_elbo)

            self.logger.debug(f"Iteration {i} - ELBO - {new_elbo:.4f}")

            if new_elbo > current_elbo:
                warnings.warn(f"Iteration {i} - increase in ELBO occurred")

            if delta < eps:
                print("ELBO Converged, done.")
                converged = True

        if not converged:
            warnings.warn("ELBO did not converge for this run")

        return elbos
