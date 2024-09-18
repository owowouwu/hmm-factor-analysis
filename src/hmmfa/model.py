import numpy as np
import scipy as sp
import scipy.stats
import warnings
import tqdm
import h5py
from tqdm import tqdm
from scipy.special import digamma, gamma, gammaln
from typing import Dict, List, Union, Literal
from functools import cached_property
from . import get_logger
from opt_einsum import contract

# useful constants to precompute
LOG_TWOPI = np.log(2 * np.pi)
np.set_printoptions(precision=3, suppress=True)

class HiddenMarkovFA:

    def __init__(self, data: np.ndarray, n_factors: int, init_method: Literal['random', 'prior'] = 'prior',
                 **kwargs
                 ):
        """
        :param data: data array of shape (genes x samples x time)
        :param n_factors: number of factors to use
        """
        self.Y = data
        self.K = n_factors
        self.T, self.G, self.N = data.shape
        self.init_method = init_method
        self._parse_hyperparameters(**kwargs)
        self._init_variational_params(self.init_method)
        self._updated_a_tau = False
        self.mask = ~np.eye(self.K, dtype=bool)
        self.I = np.eye(self.K, dtype=bool)
        self.logger = get_logger()

    def _parse_hyperparameters(self, **kwargs):
        self.a_alpha_prior = kwargs.get('a_alpha_prior', np.full(shape=(self.T, self.K), fill_value=1))
        self.b_alpha_prior = kwargs.get('b_alpha_prior', np.full(shape=(self.T, self.K), fill_value=1))
        self.a_tau_prior = kwargs.get('a_tau_prior', np.full(shape=(self.T, self.G), fill_value=1))
        self.b_tau_prior = kwargs.get('b_tau_prior', np.full(shape=(self.T, self.G), fill_value=1))
        self.pi = kwargs.get('pi', np.full(shape=(self.K, 2), fill_value=0.5))
        self.log_pi = np.log(self.pi)  # for later
        self.dirchlet_A_prior = kwargs.get('dirchlet_A_prior', np.full(shape=(self.K, 2, 2), fill_value=1))

    def _init_variational_params(self, how : Literal['random', 'prior'] = 'prior'):
        if how == 'random':
            self.mu_F = sp.stats.norm.rvs(size=(self.T, self.K, self.N))
            self.sigma2_F = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.K, self.N))
            self.mu_L = sp.stats.norm.rvs(size=(self.T, self.G, self.K))
            self.sigma2_L = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.G, self.K))
            self.a_tau = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.G))
            self.b_tau = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.G))
            self.a_alpha = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.K))
            self.b_alpha = sp.stats.gamma.rvs(a=1, scale=1, size=(self.T, self.K))
            self.dirchlet_A = sp.stats.gamma.rvs(a=1, scale=1, size=(self.K, 2, 2))
            self.eta = self.eta = sp.stats.beta.rvs(a=1, b=1, size=(self.T, self.G, self.K))
        elif how == 'prior':
            # fixed
            self.mu_F = np.zeros(shape = (self.T, self.K, self.N))
            self.mu_L = np.zeros(shape=(self.T, self.G, self.K))
            self.sigma2_F = np.ones(shape = (self.T, self.K, self.N))
            # based on prior mean
            self.sigma2_L = np.repeat((self.a_alpha_prior / self.b_alpha_prior)[:, np.newaxis, :], self.G, axis = 1)
            # just take prior parameters
            self.a_alpha = self.a_alpha_prior
            self.b_alpha = self.b_alpha_prior
            self.a_tau = self.a_tau_prior
            self.b_tau = self.b_tau_prior
            self.dirchlet_A = self.dirchlet_A_prior
            self.eta = np.tile(self.pi[:, 1], (self.T, self.G, 1))





    def save(self, filename, include_hmm = False):
        if not include_hmm:
            exclude_keys = {'normalisations', 'pairwise', 'variational_likelihoods', 'A_variational'}
        else:
            exclude_keys = {}
        arrays = {
            'mu_F': self.mu_F,
            'sigma2_F': self.sigma2_F,
            'mu_L': self.mu_L,
            'sigma2_L': self.sigma2_L,
            'a_tau': self.a_tau,
            'b_tau': self.b_tau,
            'a_alpha': self.a_alpha,
            'b_alpha': self.b_alpha,
            'dirchlet_A': self.dirchlet_A,
            'A_variational': self.A_variational,
            'variational_likelihoods': self.variational_likelihoods,
            'eta': self.eta,
            'pairwise': self.pairwise,
            'normalisations': self.normalisations
        }

        with h5py.File(filename, 'w') as hf:
            for key, array in arrays.items():
                if key not in exclude_keys:
                    hf.create_dataset(key, data=array)
                    self.logger.debug(f"Saved array '{key}' to {filename}")
                else:
                    self.logger.debug(f"Excluded array '{key}' from saving.")
        pass

    # cached intermediate terms
    def _clean_cache_L(self):
        if 'weighted_L' in self.__dict__:
            del self.__dict__['weighted_L']

        if 'mixed_moment_L' in self.__dict__:
            del self.__dict__['mixed_moment_L']

        if 'second_moment_L' in self.__dict__:
            del self.__dict__['second_moment_L']

    def _clean_cache_F(self):

        if 'mixed_moment_F' in self.__dict__:
            del self.__dict__['mixed_moment_F']

        if 'second_moment_F' in self.__dict__:
            del self.__dict__['second_moment_F']

    def _clean_cache_tau(self):
        if 'a_over_b_tau' in self.__dict__:
            del self.__dict__['a_over_b_tau']

    def _clean_cache_alpha(self):
        if 'a_over_b_alpha' in self.__dict__:
            del self.__dict__['a_over_b_alpha']

    @property
    def y_dot_y(self):
        return contract('tgn,tgn->tg', self.Y, self.Y)
    @property
    def weighted_L(self):
        return self.eta * self.mu_L

    @property
    def a_over_b_alpha(self):
        return self.a_alpha / self.b_alpha

    @property
    def a_over_b_tau(self):
        return self.a_tau / self.b_tau

    @property
    def second_moment_F(self):
        return self.mu_F ** 2 + self.sigma2_F

    @property
    def second_moment_L(self):
        return self.mu_L ** 2 + self.sigma2_L

    @property
    def mixed_moment_F(self):
        outer_prod_mu_f = contract('tkn,tin->tkin', self.mu_F, self.mu_F)
        masked_sigma2_f = contract('tkn,ki->tkin', self.sigma2_F, self.I)
        mixed_moment_f = outer_prod_mu_f + masked_sigma2_f
        return mixed_moment_f

    @property
    def mixed_moment_L(self):
        outer_prod_mu_l = contract('tgk,tgi->tgki', self.mu_L, self.mu_L)
        masked_sigma2_l = contract('tgk,ki->tgki', self.sigma2_L, self.I)
        mixed_moment_l = outer_prod_mu_l + masked_sigma2_l
        return mixed_moment_l


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
        mu_bar = contract('tgk,ki->tgk', self.weighted_L, self.mask)
        mu_yd_conditional_z = self.mu_L + mu_bar
        mu_yd_conditional_z = np.stack([mu_bar, mu_yd_conditional_z], axis=0)
        mu_yd_conditional_z = contract('ztgk,tkn->ztgkn', mu_yd_conditional_z, self.mu_F)  # shape (2, T, G, K, N)
        sigmas = np.sqrt(self.a_over_b_tau)[np.newaxis, :, :,np.newaxis, np.newaxis]

        cond_likelihoods = sp.stats.norm.logpdf(
            self.Y[np.newaxis, :, :, np.newaxis, :],
            loc = mu_yd_conditional_z,
            scale = sigmas
        ).sum(axis = -1)

        return A_variational_new, cond_likelihoods

    # Baum Welsh steps
    def M_step(self):
        forward, backward = self.forward_messages(), self.backward_messages()
        log_q_z = forward + backward
        pairwise_new = np.zeros(shape=(self.T - 1, self.G, self.K, 2, 2))
        for t in range(1, self.T):
            pairwise_new[t - 1] = (
                    forward[t - 1, :, :, :, np.newaxis] +
                    self.A_variational[np.newaxis, :, :, :] +
                    self.variational_likelihoods.reshape(self.T, self.G, self.K, 2)[t-1,:,:,:,np.newaxis]
            )

        # normalise to ensure these are probabilities
        max_q_z = np.max(log_q_z, axis = -1)
        q_z = np.exp(log_q_z - max_q_z[:,:,:,np.newaxis])
        normalisation =  q_z.sum(axis=-1)
        q_z = q_z / normalisation[:,:,:,np.newaxis]
        # eta is q_z(1) (expected value over hidden states)
        eta_new = q_z[:, :, :, 1]

        # normalisation constants
        log_normalisation = max_q_z + np.log(normalisation)

        pairwise_new_probs = np.exp(pairwise_new - np.max(pairwise_new, axis = (-1,-2), keepdims=True))
        pairwise_normalisations = pairwise_new_probs.sum(axis=(-1, -2), keepdims=True)
        pairwise_new_probs = pairwise_new_probs / pairwise_normalisations

        return eta_new, pairwise_new_probs, log_normalisation

    def forward_messages(self):
        forward = np.ones(shape=(self.T, self.G, self.K, 2))
        forward[0, :, :, :] = self.log_pi[np.newaxis, :, :]
        for t in range(1, self.T):
            forward[t, :, :, 0] = np.logaddexp(
                forward[t - 1, :, :, 0] + self.A_variational[np.newaxis, :, 0, 0] + self.variational_likelihoods[0, t, :, :],
                forward[t - 1, :, :, 1] + self.A_variational[np.newaxis, :, 1, 0] + self.variational_likelihoods[0, t, :, :]
                )
            forward[t, :, :, 1] = np.logaddexp(
                forward[t - 1, :, :, 0] + self.A_variational[np.newaxis, :, 0, 1] + self.variational_likelihoods[1, t, :, :],
                forward[t - 1, :, :, 1] + self.A_variational[np.newaxis, :, 1, 1] + self.variational_likelihoods[1, t, :, :]
                )

        return forward

    def backward_messages(self):
        backward = np.ones(shape=(self.T, self.G, self.K, 2))
        backward[self.T - 1, :, :, :] = 0
        # iterate backwards
        for t in range(self.T - 2, -1, -1):
            backward[t, :, :, 0] = np.logaddexp(
                backward[t + 1, :, :, 0] + self.A_variational[np.newaxis, :, 0, 0] + self.variational_likelihoods[0,t + 1, :, :],
                backward[t + 1, :, :, 1] + self.A_variational[np.newaxis, :, 0, 1] + self.variational_likelihoods[1,t + 1, :, :]
                )
            backward[t, :, :, 1] = np.logaddexp(
                backward[t + 1, :, :, 0] + self.A_variational[np.newaxis, :, 1, 0] + self.variational_likelihoods[0,t + 1, :, :],
                backward[t + 1, :, :, 1] + self.A_variational[np.newaxis, :, 1, 1] + self.variational_likelihoods[1,t + 1, :, :]
                )

        return backward

    # VI updates

    def update_L(self):
        # first update sigma
        sum_second_moment_F = self.second_moment_F.sum(axis=2)

        sigma2_L_new = self.a_over_b_alpha[:, np.newaxis, :] + contract('td,tk->tdk', self.a_over_b_tau,
                                                                         sum_second_moment_F
                                                                         )
        sigma2_L_new = 1 / sigma2_L_new

        # update mu
        Y_mu = contract('tgn,tkn->tgk', self.Y, self.mu_F)
        # cursed outer product operations
        outer_product_mu_eta = contract('tdj,tdj,tin,tjn->tdijn', self.eta, self.eta, self.mu_F, self.mu_F)
        outer_product_sum_offdiag = contract('tdijn,ij->tdin', outer_product_mu_eta, self.mask)
        outer_product_summed_n = outer_product_sum_offdiag.sum(axis=-1)

        mu_L_new = contract('td,tdk->tdk', self.a_over_b_tau, sigma2_L_new * (Y_mu - outer_product_summed_n))
        return mu_L_new, sigma2_L_new

    def update_F(self):
        sigma2_F_new = 1 / (contract('td,tdk->tk', self.a_over_b_tau, self.eta * self.second_moment_L) + 1)
        # broadcast this

        sigma2_F_new = np.repeat(sigma2_F_new[:, :, np.newaxis], self.N, axis=2)

        # more cursed outer product operations
        outer_product_mu_eta = contract('tdj,tdi,tdi,tdj->tdji', self.eta, self.eta, self.mu_L, self.mu_L)
        outer_prod_weighted = contract('td,tdij->tdij', self.a_over_b_tau, outer_product_mu_eta)
        outer_prod_weighted_sum_genes = outer_prod_weighted.sum(axis=1)
        sum_fk_Dtau_lk_bar = contract('tki,tin,ki->tkn', outer_prod_weighted_sum_genes, self.mu_F, self.mask)

        y_Dtau_l = contract('tgn,tg,tgk->tkn', self.Y, self.a_over_b_tau, self.weighted_L)

        mu_F_new = sigma2_F_new * (y_Dtau_l - sum_fk_Dtau_lk_bar)

        return mu_F_new, sigma2_F_new

    def update_tau(self):
        if not self._updated_a_tau:
            a_tau_new = self.a_tau_prior + (self.N / 2.)
        else:
            a_tau_new = self.a_tau

        y_dot_y = contract('tgn,tgn->tg', self.Y, self.Y)
        lbar_F_y = contract('tgk,tkn,tgn->tg', self.weighted_L, self.mu_F, self.Y)

        mixed_eta = np.power(self.eta[:, :, :, np.newaxis], self.I) * self.eta[:, :, np.newaxis, :]
        weighted_mix_moment_l = self.mixed_moment_L * mixed_eta

        mixed_moment_f_sum_n = self.mixed_moment_F.sum(axis=-1)

        l_FF_l_bar = weighted_mix_moment_l * mixed_moment_f_sum_n[:, np.newaxis, :, :]
        # sum over K
        l_FF_l_bar = l_FF_l_bar.sum(axis=(-2, -1))

        b_tau_new = y_dot_y - 2 * lbar_F_y + l_FF_l_bar

        return a_tau_new, b_tau_new

    def update_alpha(self):
        a_alpha_new = self.a_alpha_prior + 0.5 * self.eta.sum(axis=1)
        b_alpha_new = self.b_alpha_prior + 0.5 * contract('tgk,tgk->tk', self.eta, self.second_moment_L)

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
                gammaln(self.a_tau_prior)
        ).sum()
        p_alpha = (
                (self.a_alpha_prior - 1) * (digam_a_alpha - log_b_alpha) -
                self.a_over_b_alpha * self.b_alpha_prior +
                self.a_alpha_prior * np.log(self.b_alpha_prior) -
                gammaln(self.a_alpha_prior)
        ).sum()

        p_L = 0.5 * (
                self.eta * (
                digam_a_alpha[:, np.newaxis, :] -
                LOG_TWOPI - log_b_alpha[:, np.newaxis, :] -
                self.a_over_b_alpha[:, np.newaxis, :] * self.second_moment_L
            )
        ).sum()

        q_F = 0.5 * (self.T * self.K * self.N * (1 + LOG_TWOPI) + np.log(self.sigma2_F.sum()))
        q_tau = (
                self.a_tau - log_b_tau + gammaln(self.a_tau) + (1 - self.a_tau) * digam_a_tau
        ).sum()

        q_alpha = (
                self.a_alpha - log_b_alpha + gammaln(self.a_alpha) + (1 - self.a_alpha) * digam_a_alpha
        ).sum()

        q_L = 0.5 * (self.eta * np.log(2 * np.pi * self.sigma2_L) + self.eta).sum()
        q_Z = self.normalisations.sum()

        # simplified expression so it's easier
        p_q_A = ((self.dirchlet_A - self.dirchlet_A_prior) * self.A_variational).sum()

        return (
                p_F + p_tau + p_alpha + p_L + q_F + q_alpha + q_tau + q_L + q_Z + p_q_A
        )

    # full algorithm        print(pairwise_new_probs)

    def run(self, eps: float = 1e-4, max_it: int = 1000, max_tries: int = 1, progress_bar: bool = False):
        elbo_converged = False
        current_elbo = -np.inf
        elbos = np.zeros(max_it)
        converged = False
        retries = 1
        while retries < max_tries:
            self.logger.info(f"Try {retries}")
            for i in tqdm(range(max_it), disable=not progress_bar):
                # local updates
                self.logger.info(f"Iteration {i}")
                self.logger.debug("Performing local updates")
                self.logger.debug("Performing V step")
                self.A_variational, self.variational_likelihoods = self.V_step()

                self.logger.debug("Updated variational likelihoods and transition matrix")

                self.logger.debug("Running forward-backward algorithm")
                self.eta, self.pairwise, self.normalisations = self.M_step()
                self.logger.debug("Updated hidden states")
                # update 'emission' parameters
                self.logger.debug("Performing global updates")
                self.mu_L, self.sigma2_L = self.update_L()
                self.logger.debug("Updated L")
                #self._clean_cache_L()

                self.mu_F, self.sigma2_F = self.update_F()
                self.logger.debug("Updated F")
                #self._clean_cache_F()

                self.a_tau, self.b_tau = self.update_tau()
                self.logger.debug("Updated tau")
                #self._clean_cache_tau()

                self.a_alpha, self.b_alpha = self.update_alpha()
                self.logger.debug("Updated alpha")
                #self._clean_cache_alpha()

                self.dirchlet_A = self.update_A()
                self.logger.debug("Updated A")


                # compute elbo
                new_elbo = self.elbo()
                if i != 0:
                    delta = np.abs(current_elbo - new_elbo)
                    pct_change = delta / np.abs(current_elbo)
                    self.logger.info(f"Iteration {i} - ELBO - {new_elbo:.4f} ({pct_change:.1e}% change)")
                else:
                    self.logger.info(f"Iteration {i} - ELBO - {new_elbo:.4f}")
                if np.isnan(new_elbo):
                    # reset
                    self._init_variational_params(self.init_method)
                    self.logger.warning("Found nan elbo, retrying")
                    retries += 1
                    current_elbo = -np.inf
                    elbos = np.zeros(max_it)
                    break

                elbos[i] = new_elbo




                if new_elbo < current_elbo:
                    warnings.warn(f"Iteration {i} - decrease in ELBO occurred")
                if i > 0:
                    if pct_change < eps:
                        print("ELBO Converged, done.")
                        return elbos[0:i]
                current_elbo = new_elbo


            if not converged:
                warnings.warn("ELBO did not converge for this run")



        return elbos
