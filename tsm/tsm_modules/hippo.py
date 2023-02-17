"""Modules for HiPPO (Recurrent Memory with Optimal Polynomial Projections)"""

import gin
import jax
import jax.numpy as jnp
import numpy as np
import functools
import haiku as hk
from scipy import special as ss
from einops import rearrange
import logging


def embed_c2r(A):
    A = rearrange(A, '... m n -> ... m () n ()')
    A = np.pad(A, ((0, 0), (0, 1), (0, 0), (0, 1))) + \
        np.pad(A, ((0, 0), (1, 0), (0, 0), (1,0)))
    return rearrange(A, 'm x n y -> (m x) (n y)')


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures
    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    elif measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1. - b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    elif measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        A = (1./L[:, None]) * A * L[None, :]
        B = (1./L[:, None]) * B * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy() # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)
    elif measure == 'fourier':
        freqs = np.arange(N//2)
        d = np.stack([freqs, np.zeros(N//2)], axis=-1).reshape(-1)[:-1]
        A = 2*np.pi*(np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N//2, N//2)))
        B = embed_c2r(np.ones((N//2, 1)))[..., :1]
    elif measure == 'random':
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
    elif measure == 'diagonal':
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
    else:
        raise NotImplementedError

    return A, B


class AdaptiveTransition(object):
    def __init__(self, N, a, b):
        self.N = N
        self.a = a
        self.b = b
        self.ones = np.ones(self.N)
        self.I = np.eye(self.N)
        self.arange = np.arange(self.N - 1)
        self._cached_B = self._B()

    @property
    def B(self):
        return self._cached_B

    @property
    def A(self):
        return self._A()

    def _A(self):
        L = np.tril(np.ones((self.N, self.N)))
        D = self.a * self.I
        return -(L+D)

    def _B(self):
        return self.b * self.ones

    def forward_mult(self, u, delta):
        """ Computes (I + delta A) u
        A: (n, n)
        u: (..., n)
        delta: (...) or scalar
        output: (..., n)
        """
        x = np.cumsum(u, -1)
        x = x + u * self.a
        x = u - delta * x  # Because A is negated in the representation
        return x

    def precompute_backward(self, delta): # TODO should be called inverse?
        """ Store elements along the diagonals of (I - d A)^{-1}
        # a' = a + 1/dt
        delta: (...)
        output: (..., N)
        """
        ad = self.a*delta  # (..., 1)
        ad_p1 = 1 + ad
        denom = ad_p1 + delta  # 1 + a'
        denom_inv = np.reciprocal(denom)  # 1. / denom
        s = - delta * denom_inv * denom_inv  # -1/(1+a')^2
        b = ad_p1 * denom_inv    # a' / (1 + a')
        pows = b ** self.arange  ## TODO benchmark against cumprod or cumsum in log space
        tail = s * pows
        ret = np.concatenate((denom_inv, tail), -1)
        print(denom_inv.shape, tail.shape, ret.shape)
        return ret

    def inverse_mult(self, u, delta):
        """ Computes (I - d A)^-1 u """
        c = self.precompute_backward(delta)
        x = self.causal_convolution(c, u)
        return x

    def bilinear(self, dt, u, v, alpha=.5):
        """ Computes the bilinear (aka trapezoid or Tustin's) update rule.
        (I - d/2 A)^-1 (I + d/2 A) u + d (I - d/2 A)^-1 B v
        dt: (...)
        u: (..., N)
        v: (...)
        """
        x = self.forward_mult(u, (1-alpha)*dt)
        print(x)
        v = dt * v
        v = v * self.B
        x = x + v
        print(x)
        x = self.inverse_mult(x, alpha * dt)
        return x

    def bilinear_debug(self, dt, u, v, alpha=.5):
        A = self.A
        pre = self.I - alpha * dt * A
        pre = np.linalg.inv(pre)
        B = self.B
        post = self.I + (1 - alpha) * dt * A
        print(post @ u)
        print(post @ u + (dt * B * v))
        return pre @ post @ u + dt * pre @ (B * v)

    def causal_convolution(self, u, v):
        n = u.shape[-1]
        u_expand = np.pad(u, ((0, n),))
        v_expand = np.pad(v, ((0, n),))
        u_f = np.fft.rfft(u_expand, n=2 * n, axis=-1)
        v_f = np.fft.rfft(v_expand, n=2 * n, axis=-1)
        uv_f = u_f * v_f
        output = np.fft.irfft(uv_f, n=2 * n, axis=-1)[..., :n]
        return output


@gin.configurable
class JaxAdaptiveTransition(object):
    def __init__(self, N, dt=0.1, a=-0.5, b=1.0, theta=100.0):
        self.N = N
        self.a = a
        self.b = b
        self.dt = dt
        self.inv_theta = 1 / theta
        logging.info('Hippo hyper-parameters: dt=%f, a=%f, b=%f', dt, a, b)

    def forward_mult(self, u, delta):
        """ Computes (I + delta A) u
        A: (n, n)
        u: (..., n)
        delta: (...) or scalar
        output: (..., n)
        """
        x = jnp.cumsum(u, axis=-1)
        x = x + u * self.a
        x = u - delta * x  # Because A is negated in the representation
        return x

    def precompute_backward(self, delta): # TODO should be called inverse?
        """ Store elements along the diagonals of (I - d A)^{-1}
        # a' = a + 1/dt
        delta: (...)
        output: (..., N)
        """
        ad = self.a*delta  # (..., 1)
        ad_p1 = 1 + ad
        denom = jnp.array([ad_p1 + delta])  # 1 + a'
        denom_inv = jnp.reciprocal(denom)  # 1. / denom
        s = - delta * denom_inv * denom_inv  # -1/(1+a')^2
        b = ad_p1 * denom_inv    # a' / (1 + a')
        pows = b ** jnp.arange(self.N - 1)  ## TODO benchmark against cumprod or cumsum in log space
        tail = s * pows
        ret = jnp.concatenate((denom_inv, tail), axis=-1)
        return ret

    def inverse_mult(self, u, delta):
        """ Computes (I - d A)^-1 u """
        c = self.precompute_backward(delta)
        x = self.causal_convolution(c, u)
        return x

    def causal_convolution(self, u, v):
        n = u.shape[-1]
        u_expand = jnp.pad(u, ((0, n),))
        v_expand = jnp.pad(v, ((0, n),))
        u_f = jnp.fft.rfft(u_expand, n=2 * n, axis=-1)
        v_f = jnp.fft.rfft(v_expand, n=2 * n, axis=-1)
        uv_f = u_f * v_f
        output = jnp.fft.irfft(uv_f, n=2 * n, axis=-1)[..., :n]
        return output

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def bilinear(self, u, v, alpha=.5):
        """ Computes the bilinear (aka trapezoid or Tustin's) update rule.
        (I - d/2 A)^-1 (I + d/2 A) u + d (I - d/2 A)^-1 B v
        dt: (...)
        u: (..., N)
        v: (...)
        """
        forward_mult = jnp.vectorize(
            functools.partial(self.forward_mult, delta=(1-alpha)*self.dt),
            signature='(n)->(n)')
        x = forward_mult(u)
        v = self.dt * v * self.b
        x = x + v
        inverse_mult = jnp.vectorize(
            functools.partial(self.inverse_mult, delta=alpha * self.dt),
            signature='(n)->(n)')
        x = inverse_mult(x)
        return x, None

    def bilinear_fast(self, u, v, pre, post):
        """ Computes the bilinear (aka trapezoid or Tustin's) update rule.
        (I - d/2 A)^-1 (I + d/2 A) u + d (I - d/2 A)^-1 B v
        dt: (...)
        u: (..., N)
        v: (...)
        """
        dt = self.dt
        B = self.b
        mat_vec_mul = jnp.vectorize(jnp.matmul, signature='(n,m),(m)->(n)')
        return mat_vec_mul(pre, mat_vec_mul(post, u) + dt * B * v), None

    def bilinear_dplr(self, u, v, Lambda, P, Q, D, scalar, VB):
        """ Computes the bilinear (aka trapezoid or Tustin's) update rule.
        (I - d/2 A)^-1 (I + d/2 A) u + d (I - d/2 A)^-1 B v
        """
        dt = self.dt
        mat_vec_mul = jnp.vectorize(lambda A, B: A @ B, signature='(n,m),(m)->(n)')
        A0_x = (2 / dt) * u + Lambda * u - mat_vec_mul(P, mat_vec_mul(Q, u))
        A0_x_B_u = A0_x + 2 * VB * v
        A1_result = D * A0_x_B_u - D * mat_vec_mul(P @ scalar, mat_vec_mul(Q, D * A0_x_B_u))
        return A1_result, None

    def get_init_state(self, alpha=.5):
        dt = self.dt
        L = np.tril(np.ones((self.N, self.N)))
        D = self.a * np.eye(self.N)
        A = -(L+D)
        pre = np.eye(self.N) - alpha * dt * A
        pre = np.linalg.inv(pre)
        post = np.eye(self.N) + (1 - alpha) * dt * A
        pre = hk.get_state("pre", shape=pre.shape, init=lambda _, __: jnp.array(pre))
        post = hk.get_state("post", shape=post.shape, init=lambda _, __: jnp.array(post))
        return pre, post

    def get_init_state_nplr(self, alpha=.5):
        dt = self.dt
        L = np.tril(np.ones((self.N, self.N)))
        Diag = self.a * np.eye(self.N)
        A = -(L+Diag)
        P = (0.5**0.5) * np.ones((self.N, 1))
        Q = P.T
        normal_A = A + P @ Q
        Lambda, V = np.linalg.eig(normal_A)
        D = 1.0 / (2 / dt - Lambda)

        P = V.conj().T @ P
        Q = Q @ V
        VB = (V.conj().T @ np.ones((self.N, 1))).T
        scalar = 1.0 / (1 + (D * Q) @ P)

        Lambda = hk.get_state("Lambda", shape=Lambda.shape, init=lambda _, __: jnp.array(Lambda))
        P = hk.get_state("P", shape=P.shape, init=lambda _, __: jnp.array(P))
        Q = hk.get_state("Q", shape=Q.shape, init=lambda _, __: jnp.array(Q))
        D = hk.get_state("D", shape=D.shape, init=lambda _, __: jnp.array(D))
        scalar = hk.get_state("scalar", shape=scalar.shape, init=lambda _, __: jnp.array(scalar))
        V = hk.get_state("V", shape=V.shape, init=lambda _, __: jnp.array(V))
        VB = hk.get_state("VB", shape=V.shape, init=lambda _, __: jnp.array(VB))
        return Lambda, P, Q, D, scalar, V, VB

    def get_init_state_nplr_legT(self, alpha=.5):
        dt = self.dt
        Q = np.arange(self.N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        B = self.inv_theta * R[:, None]
        A = - self.inv_theta * R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        P = np.sqrt(1 + 2 * np.arange(self.N, dtype=np.float64))
        P0 = np.copy(P)
        P0[0::2] = 0.0
        P1 = np.copy(P)
        P1[1::2] = 0.0
        P = np.stack([P0, P1], axis=0).T * (self.inv_theta ** 0.5)
        Q = P.T
        normal_A = A + P @ P.T
        Lambda, V = np.linalg.eig(normal_A)
        D = 1.0 / (2 / dt - Lambda)
        P = V.conj().T @ P
        Q = Q @ V
        VB = (V.conj().T @ B).T
        scalar = 1.0 / (1 + (D * Q) @ P)
        Lambda = hk.get_state("Lambda", shape=Lambda.shape, init=lambda _, __: jnp.array(Lambda))
        P = hk.get_state("P", shape=P.shape, init=lambda _, __: jnp.array(P))
        Q = hk.get_state("Q", shape=Q.shape, init=lambda _, __: jnp.array(Q))
        D = hk.get_state("D", shape=D.shape, init=lambda _, __: jnp.array(D))
        scalar = hk.get_state("scalar", shape=scalar.shape, init=lambda _, __: jnp.array(scalar))
        V = hk.get_state("V", shape=V.shape, init=lambda _, __: jnp.array(V))
        VB = hk.get_state("VB", shape=V.shape, init=lambda _, __: jnp.array(VB))
        return Lambda, P, Q, D, scalar, V, VB

    @functools.partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def bilinear_scan(self, v, axis=0, alpha=.5, restore_axis=True):
        xs = v
        if axis != 0:
            xs = jnp.swapaxes(xs, axis, 0)
        u = jnp.zeros(xs.shape[1:] + (self.N,))
        carry = u
        new_carry, _ = jax.lax.scan(
            functools.partial(self.bilinear, alpha=alpha),
            init=carry,
            xs=jnp.expand_dims(xs, axis=-1),
        )
        if restore_axis:
            new_carry = jnp.transpose(new_carry, (len(new_carry.shape) - 1, ) + tuple(range(0, len(new_carry.shape) - 1)))
            if axis != 0:
                new_carry = jnp.swapaxes(new_carry, axis, 0)
        return new_carry

    def bilinear_scan_fast(self, v, pre, post, axis=0, restore_axis=True):
        xs = v
        if axis != 0:
            xs = jnp.swapaxes(xs, axis, 0)
        u = jnp.zeros(xs.shape[1:] + (self.N,))
        carry = u
        new_carry, _ = hk.scan(
            functools.partial(self.bilinear_fast, pre=pre, post=post),
            init=carry,
            xs=jnp.expand_dims(xs, axis=-1),
        )
        if restore_axis:
            new_carry = jnp.transpose(new_carry, (len(new_carry.shape) - 1, ) + tuple(range(0, len(new_carry.shape) - 1)))
            if axis != 0:
                new_carry = jnp.swapaxes(new_carry, axis, 0)
        return new_carry

    def bilinear_scan_dplr(self, v, Lambda, P, Q, D, scalar, VB, V=None, axis=0, restore_axis=True):
        xs = v
        if axis != 0:
            xs = jnp.swapaxes(xs, axis, 0)
        u = jnp.zeros(xs.shape[1:] + (self.N,), dtype=jnp.complex64)
        carry = u
        new_carry, _ = hk.scan(
            functools.partial(self.bilinear_dplr, Lambda=Lambda, P=P, Q=Q, D=D, scalar=scalar, VB=VB),
            init=carry,
            xs=jnp.expand_dims(xs, axis=-1),
        )
        if V is not None:
            mat_vec_mul = jnp.vectorize(lambda A, B: A @ B, signature='(n,m),(m)->(n)')
            new_carry = mat_vec_mul(V, new_carry).real
        if restore_axis:
            new_carry = jnp.transpose(new_carry, (len(new_carry.shape) - 1, ) + tuple(range(0, len(new_carry.shape) - 1)))
            if axis != 0:
                new_carry = jnp.swapaxes(new_carry, axis, 0)
        return new_carry
