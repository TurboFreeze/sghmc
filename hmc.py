from autograd import grad
import autograd.numpy as np
import scipy.stats as st


def leapfrog(M, q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for standard HMC and naive SGHMC

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    q : np.floatX
      Initial position
    p : np.floatX
      Initial momentum
    dVdq : callable
      Gradient of the velocity
    path_len : float
      How long to integrate for
    step_size : float
      How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
      New position and momentum
    """
    q, p = np.copy(q), np.copy(p)
    Minv = np.linalg.inv(M)

    p -= step_size * dVdq(q) / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * np.dot(Minv, p)  # whole step
        p -= step_size * dVdq(q)  # whole step
    q += step_size * np.dot(Minv, p)  # whole step
    p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, -p


def leapfrog_friction(M, C, q, p, dVdq, path_len, step_size):
    """Leapfrog integrator for Stochastic Gradient Hamiltonian Monte Carlo.
    Includes friction term per https://arxiv.org/abs/1402.4102

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    C : matrix
      Upper bound parameter for friction term
    q : np.floatX
      Initial position
    p : np.floatX
      Initial momentum
    dVdq : callable
      Gradient of the velocity
    path_len : float
      How long to integrate for
    step_size : float
      How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
      New position and momentum
    """
    q, p = np.copy(q), np.copy(p)
    Minv = np.linalg.inv(M)

    p -= step_size * (dVdq(q) + np.dot(C, np.dot(Minv, p))) / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * np.dot(Minv, p)  # whole step
        p -= step_size * (dVdq(q) + np.dot(C, np.dot(Minv, p)))  # whole step
    q += step_size * np.dot(Minv, p)  # whole step
    p -= step_size * (dVdq(q) + np.dot(C, np.dot(Minv, p))) / 2  # half step

    # momentum flip at end
    return q, -p


def hmc(M, n_samples, negative_log_prob, initial_position, path_len=1, step_size=0.5):
    """Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    n_samples : int
      Number of samples to return
    negative_log_prob : callable
      The negative log probability to sample from
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
      Array of length `n_samples`.
    """

    # autograd magic
    dVdq = grad(negative_log_prob)

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]

    for p0 in momentum.rvs(size=size):
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            M,
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1]) - np.sum(momentum.logpdf(p0))
        new_log_p = negative_log_prob(q_new) - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p:
            samples.append(q_new)
        else:
            samples.append(np.copy(samples[-1]))

    return np.array(samples[1:])


def nsghmc(X, y, M, n_samples, negative_log_prob, initial_position,
    batch_size=1, path_len=1, step_size=0.5):
    """Naive Stochastic Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    X : np.matrix
      Predictor data in matrix of dimensions N x D.
    y: np.array
      Response data in a vector of length N.
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    n_samples : int
      Number of samples to return
    negative_log_prob : callable
      The negative log probability to sample from. Should be a function taking
      three arguments: p(w, x_train, y_train) for the parameters, predictor data,
      and response data.
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
      Array of length `n_samples`.
    """
    n = len(y)

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]

    # shuffle the data
    indices_shuffled = np.random.choice(n, n, replace=False)
    X = X[indices_shuffled,]
    y = y[indices_shuffled]

    batch_num = 0

    # iterate for samples
    for p0 in momentum.rvs(size=size):
        # subset the data
        indices_subset = range(batch_num * batch_size, (batch_num + 1) * batch_size)
        X_sub = X[indices_subset,]
        y_sub = y[indices_subset]

        # autograd stochastic gradient on batch magic
        dVdq = grad(lambda q: negative_log_prob(q, X_sub, y_sub))

        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            M,
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1], X_sub, y_sub) - np.sum(momentum.logpdf(p0))
        new_log_p = negative_log_prob(q_new, X_sub, y_sub) - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p:
            samples.append(q_new)
        else:
            samples.append(np.copy(samples[-1]))

    return np.array(samples[1:])


def sghmc(X, y, M, C, n_samples, negative_log_prob, initial_position,
    batch_size=1, path_len=1, step_size=0.5):
    """ Stochastic Hamiltonian Monte Carlo sampling.
    Based on https://arxiv.org/abs/1402.4102

    Parameters
    ----------
    X : np.matrix
      Predictor data in matrix of dimensions N x D.
    y: np.array
      Response data in a vector of length N.
    M : np.matrix
      Mass of the Euclidean-Gaussian kinetic energy of shape D x D
    C : matrix
      Upper bound parameter for friction term
    n_samples : int
      Number of samples to return
    negative_log_prob : callable
      The negative log probability to sample from
    initial_position : np.array
      A place to start sampling from.
    path_len : float
      How long each integration path is. Smaller is faster and more correlated.
    step_size : float
      How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
      Array of length `n_samples`.
    """
    n = len(y)

    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]

    # shuffle the data
    indices_shuffled = np.random.choice(n, n, replace=False)
    X = X[indices_shuffled,]
    y = y[indices_shuffled]

    batch_num = 0

    # iterate for samples
    for p0 in momentum.rvs(size=size):
        # subset the data
        indices_subset = range(batch_num * batch_size, (batch_num + 1) * batch_size)
        X_sub = X[indices_subset,]
        y_sub = y[indices_subset]

        # autograd stochastic gradient on batch magic
        dVdq = grad(lambda q: negative_log_prob(q, X_sub, y_sub))

        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog_friction(
            M,
            C,
            samples[-1],
            p0,
            dVdq,
            path_len=path_len,
            step_size=step_size,
        )

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1], X_sub, y_sub) - np.sum(momentum.logpdf(p0))
        new_log_p = negative_log_prob(q_new, X_sub, y_sub) - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p:
            samples.append(q_new)
        else:
            samples.append(np.copy(samples[-1]))

    return np.array(samples[1:])
