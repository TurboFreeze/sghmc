# hmc

## leapfrog
```python
leapfrog(M, q, p, dVdq, path_len, step_size)
```
Leapfrog integrator for standard HMC and naive SGHMC

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

## leapfrog_friction
```python
leapfrog_friction(M, C, q, p, dVdq, path_len, step_size)
```
Leapfrog integrator for Stochastic Gradient Hamiltonian Monte Carlo.
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

## hmc
```python
hmc(M, n_samples, negative_log_prob, initial_position, path_len=1, step_size=0.5, debug=False)
```
Hamiltonian Monte Carlo sampling.

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
debug : bool
  Flag to include debugging information like timing in the returned values

Returns
-------
samples, debug : np.array, dict
  Array of length `n_samples`;
  Dictionary of debugging output (if debug=False, this is empty dict)

## nsghmc
```python
nsghmc(X, y, M, n_samples, negative_log_prob, initial_position, batch_size=1, path_len=1, step_size=0.5, debug=False)
```
Naive Stochastic Hamiltonian Monte Carlo sampling.

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
debug : bool
  Flag to include debugging information like timing in the returned values

Returns
-------
samples, debug : np.array, dict
  Array of length `n_samples`;
  Dictionary of debugging output (if debug=False, this is empty dict)

## sghmc
```python
sghmc(X, y, M, C, n_samples, negative_log_prob, initial_position, batch_size=1, path_len=1, step_size=0.5)
```
Stochastic Hamiltonian Monte Carlo sampling.
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
debug : bool
  Flag to include debugging information like timing in the returned values

Returns
-------
samples, debug : np.array, dict
  Array of length `n_samples`;
  Dictionary of debugging output (if debug=False, this is empty dict)

