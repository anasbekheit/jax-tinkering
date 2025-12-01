from __future__ import annotations

import os
from functools import partial
import timeit

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import numpy as np
from jax import jit
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Scalar
from sklearn.cluster import KMeans

os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Types
Dataset = Float[Array, "n_samples n_features"]
Centroids = Float[Array, "k_clusters n_features"]
Assignments = Int[Array, "n_samples"]


@register_pytree_node_class
class KMeansState:
    def __init__(self, iteration: Scalar, centroids: Centroids, change_sq: Scalar):
        self.iteration = iteration
        self.centroids = centroids
        self.change_sq = change_sq

    def tree_flatten(self):
        return (self.iteration, self.centroids, self.change_sq), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class KMeansResult:
    def __init__(self, centroids: Centroids, assignments: Assignments):
        self.centroids = centroids
        self.assignments = assignments

    def tree_flatten(self):
        return (self.centroids, self.assignments), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jit
def assign_clusters(X: Dataset, centroids: Centroids) -> Assignments:
    """Computes the nearest cluster centroid for each sample."""
    # (n_samples,)
    X_norm_sq = jnp.sum(X**2, axis=1)
    # (k_clusters,)
    C_norm_sq = jnp.sum(centroids**2, axis=1)
    # (n_samples, k_clusters)
    dot_prod = jnp.dot(X, centroids.T)

    dists_sq = X_norm_sq[:, jnp.newaxis] - 2 * dot_prod + C_norm_sq[jnp.newaxis, :]

    return jnp.argmin(dists_sq, axis=1)


@partial(jit, static_argnums=(2, 3, 4))
def jax_kmeans(
    key: PRNGKeyArray,
    X: Dataset,
    k: int,
    max_iters: int = 20,
    tol: float = 1e-4,
) -> KMeansResult:
    init_centroids: Centroids = jax.random.permutation(key, X)[:k]

    init_state = KMeansState(
        iteration=jnp.int32(0),
        centroids=init_centroids,
        change_sq=jnp.array(float("inf")),
    )

    def cond_fun(state: KMeansState) -> Bool[Scalar, "1"]:
        return (state.iteration < max_iters) & (state.change_sq > tol)

    def body_fun(state: KMeansState) -> KMeansState:
        assignments = assign_clusters(X, state.centroids)

        counts = jnp.bincount(assignments, length=k)

        sums = jax.ops.segment_sum(data=X, segment_ids=assignments, num_segments=k)

        safe_counts = jnp.maximum(counts, 1.0)

        new_centroids = sums / safe_counts[:, jnp.newaxis]

        is_empty = (counts == 0)[:, jnp.newaxis]
        final_new_centroids = jnp.where(is_empty, state.centroids, new_centroids)

        change_sq = jnp.sum((final_new_centroids - state.centroids) ** 2)

        return KMeansState(
            iteration=state.iteration + 1,
            centroids=final_new_centroids,
            change_sq=change_sq,
        )

    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    final_assignments = assign_clusters(X, final_state.centroids)

    return KMeansResult(centroids=final_state.centroids, assignments=final_assignments)


if __name__ == "__main__":
    # Parameters
    N_SAMPLES = 50_000
    N_FEATURES = 1000
    K_CLUSTERS = 500
    N_ITERS = 500
    N_BENCH_REPS = 10
    TOL = 1e-10

    print("K-Means Benchmark")
    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features")
    print(f"Parameters: {K_CLUSTERS} clusters, {N_ITERS} iterations")
    print(f"Averaging over {N_BENCH_REPS} runs.\n")

    # Data Setup
    key = jax.random.key(42)
    key, data_key = jax.random.split(key)

    # JAX data (lives on accelerator)
    X_jax = jax.random.normal(data_key, (N_SAMPLES, N_FEATURES))

    # NumPy data (lives on host CPU)
    X_np = np.array(X_jax)

    # JAX Benchmark
    print("1. Running JAX Benchmark...")

    # Warm-up run
    print("   Warming up / Compiling JAX function...")
    warmup_key, run_key = jax.random.split(key)
    result = jax_kmeans(warmup_key, X_jax, K_CLUSTERS, N_ITERS, TOL)
    result.assignments.block_until_ready()
    print("   Compilation complete.")

    # Timed runs
    jax_times = []
    for i in range(N_BENCH_REPS):
        run_key, subkey = jax.random.split(run_key)
        start_time = timeit.default_timer()

        result = jax_kmeans(subkey, X_jax, K_CLUSTERS, N_ITERS)
        result.assignments.block_until_ready()

        end_time = timeit.default_timer()
        jax_times.append(end_time - start_time)
        print(f"   JAX Run {i + 1}/{N_BENCH_REPS}: {jax_times[-1]:.6f} s")

    avg_jax_time = sum(jax_times) / N_BENCH_REPS

    # Scikit-learn Benchmark
    print("\n2. Running Scikit-learn Benchmark...")

    sklearn_times = []
    for i in range(N_BENCH_REPS):
        kmeans_sklearn = KMeans(
            n_clusters=K_CLUSTERS,
            init="random",
            n_init=1,
            max_iter=N_ITERS,
            tol=TOL,
            algorithm="lloyd",
            random_state=42 + i,
        )

        start_time = timeit.default_timer()
        kmeans_sklearn.fit(X_np)
        end_time = timeit.default_timer()

        sklearn_times.append(end_time - start_time)
        print(f"   Sklearn Run {i + 1}/{N_BENCH_REPS}: {sklearn_times[-1]:.6f} s")

    avg_sklearn_time = sum(sklearn_times) / N_BENCH_REPS

    # Results
    print("\nBenchmark Results")
    print(f"Average JAX time:       {avg_jax_time:.6f} seconds")
    print(f"Average Sklearn time:   {avg_sklearn_time:.6f} seconds")

    speedup = avg_sklearn_time / avg_jax_time
    print(f"\nJAX implementation is {speedup:.2f}x faster than sklearn.")
