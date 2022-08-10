from functools import partial
from typing import NamedTuple

import deq
import diffrax as dax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


@hk.transform
def model(x):
    def flatten(a):
        return a.reshape(a.shape[0], -1)

    x = (x - 33.3) / 78.6
    x = hk.Linear(128, name="downsample")(flatten(x))
    x = hk.LayerNorm(-1, True, True)(x)

    @deq.DEQCell
    def cell(x):
        z = hk.LayerNorm(-1, True, True)(x)
        z = hk.Linear(128, name="cell_proj")(z)
        z = hk.dropout(hk.next_rng_key(), 0.1, z)
        return z

    solver = deq.DEQSolver(wrap=dax.Tsit5(scan_stages=False))
    core = deq.DEQCore(cell, solver)
    z_init, z_star = map(flatten, core(x))
    logits = hk.Linear(10, name="classifier")(z_star)
    return {"z_init": z_init, "z_star": z_star, "logits": logits}


def loss_fn(params, rng, x, y, return_output=False):
    output = model.apply(params, rng, x)
    l_skip = 0.01 * jnp.abs(output["z_init"] - output["z_star"]).mean()
    p = output["logits"]
    t = jax.nn.one_hot(y, p.shape[-1])
    loss = optax.softmax_cross_entropy(p, t).mean() + l_skip
    output["l_skip"] = l_skip

    if return_output:
        return loss, output
    else:
        return loss


def chunks(arr, size, axis=0):
    return np.split(arr, range(size, len(arr), size), axis=axis)


def mkopt():
    return optax.rmsprop(1e-3)


def main():
    # jax.config.update("jax_platform_name", "cpu")
    dataset = np.load("mnist.npz")
    batchsz = 32
    X = dataset["train_images"]
    Y = dataset["train_labels"]
    X, Y = map(partial(chunks, size=batchsz), (X, Y))
    rng = hk.PRNGSequence(42)
    steps = len(X) * 24

    class TrainState(NamedTuple):
        params: optax.Params
        opt_st: optax.OptState
        step: int
        loss: float
        acc: float

    def train_init(rng, x):
        params = model.init(rng, x)
        opt_st = mkopt().init(params)
        return TrainState(params, opt_st, step=0, loss=0.0, acc=0.0)

    @partial(jax.jit, donate_argnums=0)
    def train_step(state, rng, x, y):
        (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, rng, x, y, return_output=True
        )
        updates, opt_st = mkopt().update(grads, state.opt_st, state.params)
        params = optax.apply_updates(state.params, updates)
        step_inc = optax.safe_int32_increment(state.step)
        loss_avg = (state.loss * state.step + loss) / step_inc
        acc = jnp.mean(output["logits"].argmax(-1) == y)
        acc_avg = (state.acc * state.step + acc) / step_inc
        return TrainState(params, opt_st, step=step_inc, loss=loss_avg, acc=acc_avg)

    def status(state):
        digits = np.ceil(np.log(steps) / np.log(10)).astype(int)
        step = str(state.step).rjust(digits, "0")
        total = str(steps).rjust(digits, "0")
        return f"step {step}/{total}: loss = {state.loss:.3g} acc = {state.acc:.3g}"

    def data():
        seed = jax.random.randint(next(rng), (), 0, np.iinfo(np.int32).max)
        nprng = np.random.default_rng(int(seed))
        itinerary = np.arange(len(X), dtype=int)
        while True:
            nprng.shuffle(itinerary)
            yield from ((X[i], Y[i]) for i in itinerary)

    state = train_init(next(rng), X[0])
    for i, (x, y) in enumerate(data()):
        state = train_step(state, next(rng), x, y)
        print(status(state), end=" " * 16 + "\r")
        if i == steps - 1:
            break


if __name__ == "__main__":
    main()
