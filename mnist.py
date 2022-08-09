from functools import partial
from typing import NamedTuple

import diffrax as dax
import haiku as hk
import haiku.data_structures as hkds
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sam
from deq import DEQCore, DEQSolver


@hk.transform
def model(x):
    x = (x - 127.5) / 255.0

    def flatten(a):
        return a.reshape(a.shape[0], -1)

    x = hk.Linear(128, name="downsample")(flatten(x))
    x = hk.LayerNorm(-1, True, True)(x)

    def cell(z):
        z = hk.Linear(128, name="cell_proj")(z)
        return hk.dropout(hk.next_rng_key(), 0.1, z)

    solver = DEQSolver(wrap=dax.Tsit5(scan_stages=False), atol=1e-4, max_depth=64)
    core = DEQCore(cell, solver)
    z_init, z_star = map(flatten, core(x))
    logits = hk.Linear(10, name="classifier")(z_star)
    return {"z_init": z_init, "z_star": z_star, "logits": logits}


def loss_fn(params, rng, x, y, return_output=False):
    def sup_con(p, t):
        p /= optax.safe_norm(p, 1e-4, axis=-1, keepdims=True)
        t /= t.sum(axis=-1, keepdims=True)
        return optax.softmax_cross_entropy(
            p @ p.swapaxes(-1, -2), t @ t.swapaxes(-1, -2)
        )

    output = model.apply(params, rng, x)

    logits = output["logits"]
    labels = jax.nn.one_hot(y, 10)
    latent = output["z_star"]

    rgterm = (
        0.01 * jnp.abs(output["z_init"] - output["z_star"]).sum()
        + 5e-3 * sup_con(latent, labels).mean()
    )
    loss = optax.softmax_cross_entropy(logits, labels).mean() + rgterm
    output["rgterm"] = rgterm

    if return_output:
        return loss, output
    else:
        return loss


def chunks(arr, size, axis=0):
    return np.split(arr, range(size, len(arr), size), axis=axis)


def mkopt(steps, rng, x, y):
    msched = optax.linear_onecycle_schedule(steps, 0.95, div_factor=0.95 / 0.85)
    lsched = lambda step: 1e-3  # optax.linear_onecycle_schedule(steps, 1e-3)
    dsched = lambda step: 1e-5  # optax.linear_onecycle_schedule(steps, 1e-4)

    def climb_fn(params):
        return jax.grad(loss_fn)(params, rng, x, y)

    def non_norm(module, name, value):
        del name
        del value
        return "norm" not in module.lower()

    return optax.chain(
        sam.look_sharpness_aware(climb_fn, rho=0.1, adaptive=True, skips=6),
        optax.inject_hyperparams(optax.scale_by_rms)(msched),
        optax.inject_hyperparams(optax.add_decayed_weights, static_args="mask")(
            dsched, mask=partial(hkds.map, non_norm)
        ),
        optax.scale_by_schedule(lsched),
        optax.scale(-1),
    )


def main():
    # jax.config.update("jax_platform_name", "cpu")
    dataset = np.load("mnist.npz")
    batchsz = 32
    X = chunks(dataset["train_images"], batchsz)
    Y = chunks(dataset["train_labels"], batchsz)

    class TrainState(NamedTuple):
        params: optax.Params
        opt_st: optax.OptState
        loss: float
        step: int

    def train_init(steps, rng, x, y):
        params = model.init(rng, x)
        opt_st = mkopt(steps, rng, x, y).init(params)
        return TrainState(params, opt_st, loss=0.0, step=0)

    @partial(jax.jit, donate_argnums=1, static_argnums=0)
    def train_step(steps, state, rng, x, y):
        (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, rng, x, y, return_output=True
        )
        params, opt_st = mkopt(steps, rng, x, y).update(
            grads, state.opt_st, state.params
        )
        step_inc = optax.safe_int32_increment(state.step)
        loss_avg = (state.loss * state.step + loss) / step_inc
        state = TrainState(params, opt_st, loss=loss_avg, step=step_inc)
        return state, output

    rng = hk.PRNGSequence(42)
    steps = len(X) * 24
    state = train_init(steps, next(rng), X[0], Y[0])

    acc_avg = 0

    def status(step, state, output, labels):
        nonlocal acc_avg
        digits = np.ceil(np.log(steps) / np.log(10)).astype(int)
        step = str(step).rjust(digits, "0")
        total = str(steps).rjust(digits, "0")
        acc = np.mean(output["logits"].argmax(axis=-1) == labels)
        acc_avg = ((state.step - 1) * acc_avg + acc) / state.step
        loss = state.loss - output["rgterm"]
        return f"step {step}/{total}: loss = {loss:.3g} acc = {acc_avg:.3g}"

    def data():
        while True:
            yield from zip(X, Y)

    for i, (x, y) in enumerate(data()):
        state, output = train_step(steps, state, next(rng), x, y)
        print(status(i + 1, state, output, y), end=" " * 16 + "\r")
        if i == steps - 1:
            break


if __name__ == "__main__":
    main()
