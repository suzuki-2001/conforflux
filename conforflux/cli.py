from __future__ import annotations

import sys

import click

from conforflux.config import ConforFluxConfig


@click.group()
def cli() -> None:
    pass


@cli.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
)
@click.argument("data", type=click.Path(exists=True))
@click.option("--num_particles", type=int, default=5, show_default=True,
              help="Number of coupled particles (M).")
@click.option("--sigma", type=float, default=2.0, show_default=True,
              help="RBF kernel bandwidth on Cα RMSD (Å).")
@click.option("--alpha_s", type=float, default=0.02, show_default=True,
              help="RMS-normalised step size for s_trunk.")
@click.option("--alpha_z", type=float, default=0.02, show_default=True,
              help="RMS-normalised step size for z_trunk.")
@click.option("--start_frac", type=float, default=0.0, show_default=True,
              help="Guidance starts at this fraction of the trajectory.")
@click.option("--stop_frac", type=float, default=0.8, show_default=True,
              help="Guidance stops at this fraction of the trajectory.")
@click.option("--update_interval", type=int, default=3, show_default=True,
              help="Fire the gradient every K diffusion steps.")
@click.pass_context
def predict(
    ctx: click.Context,
    data: str,
    num_particles: int,
    sigma: float,
    alpha_s: float,
    alpha_z: float,
    start_frac: float,
    stop_frac: float,
    update_interval: int,
) -> None:
    """Run ConforFlux prediction; extra flags are forwarded to `boltz predict`."""
    config = ConforFluxConfig(
        sigma=sigma,
        alpha_s=alpha_s,
        alpha_z=alpha_z,
        start_frac=start_frac,
        stop_frac=stop_frac,
        update_interval=update_interval,
    )
    _install_callback(config, num_particles)
    _run_boltz_predict([data, *ctx.args])


def _install_callback(config: ConforFluxConfig, num_particles: int) -> None:
    from pytorch_lightning import Trainer

    from conforflux.callback import ConforFluxCallback

    orig_init = Trainer.__init__

    def patched_init(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks.append(ConforFluxCallback(config, num_particles=num_particles))
        kwargs["callbacks"] = callbacks
        orig_init(self, *args, **kwargs)

    Trainer.__init__ = patched_init


def _run_boltz_predict(argv: list[str]) -> None:
    from boltz.main import cli as boltz_cli

    try:
        boltz_cli.main(["predict", *argv], standalone_mode=False)
    except SystemExit as exc:
        sys.exit(exc.code)


if __name__ == "__main__":
    cli()
