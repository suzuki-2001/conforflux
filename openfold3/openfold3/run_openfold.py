# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""

Main run script for OpenFold3. Please see the README for usage details.

"""
# ruff: noqa: F821

import logging
from pathlib import Path

import click

from openfold3.core.config import config_utils
from openfold3.entry_points.import_utils import _torch_gpu_setup

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Yaml that specifies model and dataset parameters,"
    " see examples/training_new.yml",
)
@click.option("--seed", type=int, help="Initial seed for all processes")
@click.option(
    "--data-seed",
    "--data_seed",
    type=int,
    help="Initial seed for data pipeline. Defaults to seed if not specified.",
)
def train(runner_yaml: Path, seed: int | None = None, data_seed: int | None = None):
    """Perform a training experiment with a preprepared dataset cache."""
    _torch_gpu_setup()
    from openfold3.entry_points.experiment_runner import (
        TrainingExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        TrainingExperimentConfig,
    )

    runner_dict = config_utils.load_yaml(runner_yaml)

    # overwrite seed defaults if provided:
    if seed is not None:
        runner_dict["experiment_settings"]["seed"] = seed

    if data_seed is not None:
        runner_dict["data_module_args"]["data_seed"] = data_seed

    expt_config = TrainingExperimentConfig.model_validate(runner_dict)

    expt_runner = TrainingExperimentRunner(expt_config)
    expt_runner.setup()
    expt_runner.run()


@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--inference-ckpt-path",
    "--inference_ckpt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Path for model checkpoint to be used for inference. "
    "If not specified, will attempt to find or download parameters to "
    "$OPENFOLD_CACHE [default: ~/.openfold3/]",
)
@click.option(
    "--inference-ckpt-name",
    "--inference_ckpt_name",
    type=str,
    required=False,
    help="Name of the checkpoint to be used for inference."
    " Only used if `inference_ckpt_path` is not specified.",
)
@click.option(
    "--num-diffusion-samples",
    "--num_diffusion_samples",
    type=int,
    default=None,
    required=False,
    help="Number of diffusion samples to generate for each query.",
)
@click.option(
    "--num-model-seeds",
    "--num_model_seeds",
    type=int,
    default=None,
    required=False,
    help="Number of model seeds to use for each query.",
)
@click.option(
    "--seeds",
    type=str,
    default=None,
    required=False,
    help="Explicit model seeds, comma-separated (e.g. 101,102,103). Takes precedence over "
    "--num_model_seeds, which can only draw its seeds from a fixed start seed. Without "
    "this the seeds come from the runner yaml as before.",
)
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option(
    "--use-msa-server",
    "--use_msa_server",
    type=bool,
    default=True,
    help="Use ColabFold MSA server to perform alignments.",
)
@click.option(
    "--use-templates",
    "--use_templates",
    type=bool,
    default=True,
    help="Use ColabFold MSA server to perform template alignments.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Output directory for writing results",
)
@click.option(
    "--num-particles",
    "--num_particles",
    type=int,
    default=5,
    help="ConforFlux coupled particles M; replaces --num_diffusion_samples. "
    "0 runs OpenFold3 unguided.",
)
@click.option("--sigma", type=float, default=2.0,
              help="ConforFlux: RBF kernel bandwidth on Ca RMSD (A).")
@click.option("--alpha-s", "--alpha_s", "alpha_s", type=float, default=0.02,
              help="ConforFlux: RMS-normalised step size for si_trunk.")
@click.option("--alpha-z", "--alpha_z", "alpha_z", type=float, default=0.02,
              help="ConforFlux: RMS-normalised step size for zij_trunk.")
@click.option("--update-interval", "--update_interval", "update_interval", type=int, default=3,
              help="ConforFlux: fire the gradient every K diffusion steps.")
@click.option("--noise-level/--no-noise-level", "noise_level", default=False,
              help="ConforFlux: scale the step by the noise level of the step it is applied at. "
                   "Pushes harder, at a higher clashscore.")
@click.option("--gradient-checkpointing", "--gradient_checkpointing", "gradient_checkpointing",
              is_flag=True, default=False,
              help="ConforFlux: recompute diffusion-transformer blocks in the backward pass "
                   "instead of storing them, to reduce peak GPU memory on large targets.")
def predict(
    query_json: Path,
    inference_ckpt_path: Path | None = None,
    inference_ckpt_name: str | None = None,
    num_diffusion_samples: int | None = None,
    num_model_seeds: int | None = None,
    seeds: str | None = None,
    runner_yaml: Path | None = None,
    use_msa_server: bool = True,
    use_templates: bool = True,
    output_dir: Path | None = None,
    num_particles: int = 5,
    sigma: float = 2.0,
    alpha_s: float = 0.02,
    alpha_z: float = 0.02,
    update_interval: int = 5,
    noise_level: bool = False,
    gradient_checkpointing: bool = False,
):
    """Perform inference on a set of queries defined in the query_json."""
    _torch_gpu_setup()

    from openfold3.entry_points.experiment_runner import (
        InferenceExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        InferenceExperimentConfig,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    logging.basicConfig(level=logging.INFO)
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()

    expt_config = InferenceExperimentConfig(
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
        **runner_args,
    )
    expt_runner = InferenceExperimentRunner(
        expt_config,
        num_particles if num_particles > 0 else num_diffusion_samples,
        num_model_seeds,
        use_msa_server,
        use_templates,
        output_dir,
    )

    # The seeds the sampler is reseeded with come from experiment_settings, not from the
    # query json (whose `seeds` field only names the output subdirectories), so an explicit
    # list has to be set on the runner before the data module reads it.
    if seeds:
        expt_runner.seeds = [int(s) for s in seeds.split(",") if s.strip()]
        logger.info(f"Model seeds set from --seeds: {expt_runner.seeds}")

    # Load inference query set
    query_set = InferenceQuerySet.from_json(query_json)

    # Run the forward pass
    expt_runner.setup()

    if num_particles > 0:
        from openfold3.conforflux_guidance_of3 import ConforFluxConfig

        model = expt_runner.lightning_module.model
        model.sample_diffusion.conforflux_config = ConforFluxConfig(
            sigma=sigma,
            alpha_s=alpha_s,
            alpha_z=alpha_z,
            update_interval=update_interval,
            noise_scale=noise_level,
        )
        if gradient_checkpointing:
            # use_reentrant must stay False: the guidance takes its gradient through
            # torch.autograd.grad, which reentrant checkpointing does not support.
            dt = model.diffusion_module.diffusion_transformer
            dt.blocks_per_ckpt = 1
            dt.use_reentrant = False

    expt_runner.run(query_set)
    expt_runner.cleanup()


@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory for writing alignments",
)
@click.option(
    "--msa-computation-settings-yaml",
    "--msa_computation_settings_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml file to customize Colabfold MSA settings,"
    " see MsaComputationSettings for options.",
)
def align_msa_server(
    query_json: Path,
    output_dir: Path,
    msa_computation_settings_yaml: Path | None = None,
):
    """Run MSA server alignment only with ColabFold MSA server.
    
    Example command:
    python run_openfold.py align-msa-server \
        --query_json query_example.json \
        --output_dir output/msa_server_test \
    
    More settings can be specified using the `msa_computation_settings_yaml` flag
    An example yaml file is provided in `examples/msa_server.yml`
    """
    _torch_gpu_setup()
    from openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        preprocess_colabfold_msas,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    query_set = InferenceQuerySet.from_json(query_json)

    msa_settings = MsaComputationSettings.from_config_with_cli_override(
        output_dir, msa_computation_settings_yaml
    )
    query_set = preprocess_colabfold_msas(
        inference_query_set=query_set,
        compute_settings=msa_settings,
    )

    with open(output_dir / "query_msa.json", "w") as fp:
        fp.write(query_set.model_dump_json(indent=4))


if __name__ == "__main__":
    cli()
