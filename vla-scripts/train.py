import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import yaml
import transformers

from metis.conf import VLAConfig, VLARegistry
from metis.models import load, load_vla
from metis.overwatch import initialize_overwatch
from metis.training import VLAMetrics, get_train_strategy
from metis.util import set_global_seed
from metis.vla import get_latent_vla_dataset_and_collator, get_cot_latent_vla_dataset_and_collator
from metis.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from metis.models.vlas.dexvla import DexVLA_Model

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_BRIDGE.vla_id)
    )
    # pretrain_vlm: str = '/share/project/fyk/phi-2+3b/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/phi-2+3b'
    pretrain_vlm: str = '/share/project/zjk/prism-dinosiglip-224px+7b/models--TRI-ML--prismatic-vlms/snapshots/a3ba8a19c453a82eaf5a3fb1e699dd9e441f0a12/prism-dinosiglip-224px+7b'
    motion_dynamics_path: str = "/share/project/fyk/UniDex/ckpt/rq_lam_chunk32/epoch=2-step=300000.ckpt"

    motion_dynamics_type: str = "RQ_HandLatentActionModel"


    # Dexterous LAM setting
    codebook_size: int = 16
    motion_codebook_size: int = 512
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 8
    lam_num_heads: int = 12

    use_diffusion_head: bool = False
    window_size: int = 32

    data_root_dir: Path = Path(                                     # Path to lerobot cot dataset directory
        "path/to/egoatlas/"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    pretrained_checkpoint = None
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                           # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 20000                                     # Interval for saving checkpoints (in steps)
    image_aug: bool = True                                          # Whether to enable image augmentations
    seed: int = 23                                                  # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = 'your_hf_token'

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "METIS-pretrain"                   # Name of W&B project to log to (use default!)
    wandb_entity: str = "your_wandb_entity"                              # Name of entity to log under

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on

def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    cfg: Optional[TrainConfig] = None,
    dex_token: bool = False,
    motion_RQ: bool = False,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))
    if dex_token:
        special_tokens_dict = {'additional_special_tokens': ['<BOA>', '<BOR>'] + [f'<VIDEO_{i}>' for i in range(cfg.codebook_size)] + [f'<DEX_{i}>' for i in range(cfg.codebook_size)]}
    elif motion_RQ:
        special_tokens_dict = {'additional_special_tokens': ['<BOA>', '<BOR>'] + [f'<DYN_{i}>' for i in range(cfg.codebook_size)] + [f'<DEX_{i}>' for i in range(cfg.motion_codebook_size)]}
    else:
        special_tokens_dict = {'additional_special_tokens': ['<BOA>', '<BOR>'] + [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("Metis Training :: Warming Up")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    cfg.run_id += "-" + cfg.motion_dynamics_type
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    cfg.run_id += '-Latent-Action-Pretraining-6dRot_chunk32_rebuttal_human+robot'
    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    hf_token = cfg.hf_token
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        vla = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=True, cache_dir=cfg.pretrain_vlm, action_codebook_size=cfg.codebook_size, use_diffusion_head=cfg.use_diffusion_head, window_size=cfg.window_size)
    
    else:
        vlm = load(cfg.pretrain_vlm, hf_token=hf_token, load_for_training=True, cache_dir=cfg.pretrain_vlm, window_size=cfg.window_size)
        vla = DexVLA_Model(vlm=vlm, use_diffusion_head=cfg.use_diffusion_head, window_size=cfg.window_size)
        del vlm

    
    if cfg.motion_dynamics_type == "UncontrolledHandVideo_LatentActionModel":
        dex_token = True
        smart_tokenizer_and_embedding_resize(tokenizer = vla.vlm.llm_backbone.get_tokenizer(), model = vla.vlm.llm_backbone.llm, cfg=cfg, dex_token=dex_token)
    elif cfg.motion_dynamics_type == "RQ_HandLatentActionModel":
        motion_RQ = True
        smart_tokenizer_and_embedding_resize(tokenizer = vla.vlm.llm_backbone.get_tokenizer(), model = vla.vlm.llm_backbone.llm, cfg=cfg, motion_RQ=motion_RQ)
    else:
        smart_tokenizer_and_embedding_resize(tokenizer = vla.vlm.llm_backbone.get_tokenizer(), model = vla.vlm.llm_backbone.llm, cfg=cfg)
    # [Validate] Model should be in Full Precision!
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    from motion_tokenizer.genie.modules.motion_tokenizer import RQ_HandLatentActionModel
    
    if cfg.motion_dynamics_type == "RQ_HandLatentActionModel":
        latent_action_model = RQ_HandLatentActionModel(
            in_dim=3,
            model_dim=cfg.lam_model_dim,
            latent_dim=cfg.lam_latent_dim,
            num_latents=cfg.codebook_size,
            patch_size=cfg.lam_patch_size,
            enc_blocks=cfg.lam_enc_blocks,
            dec_blocks=cfg.lam_dec_blocks,
            num_heads=cfg.lam_num_heads,
            dropout=0.,
        )
    else:
        raise TypeError("Unsupported action_tokenizer type.")


    lam_ckpt = torch.load(cfg.motion_dynamics_path, map_location='cpu')['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, action_tokenizer, collator = get_cot_latent_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vla.vlm.vision_backbone.get_image_transform(),
        image_transform_lam=transforms.ToTensor(),
        latent_action_tokenizer=latent_action_model,
        tokenizer=vla.vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size,
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vla=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Latent Action Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=cfg.save_interval,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()