import argparse
import os
from glob import glob
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from src.general_utils import set_seed, rank0_log
from src.sp_utils.parallel_states import initialize_parallel_state
from src.data_utils import test_pcd_render
from diffusers.utils import export_to_video
import wandb
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="worldstereo-camera",
                        choices=["worldstereo-camera", "worldstereo-memory", "worldstereo-memory-dmd"],
                        help="Model type (e.g., 'worldstereo-camera', 'worldstereo-memory', 'worldstereo-memory-dmd')")
    parser.add_argument("--input_path", type=str, default="examples/images", help="image input path")
    parser.add_argument("--scene_name", type=str, default=None,
                        help="Process only this scene directory below input_path")
    parser.add_argument("--output_path", type=str, default="outputs", help="Target path for output results")
    parser.add_argument("--local_files_only", action="store_true", help="If True, avoid downloading the file and return the path to the local cached file if it exists.")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP model sharding")
    parser.add_argument("--sequential_cpu_offload", action="store_true",
                        help="Load pipeline components on demand to reduce GPU memory use")
    parser.add_argument("--vae_tiling", action="store_true",
                        help="Enable VAE tiling when supported by the loaded VAE")
    parser.add_argument("--bypass-depth-and-initial-frame", action="store_true", help="Bypass deptha and using input image as first frame, for satellite input")
    parser.add_argument("--seed", default=1024, type=int, help="Random seed")
    args = parser.parse_args()

    wandb.init(project="worldstereo")
    pipeline_step = 0

    pipeline_step += 1
    wandb.log({"pipeline_step": pipeline_step, "pipeline_stage": "starting"})

    # ── distributed setup ─────────────────────────────────────────────
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
    )
    device_num = torch.cuda.device_count()
    device_mesh = init_device_mesh(
        "cuda",
        (world_size // device_num, device_num),
        mesh_dim_names=("rep", "shard"),
    )
    rank0_log(f"World size: {world_size}")

    # ── sequence parallel ──────────────────────────────────────────────
    parallel_dims = initialize_parallel_state(sp=world_size)
    sp_size = parallel_dims.sp if parallel_dims.sp_enabled else 1
    sp_rank = parallel_dims.sp_rank if parallel_dims.sp_enabled else 0
    data_rank = dist.get_rank() // sp_size

    global_seed = args.seed + data_rank
    set_seed(global_seed)
    print(f"Global rank:{dist.get_rank()}, Local rank:{local_rank}, "
          f"SP_rank:{sp_rank}, SP_group:{data_rank}, seed:{global_seed}.")

    # ── model loading ──────────────────────────────────────────────────
    torch.set_default_dtype(torch.float32)

        # worldstereo = WorldStereo.from_pretrained(
        #     "hanshanxue/WorldStereo",
        #     subfolder=args.model_type,
        #     local_files_only=args.local_files_only,
        #     sp_world_size=sp_size,
        #     fsdp=args.fsdp,
        #     device_mesh=device_mesh,
        #     device=device,
        #     sequential_cpu_offload=args.sequential_cpu_offload,
        #     vae_tiling=args.vae_tiling,
        # )
    config = json.load(open(f"/home/languo/.cache/huggingface/hub/models--hanshanxue--WorldStereo/snapshots/ac2ad97ecb043fe80c2f19cd1898006becb9d66e/{args.model_type}/config.json", "r"))

    pipeline_step += 1
    wandb.log({"pipeline_step": pipeline_step, "pipeline_stage": "worldstereo_loaded"})
    # moge is used for warp rendering
    depth_model = None

    pipeline_step += 1
    wandb.log({"pipeline_step": pipeline_step, "pipeline_stage": "moge_loaded"})

    # ── inference ──────────────────────────────────────────────────────
    dist.barrier()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Auto-select autocast precision: prefer bf16, then fp16, fall back to fp32 (disable autocast)
    if torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif torch.cuda.get_device_capability(device)[0] >= 7:  # fp16 requires SM >= 70
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None  # no half-precision support, fall back to fp32
    rank0_log(f"Autocast dtype: {autocast_dtype if autocast_dtype else 'disabled (fp32)'}")

    # load data
    if args.scene_name is not None:
        scene_list = [os.path.join(args.input_path, args.scene_name)]
        if not os.path.isdir(scene_list[0]):
            raise FileNotFoundError(f"Scene directory does not exist: {scene_list[0]}")
    elif os.path.exists(f"{args.input_path}/image.png"):
        scene_list = [args.input_path]
    else:
        scene_list = [path for path in glob(f"{args.input_path}/*") if os.path.isdir(path)]
    scene_list.sort()
    rank0_log(f"Building dataset. {len(scene_list)} scenes found.")

    for scene_path in tqdm(scene_list):
        scene_name = os.path.basename(scene_path)
        rank0_log(f"Processing scene {scene_name}.")
        output_path = f"{args.output_path}/{scene_name}"
        if rank == 0:
            os.makedirs(output_path, exist_ok=True)
        with torch.no_grad():
            wandb.log({"pipeline_step": pipeline_step, "pipeline_stage": "dataset_load_start", "scene": scene_name})
            test_pcd_render(cfg=config, input_path=scene_path, output_path=output_path,
                                              model_type=args.model_type, depth_model=depth_model, device=device,
                                              sp_size=sp_size, sp_rank=sp_rank, pipeline_step=pipeline_step, bypass_depth_and_initial_frame=args.bypass_depth_and_initial_frame)

    if dist.is_initialized():
        dist.destroy_process_group()
