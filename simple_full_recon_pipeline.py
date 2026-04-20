import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Simple Full Reconstruction script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--probesW', type=int, default=400, help="width of probes")
    parser.add_argument('--probesH', type=int, default=400, help="height of probes")
    parser.add_argument('--numProbes', type=int, default=8, help="choose 1/8/64")
    parser.add_argument('--begin_id', type=int, default=0)
    parser.add_argument('--scale_ratio', type=float, default=1.001)
    parser.add_argument('--meshproxy_pitch', type=float, default=0.01)

    #### skip baking
    parser.add_argument('--skip_bake', action="store_true", default=False)


    #### rendering
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--viewer', action="store_true", default=False)
    parser.add_argument('--viewer_mode', default="fine", type=str, help="choose coarse/fine")

    #### just_render
    parser.add_argument('--just_render', action="store_true", default=False)

    # sdf refine
    parser.add_argument('--sdf_end_iter', type=int, default=20_000) # 10_000, 15_000, 20_000
    parser.add_argument('--mesh_end_iter', type=int, default=7_000)
    parser.add_argument('--mesh_r', type=int, default=2)

    #### original
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    if not args.just_render:
        os.system(
            f"python trainer.py \
                -m {dataset.model_path} \
                -s {dataset.source_path} \
                --iterations 7000"
        )

        os.system(
            f"python coarsemesh_gs.py \
                -m {dataset.model_path}"
        )

        if not args.skip_bake: # moderate the aabb of probes if you wanna get more accurate results
            os.system(
                f"""python probes_bake.py \
                    --gs_path {os.path.join(dataset.source_path, "env_gs.ply")} \
                    --probes_path {os.path.join(dataset.model_path, "probes")} \
                    --mesh {os.path.join(dataset.model_path, "train", "ours_7000", "fuse.ply")} \
                    --W {args.probesW} \
                    --H {args.probesH} \
                    --numProbes {args.numProbes} \
                    --begin_id {args.begin_id} \
                    --scale_ratio {args.scale_ratio} \
                    --meshproxy_pitch {args.meshproxy_pitch} \
                    --probes_black"""
            )
        
        os.system(
            f"""python finemesh_sdf.py \
                --conf ./sdfproxy/confs/base.conf \
                --mode train \
                --case {"finemesh_sdf"} \
                --train_end_iter {args.sdf_end_iter} \
                --data_dir {dataset.source_path} \
                --base_exp_dir {os.path.join(dataset.model_path, "sdf_proxy")} \
                --mesh {os.path.join(dataset.model_path, "train", "ours_7000", "fuse.ply")} """
        )

        os.system(
            f"""python trainer.py \
                -m {dataset.model_path} \
                -s {dataset.source_path} \
                --stage 2 \
                --mesh {os.path.join(dataset.model_path, "sdf_proxy", "meshes", f"{args.sdf_end_iter:08d}.ply")} \
                --probes_path {os.path.join(dataset.model_path, "probes")} \
                --iterations {args.mesh_end_iter} \
                -r {args.mesh_r}"""
        )

    if args.viewer or args.just_render:
        a = os.path.join(dataset.model_path, "train", "ours_7000", "fuse.ply")
        b = os.path.join(dataset.model_path, "sdf_proxy", "meshes", f"{args.sdf_end_iter:08d}.ply")
        c = b
        for i in range(args.mesh_end_iter, 0, -1000):
            d = os.path.join(dataset.model_path, "mesh", f"iteration_{i}", "mesh.ply")
            if os.path.exists(d):
                c = d
                break

        os.system(
            f"""python renderer.py \
                --mesh {a if args.viewer_mode == "coarse" else c} \
                --gs_path {os.path.join(dataset.source_path, "env_gs.ply")} \
                --W {args.W} \
                --H {args.H} \
                --probes_path {os.path.join(dataset.model_path, "probes")} \
                --numProbes {args.numProbes} \
                --iters 64 \
                --meshproxy_pitch {args.meshproxy_pitch} """
        )