import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', default='', type=str)
    parser.add_argument('--light_type', default='probes', type=str)
    parser.add_argument('--gs_path', default='', type=str)
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--probes_path', default='', type=str)
    parser.add_argument('--numProbes', type=int, default=8, help="choose 1/8/64")
    parser.add_argument('--iters', type=int, default=5, help="choose 0-10")
    parser.add_argument('--meshproxy_pitch', type=float, default=0.01)

    #### just for baking
    parser.add_argument('--probesW', type=int, default=400, help="width of probes")
    parser.add_argument('--probesH', type=int, default=400, help="height of probes")
    parser.add_argument('--begin_id', type=int, default=0)
    parser.add_argument('--scale_ratio', type=float, default=1.001)

    #### only for pipeline
    parser.add_argument('--just_render', action="store_true", default=False)

    opt = parser.parse_args()

    # StageI: Bake GaussProbe
    if not opt.just_render:
        os.system(
            f"python probes_bake.py \
                --gs_path {opt.gs_path} \
                --probes_path {opt.probes_path} \
                --mesh {opt.mesh} \
                --W {opt.probesW} \
                --H {opt.probesH} \
                --numProbes {opt.numProbes} \
                --begin_id {opt.begin_id} \
                --scale_ratio {opt.scale_ratio} \
                --meshproxy_pitch {opt.meshproxy_pitch}"
        )

    # StageII: Boot up the renderer
    os.system(
        f"python renderer.py \
            --mesh {opt.mesh} \
            --light_type {opt.light_type} \
            --gs_path {opt.gs_path} \
            --W {opt.W} \
            --H {opt.H} \
            --radius {opt.radius} \
            --fovy {opt.fovy} \
            --probes_path {opt.probes_path} \
            --numProbes {opt.numProbes} \
            --iters {opt.iters} \
            --meshproxy_pitch {opt.meshproxy_pitch}"
    )