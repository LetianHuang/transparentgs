<p align="center">
<h1 align="center"><strong>TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians</strong></h1>
<h3 align="center">SIGGRAPH 2025 <br> (ACM Transactions on Graphics)</h3>

<p align="center">
              <span class="author-block">
                <a href="https://letianhuang.github.io/">Letian Huang</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <span class="author-block">
                <a href="https://orcid.org/0009-0004-8637-4384">Dongwei
                  Ye</a><sup>1</sup></span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              <span class="author-block">
                <a href="https://orcid.org/0009-0007-2228-4648">Jialin Dan</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <span class="author-block">
                <a href="https://orcid.org/0000-0002-0736-7951">Chengzhi
                  Tao</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <span class="author-block">
                <a href="https://orcid.org/0009-0005-6423-4812">Huiwen Liu</a><sup>2</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <br>
              <span class="author-block">
                <a href="http://kunzhou.net/">Kun Zhou</a><sup>3,4</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <span class="author-block">
                <a href="http://ren-bo.net/">Bo Ren</a><sup>2</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <span class="author-block">
                <a href="http://www.njumeta.com/liyq/">Yuanqi Li</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <span class="author-block">
                <a href="https://cs.nju.edu.cn/ywguo/index.htm">Yanwen Guo</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
              <span class="author-block">
                <a href="https://scholar.google.com.hk/citations?user=Sx4PQpQAAAAJ&hl=en">Jie Guo</a><sup>*
                  1</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
              </span>
    <br>
              <span class="author-block"><sup>1</sup>State Key Lab for Novel Software Technology, Nanjing
                University</span><br>
              <span class="author-block"><sup>2</sup>TMCC, College of Computer Science, Nankai University</span><br>
              <span class="author-block"><sup>3</sup>State Key Lab of CAD&CG, Zhejiang University</span><br>
              <span class="author-block"><sup>4</sup>Institute of Hangzhou Holographic Intelligent Technology</span>
</p>

<div align="center">
    <a href='https://doi.org/10.1145/3730892'><img src='https://img.shields.io/badge/DOI-10.1145%2F3730892-blue'></a>
    <a href=https://arxiv.org/abs/2504.18768><img  src='https://img.shields.io/badge/arXiv-2504.18768-b31b1b.svg'></a>
    <a href='https://letianhuang.github.io/transparentgs'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://www.youtube.com/watch?v=HfHC0wNYry8&t=130s'><img src='https://img.shields.io/badge/YouTube-SIGGRAPH%20Trailer-red?logo=youtube&logoColor=white'></a>
</div>

</p>

![teaser](https://github.com/LetianHuang/LetianHuang.github.io/blob/main/assets/img/transparent_teaser.png)

## News

**[2025.08.04]** 🎈 We release the code.

**[2025.07.23]** <img class="emoji" title=":smile:" alt=":smile:" src="https://github.githubassets.com/images/icons/emoji/unicode/1f604.png" height="20" width="20"> Birthday of the repository.

## TL;DR

We propose TransparentGS, a fast inverse rendering pipeline for transparent objects based on 3D-GS. The main contributions are three-fold: efficient transparent Gaussian primitives for specular refraction, GaussProbe to encode ambient light and nearby contents, and the IterQuery algorithm to reduce parallax errors in our probe-based framework.

## Overview

The overview of our TransparentGS pipeline. Each 3D scene is firstly separated into transparent objects and opaque environment using SAM2 [Ravi et al. 2024] guided by GroundingDINO [Liu et al. 2024]. For transparent objects, we propose transparent Gaussian primitives, which explicitly encode both geometric and material properties within 3D Gaussians. And the properties are rasterized into maps for subsequent deferred shading. For the opaque environment, we recover it with the original 3D-GS, and bake it into GaussProbe surrounding the transparent object. The GaussProbe are then queried through our IterQuery algorithm to compute reflection and refraction.

![pipeline](https://letianhuang.github.io/transparentgs/static/images/exp/pipeline.png)

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{transparentgs,
    author = {Huang, Letian and Ye, Dongwei and Dan, Jialin and Tao, Chengzhi and Liu, Huiwen and Zhou, Kun and Ren, Bo and Li, Yuanqi and Guo, Yanwen and Guo, Jie},
    title = {TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians},
    journal = {ACM Transactions on Graphics (TOG)},
    number = {4},
    volume = {44},
    month = {July},
    year = {2025},
    pages = {1--17},
    url = {https://doi.org/10.1145/3730892},
    publisher = {ACM New York, NY, USA}
}
```

## TransparentGS Viewer (Renderer)

![TransparentGS Renderer](assets/TransparentGS_viewer_utility.png)

### Utility

- [x] Real-time rendering and navigation of scenes that integrate traditional [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), triangle meshes and reconstructed meshes (Highly robust to complex occlusions).
- [x]  Secondary light effects (e.g., reflection and refraction).
- [x] Rendering with non-pinhole camera models (e.g., fisheye or panorama).
- [x] Material editing (e.g., IOR and base color).

### Cloning the Repository and Setup

Clone the repository and create an anaconda environment using

``` shell
git clone git@github.com:LetianHuang/transparentgs.git --recursive
cd transparentgs

SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate transparentgs
```

The repository contains several submodules, thus please check it out with

``` shell
pip install . # Thanks to https://github.com/ashawkey/raytracing
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization-fisheye 
pip install submodules/diff-gaussian-rasterization-panorama
pip install submodules/nvdiffrast
```

or choose a faster version (1. integrated with [Speedy-Splat](https://github.com/j-alex-hanson/speedy-splat), using SnugBox and AccuTile; 2. Employ CUDA scripting for computational acceleration of 64 probes).

``` shell
pip install . 
pip install submodules-speedy/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules-speedy/diff-gaussian-rasterization-fisheye
pip install submodules-speedy/diff-gaussian-rasterization-panorama
pip install submodules-speedy/compute-trilinear-weights
pip install submodules/nvdiffrast
```

### Scene Assets

First, create a `models` folder inside the project path by

```shell
mkdir models
```

The data structure will be organised as follows: 

```
transparentgs/
│── models/
│   ├── 3dgs/
│   │   ├── drjohnson.ply
│   │   ├── playroom_lego_hotdog_mouse.ply
│   │   ├── Matterport3D_h1zeeAwLh9Z_3.ply
│   │   ├── ...
│   ├── mesh/
│   │   ├── ball.ply
│   │   ├── mouse.ply
│   │   ├── bunny.ply
│   │   ├── ...
│   ├── probes/
│   │   ├── playroom_lego_hotdog_mouse/
│   │   │   ├── probes/
│   │   │   │   ├── 000_depth.exr
│   │   │   │   ├── 000.exr
│   │   │   │   ├── 333_depth.exr
│   │   │   │   ├── 333.exr
│   │   │   │   ├── ...
│   │   │   ├── probe.json
│   │   ├── ...
|   ├── meshgs_proxy/
│   │   ├── mouse.ply
│   │   ├── ...
```

#### Public scene

We release several ready-to-use scenes. Please download the assets from [Google Drive](https://drive.google.com/drive/folders/1SS7E74DapiaBWNOMLp-n0FP1hqjX42cK?usp=sharing) and move the `3dgs` and `mesh` folders into `models/` folder.

#### Custom scene

To create a custom scene, simply follow the provided instructions to set it up. Instructions on the above data structure are as follows:

1. Scenes in the `3dgs` folder should be in `.ply` format and reconstructed using traditional [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [op43dgs](https://github.com/LetianHuang/op43dgs) (for reconstruction from non-pinhole cameras) or [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/) (for anti-alias).
2. Objects in the `mesh` folder could be in any triangle mesh format (e.g, `.obj`, `.ply` or `.glb`), including both traditional and reconstructed ones.
3. Probes in the `probes` folder could be baked using `Step I: Bake GaussProbe` or similar formats. The `probes.json` file specifies the positions of the probes, while the `probes/` directory stores the corresponding RGB panorama and depth panorama in EXR format.
4. The `meshgs_proxy` folder is a byproduct of `Step I: Bake GaussProbe`. It contains the object converted into 3DGS format and can be used as a proxy of the mesh in `mesh` to assemble a new scene (`mesh` + `3DGS`). Note: modifying the files in `meshgs_proxy` does not affect the final rendering results (i.e., `Step II: Boot up the renderer`). To change the proxy configuration, you can adjust the scene’s position under the 3dgs directory and rerun `Step I: Bake GaussProbe`.



### Step I: Bake GaussProbe

The first step is to bake probes for the scene that has already been set up:

``` shell
python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --begin_id 0
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for probes_bake.py</span></summary>

  #### --gs_path 
  path to the trained 3D Gaussians directory as the environment (used to bake GaussProbe).
  #### --probes_path
  output path of GaussProbe to be baked
  #### --mesh
  path to the mesh
  #### --W
  width of the RGBD panorama
  #### --H
  height of the RGBD panorama
  #### --numProbes
  number of probes (1/8/64). In theory, any positive integer is allowed, but the released code only supports these three fixed values.
  #### --begin_id
  only to prevent OOM (Out of Memory); when GPU memory is insufficient, the process can exit and resume baking from the specified ID.
  #### --scale_ratio
  bounding box scale ratio for the mesh
  #### --meshproxy_pitch
  the voxel size (pitch), which determines the resolution of the mesh voxelization.

</details>

### Step II: Boot up the renderer

Next, boot the renderer to start rendering:

``` shell
python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --meshproxy_pitch 0.1
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for renderer.py</span></summary>

  #### --mesh
  path to the mesh
  #### --light_type
  the original design supports either environment map or GaussProbe. However, since a single probe with zero iteration is equivalent to the environment map, this design has been deprecated.
  #### --gs_path 
  path to the trained 3D Gaussians directory as the environment (used to bake GaussProbe).
  #### --W
  GUI width
  #### --H
  GUI height
  #### --radius
  default GUI camera radius from center
  #### --fovy
  default GUI camera fovy (can be modified in the GUI)
  #### --probes_path
  path of the baked GaussProbe
  #### --numProbes
  number of probes (1/8/64). In theory, any positive integer is allowed, but the released code only supports these three fixed values. (can be modified in the GUI)
  #### --iters
  count of iterations (0-10). In theory, any non-negative integer is allowed, but the released code only supports these eleven fixed values. (can be modified in the GUI)
  #### --meshproxy_pitch
  the voxel size (pitch), which determines the resolution of the mesh voxelization.

</details>

### Full Pipeline (optional)

Additionally, we offer an optional all-in-one pipeline script that produces the same effect as executing `Step I` and `Step II` independently:

``` shell
python full_render_pipeline.py --W 960 --H 540 --probesW 800 --probesH 800 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --meshproxy_pitch 0.1
# equal to
# 1. python probes_bake.py --W 800 --H 800 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --begin_id 0 --meshproxy_pitch 0.1
# 2. python renderer.py --W 960 --H 540 --gs_path ./models/3dgs/playroom_lego_hotdog_mouse.ply --probes_path ./models/probes/playroom_lego_hotdog_mouse --mesh ./models/mesh/mouse.ply --meshproxy_pitch 0.1
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for renderer.py</span></summary>

  #### --mesh
  path to the mesh
  #### --light_type
  the original design supports either environment map or GaussProbe. However, since a single probe with zero iteration is equivalent to the environment map, this design has been deprecated.
  #### --gs_path 
  path to the trained 3D Gaussians directory as the environment (used to bake GaussProbe).
  #### --W
  GUI width
  #### --H
  GUI height
  #### --radius
  default GUI camera radius from center
  #### --fovy
  default GUI camera fovy (can be modified in the GUI)
  #### --probes_path
  path of the baked GaussProbe
  #### --numProbes
  number of probes (1/8/64). In theory, any positive integer is allowed, but the released code only supports these three fixed values. (can be modified in the GUI)
  #### --iters
  count of iterations (0-10). In theory, any non-negative integer is allowed, but the released code only supports these eleven fixed values. (can be modified in the GUI)
  #### --meshproxy_pitch
  the voxel size (pitch), which determines the resolution of the mesh voxelization.
  #### --probesW
  width of the RGBD panorama
  #### --probesH
  height of the RGBD panorama
  #### --begin_id
  only to prevent OOM (Out of Memory); when GPU memory is insufficient, the process can exit and resume baking from the specified ID.
  #### --scale_ratio
  bounding box scale ratio for the mesh
  #### --just_render
  if using this argument, it will be equivalent to running `renderer.py`.

</details>

### Renderer GUI Tutorial

The following GUI usage tutorial is provided based on the current release. It is recommended to watch this in conjunction with the video available on the [project homepage](https://letianhuang.github.io/transparentgs/).


![Alt text](assets/GUI_tutorial.png)

#### Move the camera

  Bear resemblance to [raytracing](https://github.com/ashawkey/raytracing/).

  1. **drag rotate:** move with the left mouse button.
  2. **drag translation:** move with the middle mouse button
  3. **move closer:** move with the wheel

#### Options and Debug

  1. **Options**: the main ways to control, aside from moving the camera.
  2. **Debug**: display the camera pose.

#### gbuffers in Options

  Common G-buffers in typical renderers (`depth`, `mask`, `normal`, `position`), with special attention to:

  1. **reflect**: the reflection component (mesh).
  ![Alt text](assets/GUI_tutorial_gbuffers_reflect.png)
  2. **refract**: the refraction component (mesh).
  ![Alt text](assets/GUI_tutorial_gbuffers_refract.png)
  3. **render**: the weighted sum of the reflection and refraction components using the Fresnel term (mesh).
  ![Alt text](assets/GUI_tutorial_gbuffers_render.png)
  4. **gs_render**: the rendering result obtained using only traditional Gaussian primitives (3DGS).
  ![Alt text](assets/GUI_tutorial_gbuffers_gs_render.png)
  5. **semantic**: the result of hybrid rendering with Gaussians and meshes (mesh + 3DGS). Pixels belonging to the mesh are replaced with a uniform color that represents the same semantic label (e.g., purple).
   ![Alt text](assets/GUI_tutorial_gbuffers_semantic.png)

#### camera

Select the camera model.

 1. **pinhole**: the regular camera model which 3DGS also supports.
 2. **fisheye**: It can support a field of view (FOV) of up to 180°.
  ![Alt text](assets/GUI_tutorial_camera_fisheye.png)
 3. **panorama**: It can support a field of view (FOV) of 360°.
 ![Alt text](assets/GUI_tutorial_camera_panorama.png)

#### bkg

Select the background of the mesh.

  1. **black**: black color as the background
  2. **white**: white color as the background
  3. **3DGS**: Hybrid rendering of 3DGS and transparent objects (`reflect`, `refract`, `render`, `normal`).

#### normal mode

Select whether to apply normal smoothing.

1. **raw**: no
![Alt text](assets/GUI_tutorial_normal_raw.png)
2. **smooth**: yes
![Alt text](assets/GUI_tutorial_normal_smooth.png)

#### num probes

As changing the number of probes involves I/O overhead, it is not recommended to modify it through the GUI. It is advisable to configure it beforehand using terminal arguments. Additionally, increasing the number of probes demands more GPU memory.

#### num iters

Modify the count of iterations of IterQuery (0-10). In theory, any non-negative integer is allowed, but the released code only supports these eleven fixed values. Setting it to zero clearly demonstrates the superiority of the IterQuery.

#### FoV (y)

Modifying the field of view (FOV), particularly for `fisheye` cameras, allows reaching up to a 180° viewing angle.

#### IOR

![Alt text](assets/GUI_tutorial_ior_gsscale.png)

This mainly affects gbuffers with `refract` or `render` properties. When the IOR is approximately 1, the result is almost identical to the background, demonstrating the high quality of IterQuery (especially with 64 probes).

#### GS scale

Note that only the scale of the `3dgs` primitives is modified, not the overall scene scaling. Therefore, reducing the scale allows us to observe the gaps between Gaussians.

#### spp

Control the sampling rate of mesh ray tracing.

#### Pick a color

Modify the color of the mesh.

## Dataset

We release several self-captured scenes for reconstruction purposes. Please download the transparent object dataset from [Google Drive](https://drive.google.com/drive/folders/1J8H6VCA1tOKnzLXXOEopzAAZY9OOlWO3?usp=sharing).

## Standalone demo : Segmentation

- [ ] To release.

## TODO List
- [x] Release the code.
- [x] Release the dataset of transparent objects that we captured ourselves.
- [ ] Release the code of `Standalone demo : segmentation`.
- [ ] Code optimization.

## Acknowledgements

This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [GaussianShader](https://github.com/Asparagus15/GaussianShader), [GlossyGS](https://letianhuang.github.io/glossygs/), [op43dgs](https://github.com/LetianHuang/op43dgs), [raytracing](https://github.com/ashawkey/raytracing), [nvdiffrast](https://github.com/NVlabs/nvdiffrast), [instant-ngp](https://github.com/NVlabs/instant-ngp), [SAM2](https://github.com/facebookresearch/sam2), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SAM](https://github.com/facebookresearch/segment-anything), [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), and so on. Please follow the licenses. We thank all the authors for their great work and repos. We sincerely thank our colleagues for their valuable contributions to this project.

