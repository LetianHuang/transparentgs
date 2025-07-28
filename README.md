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
              <span class="author-block"><sup>3</sup>State Key Lab of CAD & CG, Zhejiang University</span><br>
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

**[2025.07.23]** <img class="emoji" title=":smile:" alt=":smile:" src="https://github.githubassets.com/images/icons/emoji/unicode/1f604.png" height="20" width="20"> Birthday of the repository.

## TL;DR

We propose TransparentGS, a fast inverse rendering pipeline for transparent objects based on 3D-GS. The main contributions are three-fold: efficient transparent Gaussian primitives for specular refraction, GaussProbe to encode ambient light and nearby contents, and the IterQuery algorithm to reduce parallax errors in our probe-based framework.

## Overview

The overview of our TransparentGS pipeline. Each 3D scene is firstly separated into transparent objects and opaque environment using SAM2 [Ravi et al. 2024] guided by GroundingDINO [Liu et al. 2024]. For transparent objects, we propose transparent Gaussian primitives, which explicitly encode both geometric and material properties within 3D Gaussians. And the properties are rasterized into maps for subsequent deferred shading. For the opaque environment, we recover it with the original 3D-GS, and bake it into GaussProbe surrounding the transparent object. The GaussProbe are then queried through our IterQuery algorithm to compute reflection and refraction.

![pipeline](https://letianhuang.github.io/transparentgs/static/images/exp/pipeline.png)

## TransparentGS Viewer (Renderer)

![TransparentGS Renderer](assets/TransparentGS_viewer_utility.png)

### Utility

- [x] Real-time rendering and navigation of scenes that integrate traditional [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and triangle meshes (Highly robust to complex occlusions).
- [x]  Secondary light effects such as specular reflections and refractions.
- [x] Rendering with non-pinhole camera models.
- [x] Material Editing (e.g., IOR and base color).


## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{transparentgs,
  author={Huang, Letian and Ye, Dongwei and Dan, Jialin and Tao, Chengzhi and Liu, Huiwen and Zhou, Kun and Ren, Bo and Li, Yuanqi and Guo, Yanwen and Guo, Jie},
  journal={ACM Transactions on Graphics}, 
  title={TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians}, 
  year={2025}
}
@article{transparentgs,
  title={TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians},
  author={Huang, Letian and Ye, Dongwei and Dan, Jialin and Tao, Chengzhi and Liu, Huiwen and Zhou, Kun and Ren, Bo and Li, Yuanqi and Guo, Yanwen and Guo, Jie},
  journal={arXiv preprint arXiv:2504.18768},
  year={2025}
}
```
