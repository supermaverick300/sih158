# Reference review and implemented optimizations

Reviewed September 7, 2026. The repositories were inspected as references, not run or copied into the application. No root license file was found in either checkout; our implementation is independently written. Downloaded checkouts and their large assets remain in ignored `output/references/`.

## What the reference applications actually do

- [singlepassdrone3d](https://github.com/ASHLIN-JOHN/singlepassdrone3d), commit `629622bb7a8151158af74bf1499adce2a5998050`: `v23d.py` uses MiDaS, a grid terrain and procedural objects. It takes the pixelwise median of sampled RGB/depth frames without warping those frames into a shared camera view. Its `_build_box`, `_build_tree` and `_build_turbine` functions generate shapes. `scripts/reconstruct_terrain.py` explicitly builds a terrain, sky backdrop and procedural turbines. This explains the stylized reference screenshot; it is not evidence that those object surfaces were recovered from video. The separate `src/` pipeline includes COLMAP and depth backprojection.
- [dronevideoto3d](https://github.com/ASHLIN-JOHN/dronevideoto3d), commit `9c19db33348f8188754f753e8723164d27c7aaf9`: the scene services/workers use video descriptions and available model metadata to arrange assets. `scene_generator_safe.py` includes specialized Taj Mahal generation and fallback paths. The Three.js viewer loads GLB models and image textures. `scene_optimizer.py` filters overlapping placed objects; this is scene layout cleanup rather than reconstruction of measured geometry.

We use their useful presentation concepts—complete surfaces, detailed textures and separate display geometry—while retaining our measured/estimated/demo distinctions. Unaligned frame medians and automatic procedural substitutes were not adopted. A procedural presentation scene could be a separate labeled feature, but it must not masquerade as recovered buildings.

## Implemented in Drone 3D Studio

1. **Embedded source textures:** MiDaS exports full and preview GLBs with UV coordinates and the real source frame embedded. Color detail no longer depends on triangle count. Colored PLY files remain available for compatibility.
2. **Complete lower-detail preview:** the textured preview uses at most a 48-pixel-long geometry grid. The full geometry remains at up to 256 pixels along its longer dimension. Every preview cell is connected; triangles are not randomly discarded. The CPU viewer applies per-triangle affine texture mapping and preserves texture orientation. Large perspective changes can still distort image reliefs.
3. **Depth reuse:** a local cache key includes model bytes, processed frame bytes, image dimensions and preprocessing version. Changed inputs invalidate it; corrupt entries are recomputed. No source video is modified and normal generation performs no downloads. Cache entries live in the project under `cache/midas/` and can increase project size.
4. **Run evidence:** relief reports include cache hit/key, embedded-texture description, depth-stage time, total time and preview face count. Model data and Qt rendering remain separate from persisted JSON.

## Measured on the supplied video

Input was the previously extracted frame at **26.28 s** of `D:\DOWNLOADS\videoplaybackreal.webm`. The full surface has **34,560 vertices / 68,340 faces**. The prior colored preview had **11,550 faces**; the textured preview has **2,256 faces**, with visibly clearer image detail.

Five offscreen Qt renders at 800×600 were measured per preview on this laptop, using the same current renderer and camera:

| Measurement | Previous vertex-color preview | Textured preview |
| --- | ---: | ---: |
| Median render time | 1.014 s | 0.172 s |
| Faces | 11,550 | 2,256 |

This is approximately **5.9× faster** for this scene, not a general frame-rate guarantee. First generation of the new textured output took **0.956 s**; a cache-hit generation took **0.451 s**. These backend timings exclude video decoding/analysis, app startup and final UI loading; the first-run sample is not a cold-machine benchmark. Raw local evidence is `output/relief-benchmark.json` and run reports under `output/MiDaS-Video-Relief/reconstruction/`.

Final checks: **53 tests passed in 20.28 s** and startup smoke test exited **0**. Added tests verify cache reuse/invalidation/corruption recovery, texture/UV preservation through GLB export/reload and actual Qt rendering of source colors in the correct corners.

These optimizations do **not** turn monocular relief into a metric, complete or multi-frame 3D reconstruction. The COLMAP pipeline remains the mode for geometry recovered across the flight. The source image texture adds visual detail, not new geometric measurements.
