
# MARTY'S MODS EPIC RESHADE EFFECTS (iMMERSE)

Advanced post processing shaders for ReShade

![Title](https://www.martysmods.com/media/MXAO_titleimg.jpg)

# OVERVIEW

*Marty's Mods Epic ReShade Effects (iMMERSE)* is a shader collection for ReShade, written in ReShade's proprietary shader language, ReShade FX. It is the successor to the popular *[qUINT](https://github.com/martymcmodding/qUINT)* library. It aims to condense most of ReShade's use cases into a small set of shaders, to improve performance, ease of use and accelerate preset prototyping. Many extended features can be enabled via preprocessor definitions in each of the shaders, so make sure to check them out.

## PREREQUISITES

To use iMMERSE, install [ReShade](https://reshade.me) 5.X (preferably the latest). As some of the effects require depth access, make sure to have your depth buffer correctly configured if you want to use them.

## HOW TO INSTALL

Download the zip archive of this repository using the green button on the top right and selecting "Download ZIP". Extract the Shaders and Textures folders somewhere on your drive and instruct ReShade to load their content. You can do so on the Settings tab of the ReShade GUI. Alternatively, place their contents into existing resource folders already listed there. Press "Reload" at the bottom of the Home tab of the ReShade GUI and they will be loaded. Now just search for "iMMERSE" in the technique list and enable what you like. You do you :)

Make sure to at least enable iMMERSE LAUNCHPAD and move it to the very top of the shader list via drag and drop. LAUNCHPAD prepares several resources that other shaders require, such as normal vectors and optical flow.

# INCLUDED EFFECTS

These effects are currently included in iMMERSE:

## [iMMERSE MXAO](https://www.martysmods.com/mxao/)
![MXAO title](https://www.martysmods.com/media/MXAO.webp)

iMMERSE MXAO is the successor of the qUINT MXAO effect, delivering high quality SSAO for video games. It uses the state of the art Ground Truth Ambient Occlusion algorithm by [\[Jimenez et al., 2016\]](https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf) and as of recent, [Screen Space Indirect Lighting with Visibility Bitmask](https://www.researchgate.net/publication/365320847_Screen_space_indirect_lighting_with_visibility_bitmask) which is as close to ray traced reference as it gets - and improves upon them. MXAO contains a better horizon falloff term than baseline GTAO and unlike the visibility bitmasks accounts for the cosine term which makes it radiometrically correct. 

Lots of microoptimization, cache aware sampling and an extremely efficient filter make it faster than reference implementations such as XeGTAO. As a result, it should be one of the most advanced SSAO implementations that exist.

## iMMERSE Anti Aliasing
![AA title](https://www.martysmods.com/media/SMAA-1.webp)

iMMERSE Anti Aliasing is a modified SMAA with many optimizations for current-gen hardware. Apart from microoptimizations yielding a performance boost of about 15% over baseline, on compute enabled platforms it can be twice as fast. iMMERSE AA makes heavy use of performance tricks such as thread reordering to reduce divergence and maximize occupancy, emulated wave operations to prevent single threads from stalling and more.

It is designed to not alter the visual output compared to regular SMAA, i.e. these optimizations do not come at the cost of reduced visual quality.

## [iMMERSE Launchpad](https://www.martysmods.com/launchpad/)
![LP title](https://www.martysmods.com/media/Launchpad-2.webp)

iMMERSE Launchpad is a prepass for several of the iMMERSE and iMMERSE Pro effects. As many depth depending effects (such as RTGI) require normal vectors and optical flow vectors for temporal reprojection and it is detrimental to performance to regenerate this data for every shader, Launchpad is designed as a one-off solution for this task. Enable it and move it to the very top of the shader list, then never worry about it again. Its motion estimation algorithm is inspired by Jak0bW's groundbreaking [Dense ReShade Motion Estimation](https://github.com/JakobPCoder/ReshadeMotionEstimation).

## iMMERSE Sharpen
![Sh title](https://www.martysmods.com/media/Sharpen-1.webp)

iMMERSE Sharpen is a depth-aware sharpening filter that leverages both depth and color to increase local contrast in desired areas, while avoiding many common artifacts usually found in sharpen algorithms, such as haloing around objects.

# License
### Copyright (c) Pascal Gilcher. All rights reserved.











