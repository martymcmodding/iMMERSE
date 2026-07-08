/*=============================================================================
                                                           
 d8b 888b     d888 888b     d888 8888888888 8888888b.   .d8888b.  8888888888 
 Y8P 8888b   d8888 8888b   d8888 888        888   Y88b d88P  Y88b 888        
     88888b.d88888 88888b.d88888 888        888    888 Y88b.      888        
 888 888Y88888P888 888Y88888P888 8888888    888   d88P  "Y888b.   8888888    
 888 888 Y888P 888 888 Y888P 888 888        8888888P"      "Y88b. 888        
 888 888  Y8P  888 888  Y8P  888 888        888 T88b         "888 888        
 888 888   "   888 888   "   888 888        888  T88b  Y88b  d88P 888        
 888 888       888 888       888 8888888888 888   T88b  "Y8888P"  8888888888                                                                 
                                                                            
    Copyright (c) Pascal Gilcher. All rights reserved.
    
    * Unauthorized copying of this file, via any medium is strictly prohibited
 	* Proprietary and confidential

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

===============================================================================

    Launchpad is a prepass effect that prepares various data to use 
	in later shaders.

    Author:         Pascal Gilcher

    More info:      https://martysmods.com
                    https://patreon.com/mcflypg
                    https://github.com/martymcmodding  	

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef LAUNCHPAD_DEBUG_OUTPUT
 #define LAUNCHPAD_DEBUG_OUTPUT 	  	0		//[0 or 1] 1: enables debug output of the motion vectors
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform int OPTICAL_FLOW_Q <
	ui_type = "combo";
    ui_label = "Flow Quality";
	ui_items = "Low\0Medium\0High\0";
	ui_tooltip = "Higher settings produce more accurate results, at a performance cost.";
	ui_category = "Motion Estimation / Optical Flow";
> = 0;

uniform int NORMALS_MODE <
	ui_type = "combo";
    ui_label = "Normal Map Mode";
	ui_items = "Basic\0Smoothed\0Smoothed + Textured\0";   
    ui_category = "Normal Mapping";	
	ui_tooltip =  "Normal maps describe the orientation of geometry in the scene.\n"
				  "ReShade cannot read the game's own normals, so they must be\n"
				  "re-derived from screen-space information.\n\n"
				  "Basic\nNormals from depth only. Fast, blocky.\n\n"
				  "Smoothed\nFiltered to hide blocky geometry. Slower.\n\n"
				  "Smoothed + Textured\nAdds surface relief, greatly improving lighting.";
> = 0;

uniform int TEXTURED_NORMALS_QUALITY_ENUM <
	ui_type = "combo";
    ui_label = "Quality";
	ui_items = "Low\0Medium\0High\0Very High\0Ultra\0";   
    ui_category = "Textured Normals";	
> = 2;

uniform float TEXTURED_NORMALS_RADIUS <
	ui_type = "drag";
	ui_label = "Sample Radius";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_category = "Textured Normals";	
> = 0.5;

uniform float TEXTURED_NORMALS_INTENSITY <
	ui_type = "drag";
	ui_label = "Intensity";
	ui_tooltip = "Higher values cause stronger surface bumpyness.";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_category = "Textured Normals";	
> = 0.5;

#if LAUNCHPAD_DEBUG_OUTPUT != 0
uniform int DEBUG_MODE < 
    ui_type = "combo";
	ui_items = "All\0Optical Flow\0Optical Flow Vectors\0Normals\0Depth\0";
	ui_label = "Debug Output";
	ui_category = "Debug";
> = 0;
#endif
/*
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF4 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF5 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF6 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF7 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF8 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform bool debug_key_down < source = "key"; keycode = 0x46; mode = ""; >;
uniform bool debug_key_down2 < source = "key"; keycode = 0x47; mode = ""; >;

uniform bool DISABLE_POOLING <  > = false;
uniform bool DISABLE_UPSCALING <  > = false;*/

/*=============================================================================
	Textures, Samplers, Globals, Structs
=============================================================================*/

//do NOT change anything here. "hurr durr I changed this and now it works"
//you ARE breaking things down the line, if the shader does not work without changes
//here, it's by design.

texture ColorInputTex : COLOR;
texture DepthInputTex : DEPTH;
sampler ColorInput 	{ Texture = ColorInputTex; };
sampler DepthInput  { Texture = DepthInputTex; };

#include ".\MartysMods\mmx_global.fxh"
#include ".\MartysMods\mmx_depth.fxh"
#include ".\MartysMods\mmx_math.fxh"
#include ".\MartysMods\mmx_camera.fxh"
#include ".\MartysMods\mmx_deferred.fxh"
#include ".\MartysMods\mmx_texture.fxh"
#include ".\MartysMods\mmx_hash.fxh"
#include ".\MartysMods\mmx_sfc.fxh"

uniform uint FRAMECOUNT < source = "framecount"; >;
uniform float FRAMETIME < source = "frametime"; >;

//Yes I know you like to optimize blue noise away in favor for some shitty PRNG function, don't.
texture BlueNoiseJitterTex     < source = "iMMERSE_bluenoise_opt.png"; > { Width = 256; Height = 256; Format = RGBA8; };
sampler	sBlueNoiseJitterTex   { Texture = BlueNoiseJitterTex; AddressU = WRAP; AddressV = WRAP; };

//miplevel 3 is copied to previous frame!
//in theory I should be computing the optical flow at the lower TAAU resolution. Maybe later.
texture LinearDepthCurr      { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F; MipLevels = 4; };
sampler sLinearDepthCurr     { Texture = LinearDepthCurr; }; 
texture LinearDepthPrevLo      { Width = BUFFER_WIDTH>>3;   Height = BUFFER_HEIGHT>>3;   Format = R16F; };
sampler sLinearDepthPrevLo     { Texture = LinearDepthPrevLo; };

texture MotionTexLA7       { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = RGBA16F; };
sampler sMotionTexLA7      { Texture = MotionTexLA7;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLA6       { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = RGBA16F; };
sampler sMotionTexLA6      { Texture = MotionTexLA6;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLA5       { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = RGBA16F; };
sampler sMotionTexLA5      { Texture = MotionTexLA5;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLA4       { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = RGBA16F; };
sampler sMotionTexLA4      { Texture = MotionTexLA4;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLA3       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLA3      { Texture = MotionTexLA3;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLA2       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLA2      { Texture = MotionTexLA2;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLA1       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLA1      { Texture = MotionTexLA1;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLA0       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLA0      { Texture = MotionTexLA0;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB7       { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = RGBA16F; };
sampler sMotionTexLB7      { Texture = MotionTexLB7;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB6       { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = RGBA16F; };
sampler sMotionTexLB6      { Texture = MotionTexLB6;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB5       { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = RGBA16F; };
sampler sMotionTexLB5      { Texture = MotionTexLB5;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB4       { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = RGBA16F; };
sampler sMotionTexLB4      { Texture = MotionTexLB4;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB3       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLB3      { Texture = MotionTexLB3;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB2       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLB2      { Texture = MotionTexLB2;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB1       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLB1      { Texture = MotionTexLB1;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexLB0       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F; };
sampler sMotionTexLB0      { Texture = MotionTexLB0;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

texture MotionTexUpscale    { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RGBA16F;};
sampler sMotionTexUpscale   { Texture = MotionTexUpscale;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexUpscale2   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RGBA16F;};
sampler sMotionTexUpscale2  { Texture = MotionTexUpscale2;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

texture FlowFeaturesCurrL0   { Width = BUFFER_WIDTH >> 0;   Height = BUFFER_HEIGHT >> 0;   Format = R16F;};
texture FlowFeaturesCurrL1   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = R16F;};
texture FlowFeaturesCurrL2   { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = R16F;};
texture FlowFeaturesCurrL3   { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = R16F;};
texture FlowFeaturesCurrL4   { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = R16F;};
texture FlowFeaturesCurrL5   { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = R16F;};
texture FlowFeaturesCurrL6   { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = R16F;};
texture FlowFeaturesCurrL7   { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = R16F;};
sampler sFlowFeaturesCurrL0  { Texture = FlowFeaturesCurrL0; AddressU = MIRROR; AddressV = MIRROR; }; 
sampler sFlowFeaturesCurrL1  { Texture = FlowFeaturesCurrL1; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL2  { Texture = FlowFeaturesCurrL2; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL3  { Texture = FlowFeaturesCurrL3; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL4  { Texture = FlowFeaturesCurrL4; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL5  { Texture = FlowFeaturesCurrL5; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL6  { Texture = FlowFeaturesCurrL6; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesCurrL7  { Texture = FlowFeaturesCurrL7; AddressU = MIRROR; AddressV = MIRROR; };
texture FlowFeaturesPrevL0   { Width = BUFFER_WIDTH >> 0;   Height = BUFFER_HEIGHT >> 0;   Format = R16F;};
texture FlowFeaturesPrevL1   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = R16F;};
texture FlowFeaturesPrevL2   { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = R16F;};
texture FlowFeaturesPrevL3   { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = R16F;};
texture FlowFeaturesPrevL4   { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = R16F;};
texture FlowFeaturesPrevL5   { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = R16F;};
texture FlowFeaturesPrevL6   { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = R16F;};
texture FlowFeaturesPrevL7   { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = R16F;};
sampler sFlowFeaturesPrevL0  { Texture = FlowFeaturesPrevL0; AddressU = MIRROR; AddressV = MIRROR; }; 
sampler sFlowFeaturesPrevL1  { Texture = FlowFeaturesPrevL1; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL2  { Texture = FlowFeaturesPrevL2; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL3  { Texture = FlowFeaturesPrevL3; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL4  { Texture = FlowFeaturesPrevL4; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL5  { Texture = FlowFeaturesPrevL5; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL6  { Texture = FlowFeaturesPrevL6; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFlowFeaturesPrevL7  { Texture = FlowFeaturesPrevL7; AddressU = MIRROR; AddressV = MIRROR; };

struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

struct CSIN 
{
    uint3 groupthreadid     : SV_GroupThreadID;         //XYZ idx of thread inside group
    uint3 groupid           : SV_GroupID;               //XYZ idx of group inside dispatch
    uint3 dispatchthreadid  : SV_DispatchThreadID;      //XYZ idx of thread inside dispatch
    uint threadid           : SV_GroupIndex;            //flattened idx of thread inside group
};

static float2 kernel[19] = 
{
	float2(0, 0),	
	float2(0.9853, 1.01448),
	float2(-1.37112, 0.34605),
	float2(0.38592, -1.36054),	
	float2(2.15615, -1.16233),
	float2(-0.07147, 2.44845),
	float2(-2.08468, -1.28612),
	float2(-0.684, -3.08742),
	float2(3.01578, 0.95134),
	float2(-2.33178, 2.13607),
	float2(-3.73529, -0.21826),
	float2(2.05667, -3.12572),
	float2(1.67862, 3.34398),
	float2(-1.39388,4.00713),
	float2(-2.77334, -3.2107),
	float2(4.16722, -0.79643),
	float2(3.8414, 2.6914),
	float2(-4.25152, 1.98105),
	float2(0.41012, -4.67245)
};

/*=============================================================================
	Functions - Common
=============================================================================*/

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
    return o;
}

VSOUT OpticalFlowVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv);
	if(!Deferred::IPC::is_requested(MARTYSMODS_IPC_FEATURE_OPTICALFLOW)) o.vpos.xy = -100000;
    return o;
}

VSOUT AlbedoVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv);

	bool albedo_requested = Deferred::IPC::is_requested(MARTYSMODS_IPC_FEATURE_ALBEDO);
	bool normal_requested_and_textured = Deferred::IPC::is_requested(MARTYSMODS_IPC_FEATURE_NORMALS) && NORMALS_MODE == 2;
	if(!(albedo_requested || normal_requested_and_textured))o.vpos.xy = -100000;
    return o;
}

VSOUT NormalsVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv);
	if(!Deferred::IPC::is_requested(MARTYSMODS_IPC_FEATURE_NORMALS)) o.vpos.xy = -100000;
    return o;
}

float3 get_jitter_blue(in int2 pos)
{
	return tex2Dfetch(sBlueNoiseJitterTex, pos % 256).xyz;
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	//return lerp(0.5, rgb, saturate(log(1 + dist * 1000.0 )));
	return lerp(0.5, rgb, saturate(log(1 + dist * 3000.0 / FRAMETIME )));//normalize by frametime such that we don't need to adjust visualization intensity all the time
}

//turbo colormap fit, turned into MADD form
float3 colormap(float t)
{	
	t = saturate(t); //inferno
	float3 res = float3(4.069046086, -4.193858954, 4.324996022);
	res = mad(res, t.xxx, float3(-8.490712758, +8.389314011, -3.608884658));
	res = mad(res, t.xxx, float3(+3.892783760, -4.821108251, +2.798380308));
	res = mad(res, t.xxx, float3(+0.278906882, +1.605395918, -5.893222355));
	res = mad(res, t.xxx, float3(+1.228188385, +0.015360518, +3.122510347));
	res = mad(res, t.xxx, float3(-0.027780558, +0.014065206, -0.019628385));
	return saturate(res);	
}

/*=============================================================================
	OF - Inputs
=============================================================================*/
/*
texture2D StateCounterTex	{ Format = R32F;  	};
sampler2D sStateCounterTex	{ Texture = StateCounterTex;  };

float4 FrameWriteVS(in uint id : SV_VertexID) : SV_Position {return float4(!debug_key_down, !debug_key_down, 0, 1);}
float  FrameWritePS(in float4 vpos : SV_Position) : SV_Target0 {return FRAMECOUNT;}
*/
void WriteCurrFeatureAndDepthPS(in VSOUT i, out float o0 : SV_Target0, out float o1 : SV_Target1)
{	
	o0 = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);
	o0 += 0.05;
	o1 = length(Camera::uv_to_proj(i.uv) / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);	
}

void WritePrevFeaturePS(in VSOUT i, out float o : SV_Target0)
{	
	o = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);
	o += 0.05;
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void WritePrevDepthMipPS(in VSOUT i, out float o : SV_Target0)
{
	o = tex2Dlod(sLinearDepthCurr, i.uv, 3).x; //reuse mip calculation for current frame depth	
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void downsample_features(sampler s0, sampler s1, float2 uv, out float4 f0, out float f1)
{
	float2 tx = rcp(tex2Dsize(s0));
	f0 = f1 = 0;
	f0 += tex2Dlod(s0, uv + float2(-1.90313, -1.90313) * tx, 0) * 0.083411; 
	f0 += tex2Dlod(s0, uv + float2(       0, -1.90313) * tx, 0) * 0.121987; 
	f0 += tex2Dlod(s0, uv + float2( 1.90313, -1.90313) * tx, 0) * 0.083411; 
	f0 += tex2Dlod(s0, uv + float2(-1.90313,        0) * tx, 0) * 0.121987; 
	f0 += tex2Dlod(s0, uv + float2(       0,        0) * tx, 0) * 0.178404; 
	f0 += tex2Dlod(s0, uv + float2( 1.90313,        0) * tx, 0) * 0.121987; 
	f0 += tex2Dlod(s0, uv + float2(-1.90313,  1.90313) * tx, 0) * 0.083411; 
	f0 += tex2Dlod(s0, uv + float2(       0,  1.90313) * tx, 0) * 0.121987; 
	f0 += tex2Dlod(s0, uv + float2( 1.90313,  1.90313) * tx, 0) * 0.083411;
	f1 += tex2Dlod(s1, uv + float2(-1.90313, -1.90313) * tx, 0).x * 0.083411; 
	f1 += tex2Dlod(s1, uv + float2(       0, -1.90313) * tx, 0).x * 0.121987; 
	f1 += tex2Dlod(s1, uv + float2( 1.90313, -1.90313) * tx, 0).x * 0.083411; 
	f1 += tex2Dlod(s1, uv + float2(-1.90313,        0) * tx, 0).x * 0.121987; 
	f1 += tex2Dlod(s1, uv + float2(       0,        0) * tx, 0).x * 0.178404; 
	f1 += tex2Dlod(s1, uv + float2( 1.90313,        0) * tx, 0).x * 0.121987; 
	f1 += tex2Dlod(s1, uv + float2(-1.90313,  1.90313) * tx, 0).x * 0.083411; 
	f1 += tex2Dlod(s1, uv + float2(       0,  1.90313) * tx, 0).x * 0.121987; 
	f1 += tex2Dlod(s1, uv + float2( 1.90313,  1.90313) * tx, 0).x * 0.083411;
}

void DownsampleFeaturesPS1(in VSOUT i, out float f0 : SV_Target0, out float f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL0, sFlowFeaturesPrevL0, i.uv, f0, f1);} 
void DownsampleFeaturesPS2(in VSOUT i, out float f0 : SV_Target0, out float f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL1, sFlowFeaturesPrevL1, i.uv, f0, f1);} 
void DownsampleFeaturesPS3(in VSOUT i, out float f0 : SV_Target0, out float f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL2, sFlowFeaturesPrevL2, i.uv, f0, f1);} 
void DownsampleFeaturesPS4(in VSOUT i, out float f0 : SV_Target0, out float f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL3, sFlowFeaturesPrevL3, i.uv, f0, f1);} 
void DownsampleFeaturesPS5(in VSOUT i, out float f0 : SV_Target0, out float f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL4, sFlowFeaturesPrevL4, i.uv, f0, f1);} 
void DownsampleFeaturesPS6(in VSOUT i, out float f0 : SV_Target0, out float f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL5, sFlowFeaturesPrevL5, i.uv, f0, f1);}
void DownsampleFeaturesPS7(in VSOUT i, out float f0 : SV_Target0, out float f1 : SV_Target1){downsample_features(sFlowFeaturesCurrL6, sFlowFeaturesPrevL6, i.uv, f0, f1);}

/*=============================================================================
	OF - OF
=============================================================================*/

float4 filter_flow(in VSOUT i, sampler s_flow, const int depth_mip = 3, const int radius = 1, const int dilation = 1)
{	
	float4 center_flow = tex2Dlod(s_flow, i.uv, 0);

	//if(DISABLE_POOLING) return center_flow;

	float2 txflow = rcp(tex2Dsize(s_flow));
	float depth = tex2Dlod(sLinearDepthCurr, i.uv, depth_mip).x; 	

	float3 feature_moments0 = 0;
	float3 feature_moments1 = 0;
	float2 guide_moments = 0;	  
	float3 min_data = 1e10;
	float3 max_data = -1e10;
	float wsum = 1e-8;

	float2 blur_scale = txflow * dilation;

	[loop]for(int y = -radius; y <= radius; y++)
	[loop]for(int x = -radius; x <= radius; x++)
	{		
		float2 tuv = i.uv + blur_scale * float2(x, y);
		float4 tap = tex2Dlod(s_flow, tuv, 0);
		if(Math::inside_screen(tuv)) 
		{
			float lw = log2(1.0 + max(0, center_flow.z / (tap.z + 1e-6) - 0.5)); //we want to explicitly get better samples from the neighbours
			float zw = exp(-abs(tap.w / (depth + 1e-6) - 1) * 64.0);
			float w = (lw * zw + 1e-7);

			float g = tap.z;
			feature_moments0 += tap.xyz * w;
			feature_moments1 += tap.xyz * g * w;
			guide_moments += float2(g, g * g) * w;
			wsum += w;

			min_data = min(min_data, tap.xyz);
			max_data = max(max_data, tap.xyz);	
		}	
	}	
	
	feature_moments0 /= wsum;
	feature_moments1 /= wsum;
	guide_moments /= wsum;

	float guide_hi = min_data.z;

	float a00 = 1.0;
	float a01 = guide_moments.x;    
	float a10 = guide_moments.x;
	float a11 = guide_moments.y;  

	a00 += exp2(-14);
	a11 += exp2(-14);
	float idet = rcp(a00 * a11 - a01 * a01); 
	float3 a = idet * (a11 * feature_moments0 - a01 * feature_moments1);
	float3 b = idet * (a00 * feature_moments1 - a10 * feature_moments0);
	float4 res;
	res.xyz = a + b * guide_hi;
	res.xy = clamp(res.xy, min_data.xy, max_data.xy);	
	res.z = guide_hi;
	return float4(res.xyz, tex2Dlod(sLinearDepthPrevLo, i.uv + res.xy, 0).x);//write prev frame depth for reprojection validation
}

//can't write to the final flow map when I read it here so
float4 filter_flow_final(in VSOUT i, sampler s_flow, const int depth_mip = 2, const int radius = 3, const int dilation = 1)
{
	//if(DISABLE_UPSCALING) return tex2Dlod(s_flow, i.uv, 0);
	float2 txflow = rcp(tex2Dsize(s_flow));
	float depth = tex2Dlod(sLinearDepthCurr, i.uv, depth_mip).x;

	float3 feature_moments0 = 0;
	float3 feature_moments1 = 0;
	float2 guide_moments = 0;	 
	float wsum = 0;
	float3 min_data = 1e10;
	float3 max_data = -1e10;

	float2 blur_scale = txflow * dilation;

	[loop]for(int y = -radius; y <= radius; y++)
	[loop]for(int x = -radius; x <= radius; x++)
	{
		float2 tuv = i.uv + blur_scale * float2(x, y);
		float4 tap = tex2Dlod(s_flow, tuv, 0);
		float ew = Math::inside_screen(tuv);
		float lw = exp2(-tap.z * 4.0); //regular relative weighting
		float zw = exp(-abs(tap.w / (depth + 1e-6) - 1) * 64.0);	
		float w = (lw * zw + 1e-7) * ew; 
		float g = tap.w;
		feature_moments0 += tap.xyz * w;
		feature_moments1 += tap.xyz * g * w;
		guide_moments += float2(g, g * g) * w;
		wsum += w;	
		min_data = min(min_data, tap.xyz);
		max_data = max(max_data, tap.xyz);
	}	

	feature_moments0 /= wsum;
	feature_moments1 /= wsum;
	guide_moments /= wsum;

	float guide_hi = depth;  

	float a00 = 1.0;
	float a01 = guide_moments.x;    
	float a10 = guide_moments.x;
	float a11 = guide_moments.y; 
	a00 += exp2(-18.0);
	a11 += exp2(-18.0);
	float idet = rcp(a00 * a11 - a01 * a01); 
	float3 a = idet * (a11 * feature_moments0 - a01 * feature_moments1);
	float3 b = idet * (a00 * feature_moments1 - a10 * feature_moments0);
	float4 res;
	res.xyz = a + b * guide_hi;
	res.xyz = clamp(res.xyz, min_data, max_data);	
	res.w = depth;
	return res;	
}

#define MATCHING_SAMPLES 16 //absolutely don't tamper with that! It has some internal logic you don't see.

float4 calc_flow_dynamic(VSOUT i,
					 sampler s_feature_curr, 
					 sampler s_feature_prev, 
					 sampler s_flow, 
					 const int level)
{
	float2 motion = 0;
	
	[branch]
	if(level < 7)//if we're not the first pass, do some neighbour pooling to get a better initial guess
	{
		motion = filter_flow(i, s_flow, 3, 1, 1).xy; //dilated, prev pass already filtered without
	}

	const float2 texsize = BUFFER_SCREEN_SIZE >> level; 
	const float2 texelsize = rcp(texsize);
	
	float randphi = get_jitter_blue(i.vpos.xy).x;
	float2 sc; sincos(randphi * TAU / 6.0, sc.x, sc.y);
	
	//wrap texelsize scale into the matrix directly. Should resolve with unroll anyhow but might compile faster
	float2x2 km = float2x2(sc.y * texelsize.x, -sc.x * texelsize.y, sc.x * texelsize.x,  sc.y * texelsize.y);
	float mean = 0;	
	float local_block[MATCHING_SAMPLES];

	[unroll]
	for(uint k = 0; k < MATCHING_SAMPLES; k++) 
	{
		float2 tuv = i.uv + mul(kernel[k], km);
		//actually need that, precision issue					
		float4 texels = tex2DgatherR(s_feature_curr, tuv);
		float2 t = frac(tuv * texsize - 0.5);
		float2 a = mad(texels.zy - texels.wx, t.xx, texels.wx);
		local_block[k] = mad(a.y - a.x, t.y, a.x);		
		mean += local_block[k];
	}

	mean /= MATCHING_SAMPLES;
	float MAD = 0;
	[unroll]
	for(uint k = 0; k < MATCHING_SAMPLES; k++)
		MAD += abs(mean - local_block[k]);	

	int num_steps = (4 + OPTICAL_FLOW_Q * 4) * (level + 1);
	num_steps = min(num_steps, 32);	
	num_steps = MAD < 1.0/255.0 ? 1 : num_steps;

	float2 best_motion = motion;
	float  best_loss = 1e10;

	[loop]
	for(int j = 0; j < num_steps; j++)
	{	
		float loss = 0;
		float2 grad = 0;

		[unroll]
		for(uint k = 0; k < MATCHING_SAMPLES; k++)
		{
			float2 tuv = i.uv + motion + mul(kernel[k], km);	
    		float4 texels = tex2DgatherR(s_feature_prev, tuv);
			float2 t = frac(tuv * texsize - 0.5);

			float3 d = texels.zyx - texels.wxw;
			float2 dfduv = d.xz + (d.y - d.x) * t.yx;	
			float f = d.z * t.y + (dfduv.x * t.x + texels.w);
		
			float tloss = f - local_block[k];	
			grad += tloss > 0.0 ? dfduv : -dfduv;
			loss += abs(tloss);
		}	
 
		[branch]
		if(loss < best_loss * 0.9999)
		{
			best_loss = loss;
			best_motion = motion;		
		}
		else 
		{	
			j += 2 - OPTICAL_FLOW_Q;
		}

		grad *= texsize;			
		float2 gradstep = grad / (1e-15 + dot(grad, grad)) * loss;
		gradstep *= saturate(0.5 * rsqrt(1e-8 + dot2(gradstep * texsize)));	
		motion -= gradstep;	
	}

	float depth_key = 0;

	[branch]
	if(level == 0) //upscaling should be bilateral on curr frame depth, more accurate
	{
		depth_key = tex2Dlod(sLinearDepthCurr, i.uv, 2).x; //2 -> upscale
	}
	else //vector pooling makes more sense to measure prev frame reprojection error, less flickery
	{
		depth_key = tex2Dlod(sLinearDepthPrevLo, i.uv + motion, 0).x;
	}
	
	best_loss = best_loss / (0.01 + MAD);		
	best_loss += saturate(1 - MAD * 255.0) * 0.5; 	//do not touch! good weight for the bilateral upscale filter.

	float4 curr_layer = float4(best_motion, best_loss, depth_key);
	return curr_layer;
}


void BlockMatchingPassNewPS7V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL7, sFlowFeaturesPrevL7, ColorInput, 7);}
void FilterFlowPS7(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexLA7, 3, 3, 3);}
void BlockMatchingPassNewPS6V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL6, sFlowFeaturesPrevL6, sMotionTexLB7, 6);}
void FilterFlowPS6(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexLA6, 3, 3, 3);}
void BlockMatchingPassNewPS5V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL5, sFlowFeaturesPrevL5, sMotionTexLB6, 5);}
void FilterFlowPS5(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexLA5, 3, 3, 3);}
void BlockMatchingPassNewPS4V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL4, sFlowFeaturesPrevL4, sMotionTexLB5, 4);}
void FilterFlowPS4(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexLA4, 3, 3, 3);}
void BlockMatchingPassNewPS3V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL3, sFlowFeaturesPrevL3, sMotionTexLB4, 3);}
void FilterFlowPS3(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexLA3, 3, 3, 3);}
void BlockMatchingPassNewPS2V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL2, sFlowFeaturesPrevL2, sMotionTexLB3, 2);}
void FilterFlowPS2(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexLA2, 3, 3, 3);}
void BlockMatchingPassNewPS1V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL1, sFlowFeaturesPrevL1, sMotionTexLB2, 1);}
void FilterFlowPS1(in VSOUT i, out float4 o : SV_Target0){o = filter_flow(i, sMotionTexLA1, 3, 3, 3);}
void BlockMatchingPassNewPS0V2(in VSOUT i, out float4 o : SV_Target0){o = calc_flow_dynamic(i, sFlowFeaturesCurrL0, sFlowFeaturesPrevL0, sMotionTexLB1, 0);}

void UpscaleFilter8to4PS(in VSOUT i, out float4 o : SV_Target0){o = filter_flow_final(i, sMotionTexLA0, 2, 3, 2);}
void UpscaleFilter4to2PS(in VSOUT i, out float4 o : SV_Target0){o = filter_flow_final(i, sMotionTexUpscale, 1, 2, 2);}
void UpscaleFilter2to1PS(in VSOUT i, out float4 o : SV_Target0){o = filter_flow_final(i, sMotionTexUpscale2, 0, 1, 2);}

/*=============================================================================
	Shader Entry Points - Normals
=============================================================================*/

void NormalsPS(in VSOUT i, out PSOUT2 o)
{
	const float2 dirs[9] = 
	{
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,-1),//TL
		BUFFER_PIXEL_SIZE_DLSS * float2(0,-1),//T
		BUFFER_PIXEL_SIZE_DLSS * float2(1,-1),//TR
		BUFFER_PIXEL_SIZE_DLSS * float2(1,0),//R
		BUFFER_PIXEL_SIZE_DLSS * float2(1,1),//BR
		BUFFER_PIXEL_SIZE_DLSS * float2(0,1),//B
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,1),//BL
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,0),//L
		BUFFER_PIXEL_SIZE_DLSS * float2(-1,-1)//TL first duplicated at end cuz it might be best pair	
	};

	//needs high precision depth unfortunately :(
	float z_center = Depth::get_linear_depth(i.uv);
	float3 center_pos = Camera::uv_to_proj(i.uv, Camera::depth_to_z(z_center));

	//z close/far
	float2 z_prev;
	z_prev.x = Depth::get_linear_depth(i.uv + dirs[0]);
	z_prev.y = Depth::get_linear_depth(i.uv + dirs[0] * 2);
	float3 dv_prev = Camera::uv_to_proj(i.uv + dirs[0], Camera::depth_to_z(z_prev.x)) - center_pos;

	float4 best_normal = float4(0,0,0,100000);
	float4 weighted_normal = 0;

	[unroll]
	for(int j = 1; j < 9; j++)
	{
		float2 z_curr;
		z_curr.x = Depth::get_linear_depth(i.uv + dirs[j]);
		z_curr.y = Depth::get_linear_depth(i.uv + dirs[j] * 2);

		float3 dv_curr = Camera::uv_to_proj(i.uv + dirs[j], Camera::depth_to_z(z_curr.x)) - center_pos;	
		float3 temp_normal = cross(dv_curr, dv_prev);

		float2 z_guessed = 2 * float2(z_prev.x, z_curr.x) - float2(z_prev.y, z_curr.y);
		float error = dot(1, abs(z_guessed - z_center));
		
		float w = rcp(dot(temp_normal, temp_normal));
		w *= rcp(error * error + exp2(-32.0));
		
		weighted_normal += float4(temp_normal, 1) * w;	
		best_normal = error < best_normal.w ? float4(temp_normal, error) : best_normal;

		z_prev = z_curr;
		dv_prev = dv_curr;
	}

	float3 normal = weighted_normal.w < 1.0 ? best_normal.xyz : weighted_normal.xyz;
	//normal = best_normal.xyz;
	normal *= rsqrt(dot(normal, normal) + 1e-8);
	//V2 geom normals to .zw
	float2 enc = Math::octahedral_enc(-normal);
	o.t0 = o.t1 = enc.xyxy;//fixes bugs in RTGI, normal.z positive gives smaller error :)	
}

//gbuffer halfres for fast filtering
texture SmoothNormalsTempTex0  { Width = BUFFER_WIDTH_DLSS;   Height = BUFFER_HEIGHT_DLSS/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex0 { Texture = SmoothNormalsTempTex0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; };
//gbuffer halfres for fast filtering
texture SmoothNormalsTempTex1  { Width = BUFFER_WIDTH_DLSS;   Height = BUFFER_HEIGHT_DLSS/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex1 { Texture = SmoothNormalsTempTex1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;  };

void SmoothNormalsMakeGbufPS(in VSOUT i, out float4 o : SV_Target0)
{
	o.xyz = Deferred::get_normals(i.uv);
	o.w = Camera::depth_to_z(Depth::get_linear_depth(i.uv));
}

void get_gbuffer(in sampler s, in float2 uv, out float3 p, out float3 n)
{
	float4 t = tex2Dlod(s, uv, 0);
	n = t.xyz;
	p = Camera::uv_to_proj(uv, t.w);
}

void get_gbuffer_hi(in float2 uv, out float3 p, out float3 n)
{
	n = Deferred::get_normals(uv);
	p = Camera::uv_to_proj(uv);
}

float sample_distribution(float x, int iteration)
{
	if(!iteration) return x * sqrt(x);
	return x;
	//return x * x;
	//return exp2(2 * x - 2);
}

float sample_pdf(float x, int iteration)
{
	if(!iteration) return 1.5 * sqrt(x);
	return 1;
	//return 2 * x;
	//return 2 * log(2.0) * exp2(2 * x - 2);
}

float2x3 to_tangent(float3 n)
{
    bool bestside = n.z < n.y;
    float3 n2 = bestside ? n.xzy : n;
    float3 k = (-n2.xxy * n2.xyy) * rcp(1.0 + n2.z) + float3(1, 0, 1);
    float3 u = float3(k.xy, -n2.x);
    float3 v = float3(k.yz, -n2.y);
    u = bestside ? u.xzy : u;
    v = bestside ? v.xzy : v;
    return float2x3(u, v);
}

float2 deproject_dir(float3 pos, float3 dir)
{
	float2 dir2d = ((dir.xy * pos.z - pos.xy * dir.z) * BUFFER_ASPECT_RATIO_DLSS);
	return dir2d * rsqrt(dot(dir2d, dir2d) + 1e-8);
}

//this projects neighbouring points onto a plane tangential to the surface
//and fits a local linear model to the way the neighbouring vectors lean away
//from the center normal. And then plainly evaluating the model at the center lmao
//this is different to a plain avg since that gives me the avg lean/normal at the centroid
//of the points, not at the actual coordinate I'm at. This makes the model handle 
//grazing edges way nicer than a stupid plain average
float4 smooth_normals_mkii(in VSOUT i, const int iteration, sampler sGbuffer)
{
	const int num_dirs = iteration ? 6 : 4;
	const int num_steps = iteration ? 3 : 6;	
	float radius_mult = iteration ? 0.2 : 1.0;	

	float2 angle_tolerance = float2(45.0, 30.0);
	radius_mult *= 0.2 * 0.2;

	float4 rotator = Math::get_rotator(TAU / num_dirs);
	float2 kernel_dir; sincos(TAU / num_dirs + TAU / 12.0, kernel_dir.x, kernel_dir.y); 
 
	float3 p, n;
	get_gbuffer_hi(i.uv, p, n);
	float2x3 kernel_matrix = to_tangent(n);
	float3 t_u = kernel_matrix[0];
	float3 t_v = kernel_matrix[1];

	float2 sigma_n = cos(radians(angle_tolerance));
	
	float wsum  = 0.001;  
	float3 mean_lean = 0;
	float3 u_x_lean = 0;
	float3 v_x_lean = 0;
	float2 uv_m1 = 0; //means of uv
	float3 uv_m2 = 0; //uu, uv, vv
	[loop]
	for(int dir = 0; dir < num_dirs; dir++)
	{
		float2 sample_dir = deproject_dir(p, mul(kernel_dir, kernel_matrix));

		[loop]
		for(int stp = 0; stp < num_steps; stp++)
		{
			float fi = float(stp + 1.0) / num_steps;
			float r = sample_distribution(fi, iteration);
			float ipdf = sample_pdf(fi, iteration);

			float2 sample_uv = i.uv + sample_dir * r * radius_mult;
			if(!Math::inside_screen(sample_uv)) break;

			float3 sp, sn;
			get_gbuffer(sGbuffer, sample_uv, sp, sn);

			float ndotn = dot(sn, n);
			float3 lean = sn - n * ndotn;

			float3 delta = sp - p;
			float u = dot(delta, t_u);
			float v = dot(delta, t_v);
			float h = dot(delta, n);		 

			float plane_distance = abs(h) + abs(dot(-delta, sn));

			float wn = linearstep(sigma_n.x, sigma_n.y, ndotn);
			float wz = exp2(-plane_distance*plane_distance * 10.0);
			float wd = exp2(-dot(delta, delta));
			float w = wn * wz * wd * ipdf;		

			mean_lean += lean * w;
			u_x_lean  += u * lean * w;
			v_x_lean  += v * lean * w;
			uv_m1     += float2(u, v) * w;
			uv_m2     += float3(u*u, u*v, v*v) * w;	
			wsum += w;		

			if(w < 0.01) break;
		}
		kernel_dir = Math::rotate_2D(kernel_dir, rotator);
	}

	uv_m1 /= wsum;
	uv_m2 /= wsum;
	u_x_lean /= wsum;
	v_x_lean /= wsum;
	mean_lean /= wsum;

	//make centered moment matrix and normalize by determinant here (less muls :))
	float3 cuv_m2 = uv_m2 - uv_m1.xxy * uv_m1.xyy;
	cuv_m2.xz += 1e-5; 
	float det = cuv_m2.x*cuv_m2.z - cuv_m2.y*cuv_m2.y;
	float idet = 1.0 / (abs(det) < 1e-12 ? 1e-12 : det);
	cuv_m2 *= idet;

	float3 c_u_x_lean = u_x_lean - uv_m1.x * mean_lean; //center cross moments
	float3 c_v_x_lean = v_x_lean - uv_m1.y * mean_lean;
	
	//solve for gradients and eval at center (uv = 0)
	float3 leangrad_u = ( cuv_m2.z*c_u_x_lean - cuv_m2.y*c_v_x_lean);
	float3 leangrad_v = (-cuv_m2.y*c_u_x_lean + cuv_m2.x*c_v_x_lean);
	float3 smooooothed = safenormalize(mean_lean - uv_m1.x * leangrad_u - uv_m1.y * leangrad_v + n);	
	//best *= sign(dot(best, n)); // keep hemisphere
	return float4(smooooothed, p.z);
}

VSOUT SmoothNormalsVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
	if(!NORMALS_MODE) o.vpos.xy = -100000; //forcing NaN here kills this in geometry stage, faster than discard()
	if(!Deferred::IPC::is_requested(MARTYSMODS_IPC_FEATURE_NORMALS)) o.vpos.xy = -100000;
    return o;
}

void SmoothNormalsPass0PS(in VSOUT i, out float4 o : SV_Target0)
{
	o = smooth_normals_mkii(i, 0, sSmoothNormalsTempTex0);	
}

//writes into geonormals
void SmoothNormalsPass1PS(in VSOUT i, out float2 o : SV_Target0)
{	
	float3 n = smooth_normals_mkii(i, 1, sSmoothNormalsTempTex1).xyz;
	o = Math::octahedral_enc(-n);	
}

//copies from geo to regular normals, and optionally applies textured normals
void CopyNormalsPS(in VSOUT i, out float2 o : SV_Target0)
{
	o = tex2D(Deferred::sGeoNormalsTexV4, i.uv).xy; //copy octahedrally encoded as-is

	[branch]
	if(NORMALS_MODE == 2)
	{
		float3 n = -Math::octahedral_dec(o);
		float3 center_p   = Camera::uv_to_proj(i.uv);
		float  center_luma = dot(Deferred::get_albedo(i.uv), 0.3333); 

		float3 e_y = (center_p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE_DLSS * float2(0, 2)));
		float3 e_x = (center_p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE_DLSS * float2(2, 0)));
		float3 t_x = normalize(cross(n, e_x));   // unit tangent dirs
		float3 t_y = normalize(cross(n, e_y));

		float radius_scale = (0.5 + RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.01 * saturate(TEXTURED_NORMALS_RADIUS)) / 50.0 * 0.15;			
		t_x = normalize(e_x - n * dot(n, e_x));  
		t_y = cross(n, t_x);            

		float3 m_parametric = 0; //ss, st, tt
		float2 m_height 	= 0; //sh, th

		int samples = 3 + 5 * TEXTURED_NORMALS_QUALITY_ENUM; 

		for(int j = 1; j <= samples; ++j)
		{
			float2 axis;
			sincos(2.39996 * j, axis.y, axis.x);
			axis *= radius_scale * j;				

			float3 virtual_p = center_p + t_x * axis.x + t_y * axis.y;
			float2 uv        = Camera::proj_to_uv(virtual_p);
			float4 albedo_z = tex2Dlod(Deferred::sAlbedoTex, uv, 0);
			float3 actual_p  = Camera::uv_to_proj(uv, albedo_z.w);

			//project onto surface
			float3 delta = actual_p - center_p;
			axis = float2(dot(delta, t_x), dot(delta, t_y));	
			
			float plane_dist = abs(dot(n, delta));
			float w = rcp(0.05 + plane_dist);
			w *= 1.0/(j*j);

			float h = dot(albedo_z.rgb, 0.3333) - center_luma;

			m_parametric += axis.xxy * axis.xyy * w;
			m_height 	 += axis * h * w;	
		}		

		float det = m_parametric.x*m_parametric.z - m_parametric.y*m_parametric.y;
		float2 grad = (m_parametric.zx * m_height.xy - m_parametric.yy * m_height.yx) / (det + 1e-6);
		grad *= 10 * saturate(TEXTURED_NORMALS_INTENSITY * TEXTURED_NORMALS_INTENSITY);
		float3 detail_n = normalize(n - grad.x * t_x - grad.y * t_y);
		n = detail_n;

		o = Math::octahedral_enc(-detail_n);				
	}
}

/*=============================================================================
	Fake albedo texture
=============================================================================*/

float3 cone_overlap(float3 c)
{
    float k = 0.5 * 0.33;
    float2 f = float2(1 - 2 * k, k);
    float3x3 m = float3x3(f.xyy, f.yxy, f.yyx);
    return mul(c, m);
}

float3 cone_overlap_inv(float3 c)
{
    float k = 0.5 * 0.33;
    float2 f = float2(k - 1, k) * rcp(3 * k - 1);
    float3x3 m = float3x3(f.xyy, f.yxy, f.yyx);
    return mul(c, m);
}

float3 unpack_hdr_rtgi(float3 color)
{
    color  = saturate(color);
    color = cone_overlap(color);
    color = color*0.283799*((2.52405+color)*color);    
    color = color * rcp(1.04 - saturate(color));    
    return color;
}

float3 pack_hdr_rtgi(float3 color)
{
    color =  1.04 * color * rcp(color + 1.0);   
    color  = saturate(color);
    color = 1.14374*(-0.126893*color+sqrt(color));
    color = cone_overlap_inv(color);
    return color;     
}

float3 sdr_to_hdr(float3 c)
{ 
    return unpack_hdr_rtgi(c);
}

float3 hdr_to_sdr(float3 c)
{   
    return pack_hdr_rtgi(c); 
}

float get_sdr_luma(float3 c)
{
    c = c*0.283799*((2.52405+c)*c);   
    float lum = dot(c, float3(0.2125, 0.7154, 0.0721));
    lum = 1.14374*(-0.126893*(lum)+sqrt(lum));
    return lum;
}

float2 downsample_kuwahara(const sampler s0, float2 uv, const bool horizontal)
{
    const float2 texelsize = rcp(tex2Dsize(s0, 0));  
    float2 axis = horizontal ? float2(texelsize.x, 0) : float2(0, texelsize.y);

    float4 mL = 0;
    float4 mR = 0;
    float2 wsum = 0;

    [unroll]
    for(int j = -11; j <= 11; j++)
    {
        float2 off = j * axis;
        float2 tuv = uv + off;
        float w = exp(-j*j/121.0 * 3.0) * Math::inside_screen(tuv);
            
        float2 t = tex2Dlod(s0, tuv, 0).xy;

        w *= j == 0 ? 0.5 : 1;
        mL += float4(t, t * t) * w * (j <= 0);
        mR += float4(t, t * t) * w * (j >= 0);
        wsum += w * float2(j <= 0, j >= 0);
    }

    mL /= wsum.x; 
    mR /= wsum.y;
    float vL = max(0, mL.w - mL.y * mL.y); //.y .w is the regular luma BS so we can use that as weight
    float vR = max(0, mR.w - mR.y * mR.y);
    float2 w = rcp(0.25 + sqrt(float2(vL, vR)));
    return (mL.xy * w.x + mR.xy * w.y) / (w.x + w.y);    
}

#define EQUALIZATION_STRENGTH 1.0

//this is really awkward but we cannot use any of the common preprocessor integer log2 macros
//as the preprocessor runs out of stack space with them. So we have to do it manually like this

#define RESOLUTION_DIV 2

#define WIDTH   (BUFFER_WIDTH / RESOLUTION_DIV)
#define HEIGHT  (BUFFER_HEIGHT / RESOLUTION_DIV)

#if HEIGHT < 128
    #define LOWEST_LEVEL  3
#elif HEIGHT < 256
    #define LOWEST_LEVEL  4
#elif HEIGHT < 512
    #define LOWEST_LEVEL  5
#elif HEIGHT < 1024
    #define LOWEST_LEVEL  6
#elif HEIGHT < 2048
    #define LOWEST_LEVEL  7
#elif HEIGHT < 4096
    #define LOWEST_LEVEL  8
#elif HEIGHT < 8192
    #define LOWEST_LEVEL  9
#elif HEIGHT < 16384
   #define LOWEST_LEVEL   10
#else 
    #error "Unsupported resolution"
#endif

texture AlbedoPyramidL0     { Width = WIDTH>>0; Height = HEIGHT>>0; Format = RG16F;};
sampler sAlbedoPyramidL0    { Texture = AlbedoPyramidL0;};
#if LOWEST_LEVEL >= 1
texture AlbedoPyramidL1Tmp  { Width = WIDTH>>1; Height = HEIGHT>>0; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL1Tmp { Texture = AlbedoPyramidL1Tmp;};
texture AlbedoPyramidL1     { Width = WIDTH>>1; Height = HEIGHT>>1; Format = RG16F;};
sampler sAlbedoPyramidL1    { Texture = AlbedoPyramidL1;};
void DownsamplePS0H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL0,    i.uv, true);}
void DownsamplePS0V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL1Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 2
texture AlbedoPyramidL2Tmp  { Width = WIDTH>>2; Height = HEIGHT>>1; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL2Tmp { Texture = AlbedoPyramidL2Tmp;};
texture AlbedoPyramidL2     { Width = WIDTH>>2; Height = HEIGHT>>2; Format = RG16F;};
sampler sAlbedoPyramidL2    { Texture = AlbedoPyramidL2;};
void DownsamplePS1H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL1,    i.uv, true);}
void DownsamplePS1V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL2Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 3
texture AlbedoPyramidL3Tmp  { Width = WIDTH>>3; Height = HEIGHT>>2; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL3Tmp { Texture = AlbedoPyramidL3Tmp;};
texture AlbedoPyramidL3     { Width = WIDTH>>3; Height = HEIGHT>>3; Format = RG16F;};
sampler sAlbedoPyramidL3    { Texture = AlbedoPyramidL3;};
void DownsamplePS2H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL2,    i.uv, true);}
void DownsamplePS2V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL3Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 4
texture AlbedoPyramidL4Tmp  { Width = WIDTH>>4; Height = HEIGHT>>3; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL4Tmp { Texture = AlbedoPyramidL4Tmp;};
texture AlbedoPyramidL4     { Width = WIDTH>>4; Height = HEIGHT>>4; Format = RG16F;};
sampler sAlbedoPyramidL4    { Texture = AlbedoPyramidL4;};
void DownsamplePS3H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL3,    i.uv, true);}
void DownsamplePS3V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL4Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 5
texture AlbedoPyramidL5Tmp  { Width = WIDTH>>5; Height = HEIGHT>>4; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL5Tmp { Texture = AlbedoPyramidL5Tmp;};
texture AlbedoPyramidL5     { Width = WIDTH>>5; Height = HEIGHT>>5; Format = RG16F;};
sampler sAlbedoPyramidL5    { Texture = AlbedoPyramidL5;};
void DownsamplePS4H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL4,    i.uv, true);}
void DownsamplePS4V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL5Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 6
texture AlbedoPyramidL6Tmp  { Width = WIDTH>>6; Height = HEIGHT>>5; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL6Tmp { Texture = AlbedoPyramidL6Tmp;};
texture AlbedoPyramidL6     { Width = WIDTH>>6; Height = HEIGHT>>6; Format = RG16F;};
sampler sAlbedoPyramidL6    { Texture = AlbedoPyramidL6;};
void DownsamplePS5H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL5,    i.uv, true);}
void DownsamplePS5V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL6Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 7
texture AlbedoPyramidL7Tmp  { Width = WIDTH>>7; Height = HEIGHT>>6; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL7Tmp { Texture = AlbedoPyramidL7Tmp;};
texture AlbedoPyramidL7     { Width = WIDTH>>7; Height = HEIGHT>>7; Format = RG16F;};
sampler sAlbedoPyramidL7    { Texture = AlbedoPyramidL7;};
void DownsamplePS6H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL6,    i.uv, true);}
void DownsamplePS6V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL7Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 8
texture AlbedoPyramidL8Tmp  { Width = WIDTH>>8; Height = HEIGHT>>7; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL8Tmp { Texture = AlbedoPyramidL8Tmp;};
texture AlbedoPyramidL8     { Width = WIDTH>>8; Height = HEIGHT>>8; Format = RG16F;};
sampler sAlbedoPyramidL8    { Texture = AlbedoPyramidL8;};
void DownsamplePS7H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL7,    i.uv, true);}
void DownsamplePS7V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL8Tmp, i.uv, false);}
#endif
#if LOWEST_LEVEL >= 9
texture AlbedoPyramidL9Tmp  { Width = WIDTH>>9; Height = HEIGHT>>8; Format = RG16F;}; //for horizontal blur
sampler sAlbedoPyramidL9Tmp { Texture = AlbedoPyramidL1Tmp;};
texture AlbedoPyramidL9     { Width = WIDTH>>9; Height = HEIGHT>>9; Format = RG16F;};
sampler sAlbedoPyramidL9    { Texture = AlbedoPyramidL9;};
void DownsamplePS8H(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL8,    i.uv, true);}
void DownsamplePS8V(in VSOUT i, out float2 o : SV_Target0){o = downsample_kuwahara(sAlbedoPyramidL9Tmp, i.uv, false);}
#endif

texture FusedAlbedoPyramid    { Width = WIDTH>>0; Height = HEIGHT>>0; Format = RG16F;};
sampler sFusedAlbedoPyramid    { Texture = FusedAlbedoPyramid;};

void InitAlbedoPyramidPS(in VSOUT i, out float2 o : SV_Target0)
{
    float3 hdr = sdr_to_hdr(tex2D(ColorInput, i.uv).rgb);
    float loglum = dot(0.3333, log2(max(1e-3, hdr)));
    o.y = loglum; //key to .y  
    o.x = -loglum;
}

float func(float a, float b, float levelnorm)
{
    float res = abs(a - b) / max3(a, b, 1);
    res *= lerp(0.02, 0.3, EQUALIZATION_STRENGTH);    
    return saturate(res / (1 + res));
}

void FusePS(in VSOUT i, out float2 o : SV_Target0)
{
    float2 G[LOWEST_LEVEL + 1];
    G[0] = tex2D(sAlbedoPyramidL0, i.uv).xy;
#if LOWEST_LEVEL >= 1
    G[1] = tex2D(sAlbedoPyramidL1, i.uv).xy;
#endif
#if LOWEST_LEVEL >= 2
    G[2] = tex2D(sAlbedoPyramidL2, i.uv).xy;
#endif
#if LOWEST_LEVEL >= 3
    G[3] = tex2D(sAlbedoPyramidL3, i.uv).xy;
#endif
#if LOWEST_LEVEL >= 4
    G[4] = Texture::sample2D_bspline(sAlbedoPyramidL4, i.uv, (BUFFER_SCREEN_SIZE / RESOLUTION_DIV) >> 4).xy;
#endif
#if LOWEST_LEVEL >= 5
    G[5] = Texture::sample2D_bspline(sAlbedoPyramidL5, i.uv, (BUFFER_SCREEN_SIZE / RESOLUTION_DIV) >> 5).xy;
#endif
#if LOWEST_LEVEL >= 6
    G[6] = Texture::sample2D_bspline(sAlbedoPyramidL6, i.uv, (BUFFER_SCREEN_SIZE / RESOLUTION_DIV) >> 6).xy;
#endif
#if LOWEST_LEVEL >= 7
    G[7] = Texture::sample2D_bspline(sAlbedoPyramidL7, i.uv, (BUFFER_SCREEN_SIZE / RESOLUTION_DIV) >> 7).xy;
#endif
#if LOWEST_LEVEL >= 8
    G[8] = Texture::sample2D_bspline(sAlbedoPyramidL8, i.uv, (BUFFER_SCREEN_SIZE / RESOLUTION_DIV) >> 8).xy;
#endif
#if LOWEST_LEVEL >= 9
    G[9] = Texture::sample2D_bspline(sAlbedoPyramidL9, i.uv, (BUFFER_SCREEN_SIZE / RESOLUTION_DIV) >> 9).xy;
#endif

    float2 bias = G[LOWEST_LEVEL]; //keep .y to blend

    [unroll]
    for(int j = LOWEST_LEVEL - 1; j >= 0; j--)
    {
        bias = lerp(bias, G[j], func(G[j].y, bias.y, float(j) / LOWEST_LEVEL));
    }

    o.x = bias.x;
    o.y = dot(0.3333, tex2D(ColorInput, i.uv).rgb);
}

void AlbedoMainPS(in VSOUT i, out float4 o : SV_Target0)
{
    float4 m = 0;
    float ws = 0.0;

	float2 texelsize = rcp(tex2Dsize(sFusedAlbedoPyramid));

    [unroll]for(int y = -1; y <= 1; y++) 
    [unroll]for(int x = -1; x <= 1; x++)    
    {
        float2 t = tex2D(sFusedAlbedoPyramid, i.uv + float2(x, y) * texelsize).xy;
        float w = exp(-(x * x + y * y));
        m += float4(t.y, t.y * t.y, t.y * t.x, t.x) * w;
        ws += w;
    }    

    m /= ws;    
    float a = (m.z - m.x * m.w) / (max(m.y - m.x * m.x, 0.0) + 0.00001);
    float b = m.w - a * m.x;

    float guide = dot(0.3333, tex2D(ColorInput, i.uv).rgb);
    float bias = a * guide + b;

	float target = 0.18;
    float3 target_hdr = sdr_to_hdr(target.xxx);
    float target_loglum = dot(0.3333, log2(max(1e-3, target_hdr)));  
    bias += target_loglum; 

    o.rgb = bias.x * 0.05 + 0.5;    
    
    o.rgb = tex2D(ColorInput, i.uv).rgb;
    o.rgb = sdr_to_hdr(o.rgb);
    
    o.rgb *= exp2(bias);  

    //Let L = lighting, A = albedo, C = final scene color, p = multiscatter probability
	//then  C = L * (A + p * A + p² * A ....)
	//this means the final color is a combination of single and multiscattering. I'm fudging things here with a constant light
	//but if we invert this MacLaurin series to get the actual albedo A, we get... a reinhard tonemap curve lmao	
	 
    float3 L = 1; //assumed lighting
	float3 C = o.rgb;
	float p = 0.5; //backscatter probability, lambert we assume 0.5
	//o.rgb = C / (L + C * p);
  
	{
       float3 reverse_multiscattered = C / (L + C * p);
       o.rgb = normalize(reverse_multiscattered + 1e-3) * length(o.rgb); //normalize to get the original color
	}

	o.w = Camera::depth_to_z(Depth::get_linear_depth(i.uv)); //if we're gonna use RGBA16F, might as well pack depth in alpha, makes the textured normals faster
}

/*=============================================================================
	Debug
=============================================================================*/

#if LAUNCHPAD_DEBUG_OUTPUT != 0
void DebugPS(in VSOUT i, out float3 o : SV_Target0)
{	
	o = 0;
	switch(DEBUG_MODE)
	{
		case 0: //all 
		{
			float2 tuv = i.uv * 2.0;
			int2 q = tuv < 1.0.xx ? int2(0,0) : int2(1,1);
			tuv = frac(tuv);
			int qq = q.x * 2 + q.y;
			if(qq == 0) o = Deferred::get_normals(tuv) * float3(0.5,0.5,-0.5) + 0.5;
			if(qq == 1) o = colormap(sqrt(Depth::get_linear_depth(tuv)));
			if(qq == 2) o = showmotion(Deferred::get_motion(tuv));	
			if(qq == 3) o = tex2Dlod(ColorInput, tuv, 0).rgb;	
			break;			
		}
		case 1: o = showmotion(Deferred::get_motion(i.uv)); break;
		case 2: o = tex2Dlod(ColorInput, i.uv, 0).rgb; break;		
		case 3: o = Deferred::get_normals(i.uv) * 0.5 + 0.5; 
				o.z = 1-o.z;
		break;
		case 4: o = colormap(sqrt(Depth::get_linear_depth(i.uv))); break;
	}	
}

#define FLOW_VECTOR_DENSITY_INV 16
#define NUM_VECTORS_X (BUFFER_WIDTH / FLOW_VECTOR_DENSITY_INV)
#define NUM_VECTORS_Y (BUFFER_HEIGHT / FLOW_VECTOR_DENSITY_INV / 0.866)

void FlowVectorDebugVS(in uint id : SV_VertexID, out float4 vpos : SV_Position, out float2 uv : TEXCOORD0, out float4 color : LINECOLOR)
{
	if(DEBUG_MODE != 2) 
	{
		vpos = color = -100000;
		return;
	}

	uint tri_id = id / 3;

	float2 gridpos = float2(tri_id % NUM_VECTORS_X, tri_id / NUM_VECTORS_X);
	gridpos = (gridpos + 0.5) / float2(NUM_VECTORS_X, NUM_VECTORS_Y);
	gridpos.x += 0.5 / NUM_VECTORS_X * ((tri_id / NUM_VECTORS_X) % 2);

	float2 mv = Deferred::get_motion(gridpos) * BUFFER_SCREEN_SIZE.xy;	
	float s = length(mv);
	float2 d = mv / (1e-8 + s);	

	float2x2 shape_mat = float2x2(d.x, -d.y, d.y, d.x);
	float2x2 scale_mat = float2x2(s, 0, 0, 4.0);

	shape_mat = mul(shape_mat, scale_mat);

	const float2 tri_offsets[3] = 
	{
		float2(1, 0),
		float2(-0.5, 0.866),
		float2(-0.5, -0.866)
	};
	
	uv = tri_offsets[id % 3];
	vpos.xy = gridpos + mul(shape_mat, uv) * BUFFER_PIXEL_SIZE.xy;
	vpos  = float4(vpos.xy * float2(2, -2) + float2(-1, 1), 0, 1);
	color = float4(showmotion(mv), 1);
}

float4 FlowVectorDebugPS(in float4 vpos : SV_Position, in float2 uv : TEXCOORD0, in float4 color : LINECOLOR) : SV_Target0
{
	float r = length(uv);
	color.w *= smoothstep(0.5, 0.5-fwidth(r), r);	
	return color;
}

#endif

/*=============================================================================
	Techniques
=============================================================================*/

technique MartysMods_Launchpad
<
    ui_label = "iMMERSE: Launchpad (enable and move to the top!)";
    ui_tooltip =        
        "                           MartysMods - Launchpad                             \n"
        "                   MartysMods Epic ReShade Effects (iMMERSE)                  \n"
        "______________________________________________________________________________\n"
        "\n"

        "Launchpad is a catch-all setup shader that prepares various data for the other\n"
        "effects. Enable this effect and move it to the top of the effect list.        \n"
        "\n"
        "\n"
        "Visit https://martysmods.com for more information.                            \n"
        "\n"       
        "______________________________________________________________________________";
>
{  
	//pass {PrimitiveTopology = POINTLIST;VertexCount = 1;VertexShader = FrameWriteVS;PixelShader  = FrameWritePS;RenderTarget = StateCounterTex;} 

	//OF
	pass {VertexShader = OpticalFlowVS;PixelShader = WriteCurrFeatureAndDepthPS;RenderTarget0 = FlowFeaturesCurrL0;RenderTarget1 = LinearDepthCurr; }
	pass {VertexShader = OpticalFlowVS;PixelShader = DownsampleFeaturesPS1;RenderTarget0 = FlowFeaturesCurrL1;RenderTarget1 = FlowFeaturesPrevL1;}
	pass {VertexShader = OpticalFlowVS;PixelShader = DownsampleFeaturesPS2;RenderTarget0 = FlowFeaturesCurrL2;RenderTarget1 = FlowFeaturesPrevL2;}
	pass {VertexShader = OpticalFlowVS;PixelShader = DownsampleFeaturesPS3;RenderTarget0 = FlowFeaturesCurrL3;RenderTarget1 = FlowFeaturesPrevL3;}
	pass {VertexShader = OpticalFlowVS;PixelShader = DownsampleFeaturesPS4;RenderTarget0 = FlowFeaturesCurrL4;RenderTarget1 = FlowFeaturesPrevL4;}
	pass {VertexShader = OpticalFlowVS;PixelShader = DownsampleFeaturesPS5;RenderTarget0 = FlowFeaturesCurrL5;RenderTarget1 = FlowFeaturesPrevL5;}
	pass {VertexShader = OpticalFlowVS;PixelShader = DownsampleFeaturesPS6;RenderTarget0 = FlowFeaturesCurrL6;RenderTarget1 = FlowFeaturesPrevL6;}
	pass {VertexShader = OpticalFlowVS;PixelShader = DownsampleFeaturesPS7;RenderTarget0 = FlowFeaturesCurrL7;RenderTarget1 = FlowFeaturesPrevL7;}	

	pass Flow7 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS7V2;	RenderTarget = MotionTexLA7;}
	pass {VertexShader = OpticalFlowVS;PixelShader = FilterFlowPS7;	RenderTarget = MotionTexLB7;}	//sMotionTexLA7 -> MotionTexLB7
	pass Flow6 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS6V2;	RenderTarget = MotionTexLA6;}
	pass {VertexShader = OpticalFlowVS;PixelShader = FilterFlowPS6;	RenderTarget = MotionTexLB6;}
	pass Flow5 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS5V2;	RenderTarget = MotionTexLA5;}
	pass {VertexShader = OpticalFlowVS;PixelShader = FilterFlowPS5;	RenderTarget = MotionTexLB5;}
	pass Flow4 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS4V2;	RenderTarget = MotionTexLA4;}
	pass {VertexShader = OpticalFlowVS;PixelShader = FilterFlowPS4;	RenderTarget = MotionTexLB4;}	
	pass Flow3 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS3V2;	RenderTarget = MotionTexLA3;}
	pass {VertexShader = OpticalFlowVS;PixelShader = FilterFlowPS3;	RenderTarget = MotionTexLB3;}	
	pass Flow2 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS2V2;	RenderTarget = MotionTexLA2;}
	pass {VertexShader = OpticalFlowVS;PixelShader = FilterFlowPS2;	RenderTarget = MotionTexLB2;}	
	pass Flow1 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS1V2;	RenderTarget = MotionTexLA1;}
	pass {VertexShader = OpticalFlowVS;PixelShader = FilterFlowPS1;	RenderTarget = MotionTexLB1;}	
	pass Flow0 {VertexShader = OpticalFlowVS;PixelShader = BlockMatchingPassNewPS0V2;	RenderTarget = MotionTexLA0;}

	pass {VertexShader = OpticalFlowVS;PixelShader = UpscaleFilter8to4PS;	RenderTarget = MotionTexUpscale;}
	pass {VertexShader = OpticalFlowVS;PixelShader = UpscaleFilter4to2PS;	RenderTarget = MotionTexUpscale2;}
	pass {VertexShader = OpticalFlowVS;PixelShader = UpscaleFilter2to1PS;	RenderTarget = Deferred::MotionVectorsTex;}

	pass {VertexShader = OpticalFlowVS;PixelShader = WritePrevFeaturePS;RenderTarget0 = FlowFeaturesPrevL0;}
	pass {VertexShader = OpticalFlowVS;PixelShader = WritePrevDepthMipPS;RenderTarget0 = LinearDepthPrevLo;}

	//Albedo
	pass AlbedoPyramidInit   {VertexShader = AlbedoVS;PixelShader = InitAlbedoPyramidPS;  RenderTarget0 = AlbedoPyramidL0; }     
#if LOWEST_LEVEL >= 1
    pass AlbedoDownsample0A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS0H;  RenderTarget0 = AlbedoPyramidL1Tmp; } 
    pass AlbedoDownsample0B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS0V;  RenderTarget0 = AlbedoPyramidL1; } 
#endif
#if LOWEST_LEVEL >= 2
    pass AlbedoDownsample1A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS1H;  RenderTarget0 = AlbedoPyramidL2Tmp; } 
    pass AlbedoDownsample1B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS1V;  RenderTarget0 = AlbedoPyramidL2; }
#endif
#if LOWEST_LEVEL >= 3 
    pass AlbedoDownsample2A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS2H;  RenderTarget0 = AlbedoPyramidL3Tmp; } 
    pass AlbedoDownsample2B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS2V;  RenderTarget0 = AlbedoPyramidL3; }
#endif
#if LOWEST_LEVEL >= 4 
    pass AlbedoDownsample3A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS3H;  RenderTarget0 = AlbedoPyramidL4Tmp; } 
    pass AlbedoDownsample3B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS3V;  RenderTarget0 = AlbedoPyramidL4; } 
#endif
#if LOWEST_LEVEL >= 5 
    pass AlbedoDownsample4A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS4H;  RenderTarget0 = AlbedoPyramidL5Tmp; } 
    pass AlbedoDownsample4B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS4V;  RenderTarget0 = AlbedoPyramidL5; }
#endif
#if LOWEST_LEVEL >= 6 
    pass AlbedoDownsample5A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS5H;  RenderTarget0 = AlbedoPyramidL6Tmp; } 
    pass AlbedoDownsample5B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS5V;  RenderTarget0 = AlbedoPyramidL6; }
#endif
#if LOWEST_LEVEL >= 7 
    pass AlbedoDownsample6A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS6H;  RenderTarget0 = AlbedoPyramidL7Tmp; } 
    pass AlbedoDownsample6B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS6V;  RenderTarget0 = AlbedoPyramidL7; }
#endif
#if LOWEST_LEVEL >= 8 
    pass AlbedoDownsample7A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS7H;  RenderTarget0 = AlbedoPyramidL8Tmp; } 
    pass AlbedoDownsample7B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS7V;  RenderTarget0 = AlbedoPyramidL8; }
#endif 
#if LOWEST_LEVEL >= 9 
    pass AlbedoDownsample8A  {VertexShader = AlbedoVS;PixelShader = DownsamplePS8H;  RenderTarget0 = AlbedoPyramidL9Tmp; } 
    pass AlbedoDownsample8B  {VertexShader = AlbedoVS;PixelShader = DownsamplePS8V;  RenderTarget0 = AlbedoPyramidL9; }
#endif
    pass FuseAlbedoPyramid   {VertexShader = AlbedoVS; PixelShader = FusePS; RenderTarget0 = FusedAlbedoPyramid; }
    pass UpscaleAlbedoPyramid{VertexShader = AlbedoVS; PixelShader = AlbedoMainPS; RenderTarget = Deferred::AlbedoTex;}

	pass Normals 		{VertexShader = NormalsVS;      PixelShader = NormalsPS; RenderTarget0 = Deferred::GeoNormalsTexV4; RenderTarget1 = Deferred::NormalsTexV4; }	
	pass SmoothNormals0 {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsMakeGbufPS;  RenderTarget = SmoothNormalsTempTex0;}
	pass SmoothNormals1 {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass0PS;  RenderTarget = SmoothNormalsTempTex1;}
	pass SmoothNormals2 {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass1PS;  RenderTarget = Deferred::GeoNormalsTexV4;}
	pass CopyNormals    {VertexShader = NormalsVS;	    PixelShader = CopyNormalsPS; RenderTarget = Deferred::NormalsTexV4; }	

	IPC_CLEAR() // I clear here such that any pass or shader following afterwards may request a feature

#if LAUNCHPAD_DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	
	//debug may or may not need all of them.
	IPC_REQUEST_FEATURE(MARTYSMODS_IPC_FEATURE_NORMALS | MARTYSMODS_IPC_FEATURE_ALBEDO | MARTYSMODS_IPC_FEATURE_OPTICALFLOW)

	pass {VertexShader = MainVS;PixelShader  = DebugPS;  }	
	pass 
	{
		PrimitiveTopology = TRIANGLELIST;
		VertexCount = NUM_VECTORS_X * NUM_VECTORS_Y * 3;
		VertexShader = FlowVectorDebugVS;
		PixelShader  = FlowVectorDebugPS;
		BlendEnable=true;
		BlendOp=ADD;
		SrcBlend=SRCALPHA;
		DestBlend=INVSRCALPHA;
	} 		
#endif

	
}