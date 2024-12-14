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

uniform int OPTICAL_FLOW_RES <
	ui_type = "combo";
    ui_label = "Flow Resolution";
	ui_items = "Quarter Resolution\0Half Resolution\0Full Resolution\0";
	ui_tooltip = "Higher resolution vectors are more accurate but cost more performance.";
    ui_category = "Motion Estimation / Optical Flow";
> = 0;

uniform int OPTICAL_FLOW_Q <
	ui_type = "combo";
    ui_label = "Flow Quality";
	ui_items = "Low\0Medium\0High\0";
	ui_tooltip = "Higher settings produce more accurate results, at a performance cost.";
	ui_category = "Motion Estimation / Optical Flow";
> = 0;

uniform bool ENABLE_SMOOTH_NORMALS <	
	ui_label = "Enable Smooth Normals";
	ui_tooltip = "Filters the normal buffer to reduce low-poly look in MXAO and RTGI."
	"\n\n"
	"Lighting algorithms depend on normal vectors, which describe the orientation\n"
	"of the geometry in the scene. As ReShade does not access the game's own normals,\n"
	"they are generated from the depth buffer instead. However, this process is lossy\n"
	"and does not contain normal maps and smoothing groups.\n"
	"As a result, they represent the true (blocky) object shapes and lighting calculated\n"
	"using them can make the low-poly appearance of geometry apparent.\n";
	ui_category = "NORMAL MAPS";	
> = false;

uniform bool ENABLE_TEXTURED_NORMALS <	
	ui_label = "Enable Texture Normals";
	ui_tooltip = "Estimates surface relief based on color information, for more accurate geometry representation.\n"
	             "Requires smooth normals to be enabled!";	
	ui_category = "NORMAL MAPS";	
> = false;

uniform float TEXTURED_NORMALS_RADIUS <
	ui_type = "drag";
	ui_label = "Textured Normals Sample Radius";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_category = "NORMAL MAPS";	
> = 0.5;

uniform float TEXTURED_NORMALS_INTENSITY <
	ui_type = "drag";
	ui_label = "Textured Normals Intensity";
	ui_tooltip = "Higher values cause stronger surface bumpyness.";
	ui_min = 0.0;
	ui_max = 1.0;
	ui_category = "NORMAL MAPS";	
> = 0.5;

uniform int TEXTURED_NORMALS_QUALITY <
	ui_type = "slider";
	ui_min = 1; ui_max = 3;
    ui_label = "Textured Normals Quality";
    ui_tooltip = "Higher settings produce more accurate results, at a performance cost.";
    ui_category = "NORMAL MAPS";	
> = 2;

#if LAUNCHPAD_DEBUG_OUTPUT != 0
uniform int DEBUG_MODE < 
    ui_type = "combo";
	ui_items = "All\0Optical Flow\0Optical Flow Vectors\0Normals\0Depth\0";
	ui_label = "Debug Output";
> = 0;
#endif

uniform int UIHELP <
	ui_type = "radio";
	ui_label = " ";	
	ui_text ="\nDescription for preprocessor definitions:\n"
	"\n"
	"LAUNCHPAD_DEBUG_OUTPUT\n"
	"\n"
	"Various debug outputs\n"
	"0: off\n"
	"1: on\n";
	ui_category_closed = false;
>;

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

uniform bool debug_key_down < source = "key"; keycode = 0x46; mode = ""; >;

uniform bool USE_SIMPLE_MIP_PYRAMID <  > = false;
*/

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
#include ".\MartysMods\mmx_qmc.fxh"
#include ".\MartysMods\mmx_camera.fxh"
#include ".\MartysMods\mmx_deferred.fxh"
#include ".\MartysMods\mmx_texture.fxh"

//todo wrap behind compute macro
namespace Deferred
{
	storage stNormalsTexV3 { Texture = NormalsTexV3;};
}

uniform uint FRAMECOUNT < source = "framecount"; >;
uniform float FRAMETIME < source = "frametime"; >;

texture MotionTexNewA       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA32F; MipLevels = 5;};
sampler sMotionTexNewA      { Texture = MotionTexNewA;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexNewB       { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA32F; MipLevels = 5;};
sampler sMotionTexNewB      { Texture = MotionTexNewB;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

texture MotionTexUpscale    { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RGBA16F;};
sampler sMotionTexUpscale   { Texture = MotionTexUpscale;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexUpscale2   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RGBA16F;};
sampler sMotionTexUpscale2  { Texture = MotionTexUpscale2;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

//Yes I know you like to optimize blue noise away in favor for some shitty PRNG function, don't.
texture BlueNoiseJitterTex     < source = "iMMERSE_bluenoise.png"; > { Width = 32; Height = 32; Format = RGBA8; };
sampler	sBlueNoiseJitterTex   { Texture = BlueNoiseJitterTex; AddressU = WRAP; AddressV = WRAP; };

#define MotionTexIntermediateTex0 			Deferred::MotionVectorsTex
#define sMotionTexIntermediateTex0 			Deferred::sMotionVectorsTex

//curr in x, prev in y
texture FeaturePyramidLevel0   { Width = BUFFER_WIDTH;   	  Height = BUFFER_HEIGHT;        Format = RG8; };
storage stFeaturePyramidLevel0  { Texture = FeaturePyramidLevel0;};
texture FeaturePyramidLevel1   { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RG16F;};
texture FeaturePyramidLevel2   { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RG16F;};
texture FeaturePyramidLevel3   { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RG16F;};
texture FeaturePyramidLevel4   { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = RG16F;};
texture FeaturePyramidLevel5   { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = RG16F;};
texture FeaturePyramidLevel6   { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = RG16F;};
texture FeaturePyramidLevel7   { Width = BUFFER_WIDTH >> 7;   Height = BUFFER_HEIGHT >> 7;   Format = RG16F;};
sampler sFeaturePyramidLevel0  { Texture = FeaturePyramidLevel0; AddressU = MIRROR; AddressV = MIRROR; }; 
sampler sFeaturePyramidLevel1  { Texture = FeaturePyramidLevel1; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel2  { Texture = FeaturePyramidLevel2; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel3  { Texture = FeaturePyramidLevel3; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel4  { Texture = FeaturePyramidLevel4; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel5  { Texture = FeaturePyramidLevel5; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel6  { Texture = FeaturePyramidLevel6; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel7  { Texture = FeaturePyramidLevel7; AddressU = MIRROR; AddressV = MIRROR; };

texture DepthLowresPacked          { Width = BUFFER_WIDTH/3;   Height = BUFFER_HEIGHT/3;   Format = RG16F; };
sampler sDepthLowresPacked         { Texture = DepthLowresPacked; MipFilter=POINT; MagFilter=POINT; MinFilter=POINT;}; 

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


//                                            o                              
//                                                                      
//             o        o   o   o         o   o   o                     
//   o         o            o                 o                         
// o x o   o o x o o    o o x o o     o   o o x o o   o                             
//   o         o            o                 o                         
//             o        o   o   o         o   o   o                     
//                                                                      
//                                            o                         

static float2 block_kernel[17] = 
{
	float2(0,  0), float2( 0, -1), float2( 0,  1), float2(-1,  0),	
	float2(1,  0), float2( 0, -2), float2( 0,  2), float2(-2,  0),	
	float2(2,  0), float2(-2, -2), float2( 2,  2), float2(-2,  2),	
	float2(2, -2), 
	float2(0, -4), float2( 0,  4), float2(-4,  0),	
	float2(4,  0)
};

static float2 lstar_kernel[16] = 
{
	float2(0.106, 0.141),
	float2(0.436, 0.030),
	float2(0.892, 0.106),
	float2(0.636, 0.215),
	float2(0.350, 0.287),
	float2(0.224, 0.402),
	float2(0.788, 0.359),
	float2(0.526, 0.471),
	float2(0.039, 0.575),
	float2(0.959, 0.563),
	float2(0.302, 0.706),
	float2(0.709, 0.649),
	float2(0.596, 0.970),
	float2(0.143, 0.859),
	float2(0.420, 0.968),
	float2(0.852, 0.894)
};

static const float2 golden_kernel[17] = 
{
	float2(0, 0), 
	float2(-0.184342, 0.168873), 
	float2(0.0309096, -0.3522), 
	float2(0.263462, 0.343639), 
	float2(-0.492357, -0.0870908), 
	float2(0.471673, -0.30004), 
	float2(-0.158974, 0.591377), 
	float2(-0.304862, -0.586992), 
	float2(0.664201, 0.242564), 
	float2(-0.693259, 0.286168), 
	float2(0.335079, -0.716046), 
	float2(0.248153, 0.791151), 
	float2(-0.749296, -0.43423), 
	float2(0.880363, -0.193547), 
	float2(-0.537983, 0.765228), 
	float2(-0.124431, -0.960217), 
	float2(0.76465, 0.644446)
};

static float2 star_kernel[13] = 
{
	float2(0, 0),
	//inner ring
	float2(-1, -2),
	float2(1, -2),
	float2(2, 0),
	float2(1, 2),
	float2(-1, 2),
	float2(-2, 0),
	//outer ring
	float2(-3, -2),
	float2(0,-4),
	float2(3, -2),
	float2(3, 2),
	float2(0, 4),
	float2(-3, 2)
};


/*=============================================================================
	Functions
=============================================================================*/

float hash11(float p)
{
    p = frac(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return frac(p);
}

float get_prev_depth(float2 uv)
{
	return tex2Dlod(sDepthLowresPacked, uv, 0).y;
}

float get_curr_depth(float2 uv)
{
	return tex2Dlod(sDepthLowresPacked, uv, 0).x;
}

float2 downsample_feature(sampler s, float2 uv)
{
	float2 res = 0;	
	float2 texelsize = rcp(tex2Dsize(s));	
	float wsum = 0;
#if 0
	for(int x = 0; x < 6; x++)
	for(int y = 0; y < 6; y++)
	{
		float2 offs = float2(x, y); //0 to 5
		offs -= 2.5; // -2.5 to 2.5
		float g = exp(-dot(offs, offs) * 0.1);
		res += g * tex2D(s, uv + offs * texelsize).rg;
		wsum += g;
	}
#else
	[unroll]for(int x = -1; x <= 1; x++)
	[unroll]for(int y = -1; y <= 1; y++)
	{
		float2 offs = float2(x, y) * 2;

		float2 offs_tl = offs + float2(-0.5, -0.5);
		float2 offs_tr = offs + float2( 0.5, -0.5);
		float2 offs_bl = offs + float2(-0.5,  0.5);
		float2 offs_br = offs + float2( 0.5,  0.5);

		float4 g;
		g.x = dot(offs_tl, offs_tl);
		g.y = dot(offs_tr, offs_tr);
		g.z = dot(offs_bl, offs_bl);
		g.w = dot(offs_br, offs_br);
		g = exp(-g*0.1);
		float tg = dot(g, 1);
		offs = (offs_tl * g.x + offs_tr * g.y + offs_bl * g.z + offs_br * g.w) / tg;
	

		//float tg = exp(-dot(offs, offs) * 0.1);
		res += tg * tex2Dlod(s, uv + offs * texelsize, 0).rg;
		wsum += tg;
	}
#endif
	

	return res / wsum;	
}

void DownsampleFeaturePS1(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel0, i.uv);} 
void DownsampleFeaturePS2(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel1, i.uv);} 
void DownsampleFeaturePS3(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel2, i.uv);} 
void DownsampleFeaturePS4(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel3, i.uv);} 
void DownsampleFeaturePS5(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel4, i.uv);} 
void DownsampleFeaturePS6(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel5, i.uv);}
void DownsampleFeaturePS7(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel6, i.uv);}

float3 get_jitter_blue(in int2 pos)
{
	return tex2Dfetch(sBlueNoiseJitterTex, pos % 32).xyz;
}

float loss(float a, float b)
{
	float t = a - b;
	return abs(t); //SAD
}

float3 loss_grad(float3 a, float b)
{
	float3 t = a - b;
	return abs(t); //SAD
}

struct AdamOptimizer
{
	float2 m;
	float v;
	float beta1decayed, beta2decayed;
	float beta1, beta2, epsilon;
	float lr;
};

AdamOptimizer init_adam(int T)
{
	AdamOptimizer a;
	a.m = a.v = 0;
	a.beta1decayed = a.beta2decayed = 1;

	a.epsilon = 0.00000001;
	a.beta1 = 0.9;
	a.beta2 = 0.999;
	a.lr = 0.000625;
	return a;
}

float2 update_adam(inout AdamOptimizer a, float2 grad)
{
	float2 g = grad;
	a.m = lerp(g, a.m, a.beta1);
	a.v = lerp(dot(g, g), a.v, a.beta2);

	a.beta1decayed *= a.beta1;
	a.beta2decayed *= a.beta2;

	{
		a.beta1decayed = 0;
		a.beta2decayed = 0;
	}

	float2 mhat = a.m / (1 - a.beta1decayed);
	float vhat  = a.v / (1 - a.beta2decayed);

	mhat *= 0.2;
	//return a.lr * mhat / (sqrt(vhat) + a.epsilon);
	return a.lr * (mhat * rsqrt(max(vhat, a.epsilon)));
}

float4 gradient_block_matching_new(sampler s_feature, sampler s_flow, float2 uv, int level, const int blocksize)
{	
	float2 texelsize = rcp(tex2Dsize(s_feature));
	float2 search_scale = texelsize;

	float level_fi = float(level / 7.0); //0 to 1

	int num_steps = level < 2 ? 4 : 8;
	num_steps *= 1 + OPTICAL_FLOW_Q;

	float2 deltax = texelsize * float2(0.01, 0);
	float2 deltay = texelsize * float2(0, 0.01);

	//get local block data
	float local_block[13];

	[unroll]
	for(uint k = 0; k < blocksize; k++) //always fetch it completely
	{
		float2 tuv = uv + star_kernel[k] * search_scale;
		local_block[k] = tex2Dlod(s_feature, tuv, 0).x;
	}	

	float4 coarse_layer = 0;//tex2D(s_flow, uv);	

	//if we're not the first pass, do some neighbour pooling to get a better initial guess
	[branch]
	if(level < 7)
	{
		coarse_layer = tex2Dlod(s_flow, uv, 0);
		float best_sad = 0;

		[unroll]
		for(uint k = 0; k < blocksize; k++)
		{
			float2 tuv = uv + coarse_layer.xy + star_kernel[k] * search_scale;
			best_sad += loss(local_block[k], tex2Dlod(s_feature, tuv, 0).y);	
		}

		float2 motion_texelsize = rcp(tex2Dsize(s_flow));	
		motion_texelsize = max(motion_texelsize, texelsize);
		int2 sector_offs[4] = {int2(-1, -2), int2(-2, 0), int2(1, -1), int2(0, 1)};
		[unroll]
		for(int sec = 0; sec < 4; sec++)
		{
			float2 flows[4];
			flows[0] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(0, 0)), 0).xy;
			flows[1] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(1, 0)), 0).xy;
			flows[2] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(0, 1)), 0).xy;
			flows[3] = tex2Dlod(s_flow, uv + motion_texelsize * (sector_offs[sec] + float2(1, 1)), 0).xy;

			float3 median = float3(0, 0, 1e10);
			
			[unroll]for(int j = 0; j < 4; j++)
			{
				float diffsum = 0;
				diffsum += distance(flows[j], flows[0]);
				diffsum += distance(flows[j], flows[1]);
				diffsum += distance(flows[j], flows[2]);
				diffsum += distance(flows[j], flows[3]);

				median = diffsum < median.z ? float3(flows[j], diffsum) : median;	
			}

			median.z = 0; //now loss

			[loop]
			for(uint k = 0; k < blocksize; k++)
			{
				float2 tuv = uv + median.xy + star_kernel[k] * search_scale;
				median.z += loss(local_block[k], tex2Dlod(s_feature, tuv, 0).y);
				if(median.z > best_sad) break;
			}

			[branch]
			if(median.z < best_sad)
			{
				best_sad = median.z;
				coarse_layer.xy = median.xy;
			}			
		}
	}
	
	//once found, proceed
	float2 total_motion = coarse_layer.xy;
	
	float3 SAD = 0; //center, +dx, +dy
	float2 texturesize = tex2Dsize(s_feature);	

	//read local gradient
	[unroll]
	for(uint k = 0; k < blocksize; k++)
	{		
		float2 tuv = uv + star_kernel[k] * search_scale;
		float g = local_block[k];
		float f;
		f = tex2Dlod(s_feature, tuv + total_motion,          0).y;
		SAD.x += loss(f, g);
		f = tex2Dlod(s_feature, tuv + total_motion + deltax, 0).y;		
		SAD.y += loss(f, g);
		f = tex2Dlod(s_feature, tuv + total_motion + deltay, 0).y;
		SAD.z += loss(f, g);
    }

	float2 grad = (SAD.yz - SAD.x) / float2(deltax.x, deltay.y);
	AdamOptimizer adam = init_adam(num_steps);

	float2 local_motion = 0;
	float2 best_local_motion = 0;
	float  best_SAD = SAD.x;
	adam.lr *= 1.0 + level;	
	adam.lr *= 0.5;
	adam.lr /= 1.0 + OPTICAL_FLOW_Q;
	float2 local_motion_prev = local_motion;
	
	int fails = 0;
	int max_fails = 4 * (1 + OPTICAL_FLOW_Q);
	[loop]
	while(num_steps-- >= 0 && fails < max_fails)
	{		
		//nesterov momentum
		float2 curr_grad_step = update_adam(adam, grad);

		if(maxc(abs(curr_grad_step) * BUFFER_SCREEN_SIZE) < 0.1) 
			break;

		local_motion = local_motion_prev - curr_grad_step;
		local_motion_prev = local_motion;
	
		local_motion -= curr_grad_step;//look ahead using curr gradient
		SAD = 0;

		[unroll]
		for(uint k = 0; k < blocksize; k++)
		{
			float2 tuv = uv + total_motion + local_motion + star_kernel[k] * search_scale;
			float g = local_block[k];

			float f;	
			f = tex2Dlod(s_feature, tuv, 0).y;	
			SAD.x += loss(f, g);
			f = tex2Dlod(s_feature, tuv + deltax, 0).y;
			SAD.y += loss(f, g);
			f = tex2Dlod(s_feature, tuv + deltay, 0).y;
			SAD.z += loss(f, g);
		}		

		[flatten]
		if(SAD.x < best_SAD)
		{
			best_SAD = SAD.x;
			best_local_motion = local_motion;
			fails = 0;
		}
		else 
		{
			fails++;
		}
		
		grad = (SAD.yz - SAD.x) / float2(deltax.x, deltay.y);		
	}

	local_motion = best_local_motion;
	total_motion += local_motion;

	float prev_depth_at_motion = get_prev_depth(uv + total_motion);
	float4 curr_layer = float4(total_motion, prev_depth_at_motion, best_SAD);
	return curr_layer;
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(log(1 + dist * 1000.0  /* / FRAMETIME */)));//normalize by frametime such that we don't need to adjust visualization intensity all the time
}

//turbo colormap fit, turned into MADD form
float3 gradient(float t)
{	
	t = saturate(t);
	float3 res = float3(59.2864, 2.82957, 27.3482);
	res = mad(res, t.xxx, float3(-152.94239396, 4.2773, -89.9031));	
	res = mad(res, t.xxx, float3(132.13108234, -14.185, 110.36276771));
	res = mad(res, t.xxx, float3(-42.6603, 4.84297, -60.582));
	res = mad(res, t.xxx, float3(4.61539, 2.19419, 12.6419));
	res = mad(res, t.xxx, float3(0.135721, 0.0914026, 0.106673));
	return saturate(res);
}

/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
    return o;
}
/*
texture2D StateCounterTex	{ Format = R32F;  	};
sampler2D sStateCounterTex	{ Texture = StateCounterTex;  };

float4 FrameWriteVS(in uint id : SV_VertexID) : SV_Position {return float4(!debug_key_down, !debug_key_down, 0, 1);}
float  FrameWritePS(in float4 vpos : SV_Position) : SV_Target0 {return FRAMECOUNT;}
*/
void WriteDepthFeaturePS(in VSOUT i, out float2 o : SV_Target0)
{
	o = Depth::get_linear_depth(i.uv);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x + 1) discard;
}

void WriteFeaturePS(in VSOUT i, out float4 o : SV_Target0)
{	
	o = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x + 1) discard;
}

void WritePrevLowresDepthPS(in VSOUT i, out float2 o : SV_Target0)
{
	o = Depth::get_linear_depth(i.uv);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void WriteFeaturePS2(in VSOUT i, out float4 o : SV_Target0)
{	
	o = dot(0.3333, tex2Dfetch(ColorInput, int2(i.vpos.xy)).rgb);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void BlockMatchingPassPS8(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel7, sMotionTexNewB, i.uv, 7, 7);}
void BlockMatchingPassPS7(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel6, sMotionTexNewA, i.uv, 6, 7);}
void BlockMatchingPassPS6(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel5, sMotionTexNewB, i.uv, 5, 7);}
void BlockMatchingPassPS5(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel4, sMotionTexNewA, i.uv, 4, 7);}
void BlockMatchingPassPS4(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel3, sMotionTexNewB, i.uv, 3, 7);}
void BlockMatchingPassPS3(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel2, sMotionTexNewA, i.uv, 2, 7);}
void BlockMatchingPassPS2(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel1, sMotionTexNewB, i.uv, 1, 13);}
void BlockMatchingPassPS1(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching_new(sFeaturePyramidLevel0, sMotionTexNewA, i.uv, 0, 13);}

void UpscaleFlowPS0(in VSOUT i, out float4 o : SV_Target0)
{
	if(OPTICAL_FLOW_RES < 1) 
		discard;
	o = gradient_block_matching_new(sFeaturePyramidLevel0, sMotionTexNewB, i.uv, 0, 13);
}

void UpscaleFlowPS1(in VSOUT i, out float4 o : SV_Target0)
{	
	if(OPTICAL_FLOW_RES < 2) 
		discard;
	o = gradient_block_matching_new(sFeaturePyramidLevel0, sMotionTexUpscale, i.uv, 0, 13);
}

void CopyToFullres(in VSOUT i, out float4 o : SV_Target0)
{
	o = 0;
	if(OPTICAL_FLOW_RES == 0)      o = tex2Dlod(sMotionTexNewB, i.uv, 0);
	else if(OPTICAL_FLOW_RES == 1) o = tex2Dlod(sMotionTexUpscale, i.uv, 0);
	else if(OPTICAL_FLOW_RES == 2) o = tex2Dlod(sMotionTexUpscale2, i.uv, 0);
}

/*=============================================================================
	Shader Entry Points - Normals
=============================================================================*/

//we need 2 pixels padding on each side for the 5x5 spanning kernel
groupshared float z_tgsm[(32+4)*(32+4)];

void NormalsCS(in CSIN i)
{
	int id = i.threadid;
	if(i.threadid < 18*18)
	{
		int2 p = int2(i.threadid % 18, i.threadid / 18);
		p *= 2;
		int2 screenpos = i.groupid.xy * 32 + p - 2;
		float2 uv = saturate((screenpos + 0.75) * BUFFER_PIXEL_SIZE_DLSS);

		float2 corrected_uv = Depth::correct_uv(uv); //fixed for lookup 

#if RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN
    	corrected_uv.y -= BUFFER_PIXEL_SIZE.y * 0.5;    //shift upwards since gather looks down and right
    	float4 depth_texels = tex2DgatherR(DepthInput, corrected_uv).wzyx;  
#else
    	float4 depth_texels = tex2DgatherR(DepthInput, corrected_uv);
#endif

	 	depth_texels = Depth::linearize(depth_texels);
		//WZ
		//XY
		z_tgsm[p.x + p.y * (32+4)]           = depth_texels.w;
		z_tgsm[p.x + p.y * (32+4) + 1]       = depth_texels.z;
		z_tgsm[p.x + (p.y + 1) * (32+4)]     = depth_texels.x;
		z_tgsm[p.x + 1 + (p.y + 1) * (32+4)] = depth_texels.y;
	}

	barrier();
	if(any(i.dispatchthreadid.xy >= BUFFER_SCREEN_SIZE_DLSS))
	{
		return;
	}

	//XY = 2D offset
	const int2 dirs[9] = 
	{
		int2(-1,-1),//TL
		int2(0,-1),//T
		int2(1,-1),//TR
		int2(1,0),//R
		int2(1,1),//BR
		int2(0,1),//B
		int2(-1,1),//BL
		int2(-1,0),//L
		int2(-1,-1)//TL first duplicated at end cuz it might be best pair	
	};

	int2 tgsm_coord = i.groupthreadid.xy + 2;
	int tgsm_id = tgsm_coord.x + tgsm_coord.y * (32+4);
	float2 uv = (i.dispatchthreadid.xy + 0.5) * BUFFER_PIXEL_SIZE_DLSS;	

	float z_center = z_tgsm[tgsm_id];
	float3 center_pos = Camera::uv_to_proj(uv, Camera::depth_to_z(z_center));

	float2 z_prev;
	z_prev.x = z_tgsm[tgsm_id + dirs[0].x     + dirs[0].y     * (32+4)];
	z_prev.y = z_tgsm[tgsm_id + dirs[0].x * 2 + dirs[0].y * 2 * (32+4)];

	float4 best_normal = float4(0,0,0,100000);
	float4 weighted_normal = 0;

	[unroll]
	for(int j = 1; j < 9; j++)
	{
		float2 z_curr;
		z_curr.x = z_tgsm[tgsm_id + dirs[j].x     + dirs[j].y     * (32+4)];
		z_curr.y = z_tgsm[tgsm_id + dirs[j].x * 2 + dirs[j].y * 2 * (32+4)];

		float2 z_guessed = 2 * float2(z_prev.x, z_curr.x) - float2(z_prev.y, z_curr.y);
		float score = dot(1, abs(z_guessed - z_center));
	
		float3 dd_0 = Camera::uv_to_proj(uv + BUFFER_PIXEL_SIZE * dirs[j],     Camera::depth_to_z(z_curr.x)) - center_pos;
		float3 dd_1 = Camera::uv_to_proj(uv + BUFFER_PIXEL_SIZE * dirs[j - 1], Camera::depth_to_z(z_prev.x)) - center_pos;
		float3 temp_normal = cross(dd_0, dd_1);
		float w = rcp(dot(temp_normal, temp_normal));
		w *= rcp(score * score + exp2(-32.0));
		weighted_normal += float4(temp_normal, 1) * w;	

		best_normal = score < best_normal.w ? float4(temp_normal, score) : best_normal;
		z_prev = z_curr;
	}

	float3 normal = weighted_normal.w < 1.0 ? best_normal.xyz : weighted_normal.xyz;
	//normal = best_normal.xyz;
	normal *= rsqrt(dot(normal, normal) + 1e-8);
	//V2 geom normals to .zw	
	float2 packed_normal = Math::octahedral_enc(-normal); //fixes bugs in RTGI, normal.z positive gives smaller error :)	
	tex2Dstore(Deferred::stNormalsTexV3, i.dispatchthreadid.xy, packed_normal.xyxy);
}

void NormalsPS(in VSOUT i, out float4 o : SV_Target0)
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

	float z_center = Depth::get_linear_depth(i.uv);
	float3 center_pos = Camera::uv_to_proj(i.uv, Camera::depth_to_z(z_center));

	//z close/far
	float2 z_prev;
	z_prev.x = Depth::get_linear_depth(i.uv + dirs[0]);
	z_prev.y = Depth::get_linear_depth(i.uv + dirs[0] * 2);

	float4 best_normal = float4(0,0,0,100000);
	float4 weighted_normal = 0;

	[loop]
	for(int j = 1; j < 9; j++)
	{
		float2 z_curr;
		z_curr.x = Depth::get_linear_depth(i.uv + dirs[j]);
		z_curr.y = Depth::get_linear_depth(i.uv + dirs[j] * 2);

		float2 z_guessed = 2 * float2(z_prev.x, z_curr.x) - float2(z_prev.y, z_curr.y);
		float score = dot(1, abs(z_guessed - z_center));
	
		float3 dd_0 = Camera::uv_to_proj(i.uv + dirs[j],     Camera::depth_to_z(z_curr.x)) - center_pos;
		float3 dd_1 = Camera::uv_to_proj(i.uv + dirs[j - 1], Camera::depth_to_z(z_prev.x)) - center_pos;
		float3 temp_normal = cross(dd_0, dd_1);
		float w = rcp(dot(temp_normal, temp_normal));
		w *= rcp(score * score + exp2(-32.0));
		weighted_normal += float4(temp_normal, 1) * w;	

		best_normal = score < best_normal.w ? float4(temp_normal, score) : best_normal;
		z_prev = z_curr;
	}

	float3 normal = weighted_normal.w < 1.0 ? best_normal.xyz : weighted_normal.xyz;
	//normal = best_normal.xyz;
	normal *= rsqrt(dot(normal, normal) + 1e-8);
	//V2 geom normals to .zw	
	o = Math::octahedral_enc(-normal).xyxy;//fixes bugs in RTGI, normal.z positive gives smaller error :)	
}

//gbuffer halfres for fast filtering
texture SmoothNormalsTempTex0  { Width = BUFFER_WIDTH_DLSS/2;   Height = BUFFER_HEIGHT_DLSS/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex0 { Texture = SmoothNormalsTempTex0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; };
//gbuffer halfres for fast filtering
texture SmoothNormalsTempTex1  { Width = BUFFER_WIDTH_DLSS/2;   Height = BUFFER_HEIGHT_DLSS/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex1 { Texture = SmoothNormalsTempTex1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;  };
//high res copy back so we can fetch center tap at full res always
texture SmoothNormalsTempTex2  < pooled = true; > { Width = BUFFER_WIDTH_DLSS;   Height = BUFFER_HEIGHT_DLSS;   Format = RGBA16;  };
sampler sSmoothNormalsTempTex2 { Texture = SmoothNormalsTempTex2; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;  };

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

float4 smooth_normals_mkii(in VSOUT i, int iteration, sampler sGbuffer)
{
	int num_dirs = iteration ? 6 : 4;
	int num_steps = iteration ? 3 : 6;	
	float radius_mult = iteration ? 0.2 : 1.0;	

	float2 angle_tolerance = float2(45.0, 30.0); //min/max

	radius_mult *= 0.2 * 0.2;

	float4 rotator = Math::get_rotator(TAU / num_dirs);
	float2 kernel_dir; sincos(TAU / num_dirs + TAU / 12.0, kernel_dir.x, kernel_dir.y); 
	
	float3 p, n;
	get_gbuffer_hi(i.uv, p, n);
	float2x3 kernel_matrix = to_tangent(n);

	float4 bin_front = float4(n, 1) * 0.001;
	float4 bin_back = float4(n, 1) * 0.001;

	float2 sigma_n = cos(radians(angle_tolerance));

	[loop]
	for(int dir = 0; dir < num_dirs; dir++)
	{
		[loop]
		for(int stp = 0; stp < num_steps; stp++)
		{
			float fi = float(stp + 1.0) / num_steps;

			float r = sample_distribution(fi, iteration);
			float ipdf = sample_pdf(fi, iteration);

			float2 sample_dir = normalize(Camera::proj_to_uv(p + 0.1 * mul(kernel_dir, kernel_matrix)) - i.uv);
			//sample_dir = 0.8 * BUFFER_ASPECT_RATIO * kernel_dir;//
			//sample_dir = kernel_dir * 0.2;

			float2 sample_uv = i.uv + sample_dir * r * radius_mult;
			if(!Math::inside_screen(sample_uv)) break;

			float3 sp, sn;
			get_gbuffer(sGbuffer, sample_uv, sp, sn);

			float ndotn = dot(sn, n);
			float plane_distance = abs(dot(sp - p, n)) + abs(dot(p - sp, sn));

			float wn = smoothstep(sigma_n.x, sigma_n.y, ndotn);
			float wz = exp2(-plane_distance*plane_distance * 10.0);
			float wd = exp2(-dot(p - sp, p - sp));

			float w = wn * wz * wd;

			//focal point detection, find closest point to both 3D lines
			/*
			//find connecting axis
			float3 A = cross(n, sn);

			//find segment lengths for both line equations p + lambda * n
			float d2 = dot(p - sp, cross(n, A)) / dot(sn, cross(n, A));
			float d1 = dot(sp - p, cross(sn, A)) / dot(n, cross(sn, A));
			*/

			//heavily simplified math of the above using Lagrange identity and dot(n,n)==dot(sn,sn)==1
			float d2 = (ndotn * dot(p - sp,  n) - dot(p - sp, sn)) / (ndotn*ndotn - 1);
			float d1 = (ndotn * dot(p - sp, sn) - dot(p - sp,  n)) / (1 - ndotn*ndotn);

			//calculate points where each line is closest to the other line
			float3 hit1 = p + n * d1;
			float3 hit2 = sp + sn * d2;

			//mutual focal point is the mid point between those 2
			float3 middle = (hit1 + hit2) * 0.5;
			float side = dot(middle - p, n);

			//a hard sign split causes flickering, so do a smooth classifier as front or back
			float front_weight = saturate(side * 3.0 + 0.5);
			float back_weight = 1 - front_weight;

			if(ndotn > 0.9999) //fix edge case with parallel lines
			{
				front_weight = 1;
				back_weight = 1;
			}

			bin_front += float4(sn, 1) * ipdf * w * front_weight;
			bin_back += float4(sn, 1) * ipdf * w * back_weight;

			if(w < 0.01) break;
		}

		kernel_dir = Math::rotate_2D(kernel_dir, rotator);
	}

	bin_back.xyz = normalize(bin_back.xyz);
	bin_front.xyz = normalize(bin_front.xyz);

	//smooth binary select
	float bal = bin_back.w / (bin_front.w + bin_back.w);
	bal = smoothstep(0, 1, bal);
	bal = smoothstep(0, 1, bal);

	float3 best_bin = lerp(bin_front.xyz, bin_back.xyz, bal);
	return float4(safenormalize(best_bin), p.z);
}

VSOUT SmoothNormalsVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
	if(!ENABLE_SMOOTH_NORMALS) o.vpos = -100000; //forcing NaN here kills this in geometry stage, faster than discard()
    return o;
}

void SmoothNormalsPass0PS(in VSOUT i, out float4 o : SV_Target0)
{
	o = smooth_normals_mkii(i, 0, sSmoothNormalsTempTex0);	
}

void SmoothNormalsPass1PS(in VSOUT i, out float4 o : SV_Target0)
{	
	float3 n = -smooth_normals_mkii(i, 1, sSmoothNormalsTempTex1).xyz;
	float3 orig_n = n;

	[branch]
	if(ENABLE_TEXTURED_NORMALS)
	{
		float3 p = Camera::uv_to_proj(i.uv);
		float luma = dot(tex2D(ColorInput, i.uv).rgb, 0.3333);

		float3 e_y = (p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE_DLSS * float2(0, 2)));
		float3 e_x = (p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE_DLSS * float2(2, 0)));
		e_y = normalize(cross(n, e_y));
		e_x = normalize(cross(n, e_x));

		float radius_scale = (0.5 + RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.01 * saturate(TEXTURED_NORMALS_RADIUS)) / 50.0;

		float3 v_y = e_y * radius_scale;
		float3 v_x = e_x * radius_scale;

		float3 center_color = tex2D(ColorInput, i.uv).rgb;
		float center_luma = dot(center_color * center_color, float3(0.2126, 0.7152, 0.0722));

		float3 center_p_height = p + center_luma * n;
		float3 summed_normal = n * 0.01;

		int octaves = TEXTURED_NORMALS_QUALITY;	

		float total_luma = center_luma;

		[loop]
		for(int octave = 0; octave < octaves; octave++)
		{
			float3 height[4];
			float4 plane_dist;

			float2 axis; sincos(HALF_PI * octave / float(octaves), axis.y, axis.x); //modulate directions per octave to get better rotation invariance
			const float4 next_axis = Math::get_rotator(HALF_PI);

			float fi = exp2(octave);
			axis *= fi;

			[unroll]
			for(int a = 0; a < 4; a++)
			{
				float3 virtual_p = p + v_x * axis.x + v_y * axis.y;
				float2 uv = Camera::proj_to_uv(virtual_p);	
				float3 actual_p = Camera::uv_to_proj(uv);

				float3 tap_color = tex2Dlod(ColorInput, uv, 0).rgb;
				float tap_luma = dot(tap_color * tap_color, float3(0.2126, 0.7152, 0.0722));
				total_luma += tap_luma;
				
				height[a] = virtual_p + tap_luma * n;
				plane_dist[a] = abs(dot(n, actual_p - p));

				axis = Math::rotate_2D(axis, next_axis);
			}

			[unroll]
			for(int j = 0; j < 4; j++)
			{
				uint this_idx = j;
				uint next_idx = (j + 1) % 4;

				float w = rcp(0.05 + plane_dist[this_idx] + plane_dist[next_idx]);
				float3 curr_n = -cross(height[this_idx] - center_p_height, height[next_idx] - center_p_height);
				curr_n *= rsqrt(1e-5 + dot(curr_n, curr_n));
				w *= exp2(-octave);
				summed_normal += curr_n * w;
			}
		}

		summed_normal.xyz = safenormalize(summed_normal.xyz);
		float3 halfvec = n - summed_normal.xyz * 0.95;
		halfvec.xyz /= lerp(total_luma, 0.5,  0.5);
		n += halfvec * saturate(TEXTURED_NORMALS_INTENSITY * TEXTURED_NORMALS_INTENSITY * TEXTURED_NORMALS_INTENSITY) * 10.0;
		n = normalize(n);
	}	

	o.xy = Math::octahedral_enc(n);
	o.zw = Math::octahedral_enc(orig_n);
}

void CopyNormalsPS(in VSOUT i, out float4 o : SV_Target0)
{
	o = tex2D(sSmoothNormalsTempTex2, i.uv);
}

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
			if(qq == 0) o = Deferred::get_normals(tuv) * 0.5 + 0.5;
			if(qq == 1) o = gradient(Depth::get_linear_depth(tuv));
			if(qq == 2) o = showmotion(Deferred::get_motion(tuv));	
			if(qq == 3) o = tex2Dlod(ColorInput, tuv, 0).rgb;	
			break;			
		}
		case 1: o = showmotion(Deferred::get_motion(i.uv)); break;
		case 2:
		{
			float2 tile_size = 16.0;
			float2 tile_uv = i.uv * BUFFER_SCREEN_SIZE / tile_size;
			float2 motion = Deferred::get_motion((floor(tile_uv) + 0.5) * tile_size * BUFFER_PIXEL_SIZE);

			float3 chroma = showmotion(motion);
			
			motion *= BUFFER_SCREEN_SIZE;
			float velocity = length(motion);
			float2 mainaxis = velocity == 0 ? 0 : motion / velocity;
			float2 otheraxis = float2(mainaxis.y, -mainaxis.x);
			float2x2 rotation = float2x2(mainaxis, otheraxis);

			tile_uv = (frac(tile_uv) - 0.5) * tile_size;
			tile_uv = mul(tile_uv, rotation);
			o = tex2Dlod(ColorInput, i.uv, 0).rgb;
			float mask = smoothstep(min(velocity, 2.5), min(velocity, 2.5) - 1, abs(tile_uv.y)) * smoothstep(velocity, velocity - 1.0, abs(tile_uv.x));

			o = lerp(o, chroma, mask);
			break;
		}
		case 3: o = Deferred::get_normals(i.uv) * 0.5 + 0.5; break;
		case 4: o = gradient(Depth::get_linear_depth(i.uv)); break;
	}	
}
#endif



float3 srgb_to_AgX(float3 srgb)
{
    float3x3 toagx = float3x3(0.842479, 0.0784336, 0.0792237, 
                              0.042328, 0.8784686, 0.0791661, 
                              0.042376, 0.0784336, 0.8791430);
    return mul(toagx, srgb);         
}

float3 AgX_to_srgb(float3 AgX)
{   
    float3x3 fromagx = float3x3(1.19688,  -0.0980209, -0.0990297,
                               -0.0528969, 1.1519,    -0.0989612,
                               -0.0529716, -0.0980435, 1.15107);
    return mul(fromagx, AgX);            
}

uniform bool ASSUME_SRGB_INPUT <
    ui_label = "Assume sRGB input";
    ui_tooltip = "Converts color to linear before converting to HDR.\nDepending on the game color format, this can improve light behavior and blending.";
    ui_category = "Experimental";
> = true;

float3 unpack_hdr_rtgi(float3 color)
{
    color  = saturate(color);   
    if(ASSUME_SRGB_INPUT) color = color*0.283799*((2.52405+color)*color);  
    color = srgb_to_AgX(color);
    color = color * rcp(1.04 - saturate(color));    
    return color;
}

float3 pack_hdr_rtgi(float3 color)
{
    color =  1.04 * color * rcp(color + 1.0);   
    color = AgX_to_srgb(color);    
    color  = saturate(color);
    if(ASSUME_SRGB_INPUT) color = 1.14374*(-0.126893*color+sqrt(color));
    return color;     
}

float3 cone_overlap(float3 c)
{
    float k = 0.4 * 0.33;
    float2 f = float2(1 - 2 * k, k);
    float3x3 m = float3x3(f.xyy, f.yxy, f.yyx);
    return mul(c, m);
}

float3 cone_overlap_inv(float3 c)
{
    float k = 0.4 * 0.33;
    float2 f = float2(k - 1, k) * rcp(3 * k - 1);
    float3x3 m = float3x3(f.xyy, f.yxy, f.yyx);
    return mul(c, m);
}

#define degamma(_v) ((_v)*0.283799*((2.52405+(_v))*(_v)))
#define regamma(_v) (1.14374*(-0.126893*(_v)+sqrt(_v)))

#define WHITEPOINT 12.0 //don't change, it has a miniscule impact on the image, but low values will cause whites to be dimmed

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
    c = degamma(c);
    float lum = dot(c, float3(0.2125, 0.7154, 0.0721));
    lum = regamma(lum);
    return lum;
}

#define INTENSITY 1.0
#define ALBEDO_EXPOSURE_TARGET 		0.3
#define LAPLACIAN_RESOLUTION_DIV  	4

#define LAPLACIAN_TILE_WIDTH     (BUFFER_WIDTH / LAPLACIAN_RESOLUTION_DIV)
#define LAPLACIAN_TILE_HEIGHT    (BUFFER_HEIGHT / LAPLACIAN_RESOLUTION_DIV)

//this is really awkward but we cannot use any of the common preprocessor integer log2 macros
//as the preprocessor runs out of stack space with them. So we have to do it manually like this
#if LAPLACIAN_TILE_HEIGHT < 128
    #define LOWEST_MIP  6
#elif LAPLACIAN_TILE_HEIGHT < 256
    #define LOWEST_MIP  7
#elif LAPLACIAN_TILE_HEIGHT < 512
    #define LOWEST_MIP  8
#elif LAPLACIAN_TILE_HEIGHT < 1024
    #define LOWEST_MIP  9
#elif LAPLACIAN_TILE_HEIGHT < 2048
    #define LOWEST_MIP  10
#else 
    #error "Unsupported resolution"
#endif

#define TARGET_MIP        ((LOWEST_MIP) - 3)
#define TARGET_MIP_SCALE  (1 << (TARGET_MIP))

#define ATLAS_TILES_X   2
#define ATLAS_TILES_Y   2

//rounded up tile resolution such that it can be cleanly divided by 2 TARGET_MIP'th times
#define ATLAS_TILE_RESOLUTION_X  CEIL_DIV(LAPLACIAN_TILE_WIDTH, TARGET_MIP_SCALE) * TARGET_MIP_SCALE
#define ATLAS_TILE_RESOLUTION_Y  CEIL_DIV(LAPLACIAN_TILE_HEIGHT, TARGET_MIP_SCALE) * TARGET_MIP_SCALE

#define ATLAS_RESOLUTION_X ((ATLAS_TILE_RESOLUTION_X) * (ATLAS_TILES_X))
#define ATLAS_RESOLUTION_Y ((ATLAS_TILE_RESOLUTION_Y) * (ATLAS_TILES_Y))

texture LaunchpadExposureAtlasL0  { Width = (ATLAS_RESOLUTION_X)>>0; Height = (ATLAS_RESOLUTION_Y)>>0; Format = RGBA16F;};
texture LaunchpadWeightAtlasL0    { Width = (ATLAS_RESOLUTION_X)>>0; Height = (ATLAS_RESOLUTION_Y)>>0; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL0 { Texture = LaunchpadExposureAtlasL0;};
sampler sLaunchpadWeightAtlasL0   { Texture = LaunchpadWeightAtlasL0;};
#if TARGET_MIP >= 1
texture LaunchpadExposureAtlasL1  { Width = (ATLAS_RESOLUTION_X)>>1; Height = (ATLAS_RESOLUTION_Y)>>1; Format = RGBA16F;};
texture LaunchpadWeightAtlasL1    { Width = (ATLAS_RESOLUTION_X)>>1; Height = (ATLAS_RESOLUTION_Y)>>1; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL1 { Texture = LaunchpadExposureAtlasL1;};
sampler sLaunchpadWeightAtlasL1   { Texture = LaunchpadWeightAtlasL1;};
#endif
#if TARGET_MIP >= 2
texture LaunchpadExposureAtlasL2  { Width = (ATLAS_RESOLUTION_X)>>2; Height = (ATLAS_RESOLUTION_Y)>>2; Format = RGBA16F;};
texture LaunchpadWeightAtlasL2    { Width = (ATLAS_RESOLUTION_X)>>2; Height = (ATLAS_RESOLUTION_Y)>>2; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL2 { Texture = LaunchpadExposureAtlasL2;};
sampler sLaunchpadWeightAtlasL2   { Texture = LaunchpadWeightAtlasL2;};
#endif
#if TARGET_MIP >= 3
texture LaunchpadExposureAtlasL3  { Width = (ATLAS_RESOLUTION_X)>>3; Height = (ATLAS_RESOLUTION_Y)>>3; Format = RGBA16F;};
texture LaunchpadWeightAtlasL3    { Width = (ATLAS_RESOLUTION_X)>>3; Height = (ATLAS_RESOLUTION_Y)>>3; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL3 { Texture = LaunchpadExposureAtlasL3;};
sampler sLaunchpadWeightAtlasL3   { Texture = LaunchpadWeightAtlasL3;};
#endif
#if TARGET_MIP >= 4
texture LaunchpadExposureAtlasL4  { Width = (ATLAS_RESOLUTION_X)>>4; Height = (ATLAS_RESOLUTION_Y)>>4; Format = RGBA16F;};
texture LaunchpadWeightAtlasL4    { Width = (ATLAS_RESOLUTION_X)>>4; Height = (ATLAS_RESOLUTION_Y)>>4; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL4 { Texture = LaunchpadExposureAtlasL4;};
sampler sLaunchpadWeightAtlasL4   { Texture = LaunchpadWeightAtlasL4;};
#endif
#if TARGET_MIP >= 5
texture LaunchpadExposureAtlasL5  { Width = (ATLAS_RESOLUTION_X)>>5; Height = (ATLAS_RESOLUTION_Y)>>5; Format = RGBA16F;};
texture LaunchpadWeightAtlasL5    { Width = (ATLAS_RESOLUTION_X)>>5; Height = (ATLAS_RESOLUTION_Y)>>5; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL5 { Texture = LaunchpadExposureAtlasL5;};
sampler sLaunchpadWeightAtlasL5   { Texture = LaunchpadWeightAtlasL5;};
#endif
#if TARGET_MIP >= 6
texture LaunchpadExposureAtlasL6  { Width = (ATLAS_RESOLUTION_X)>>6; Height = (ATLAS_RESOLUTION_Y)>>6; Format = RGBA16F;};
texture LaunchpadWeightAtlasL6    { Width = (ATLAS_RESOLUTION_X)>>6; Height = (ATLAS_RESOLUTION_Y)>>6; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL6 { Texture = LaunchpadExposureAtlasL6;};
sampler sLaunchpadWeightAtlasL6   { Texture = LaunchpadWeightAtlasL6;};
#endif
#if TARGET_MIP >= 7
texture LaunchpadExposureAtlasL7  { Width = (ATLAS_RESOLUTION_X)>>7; Height = (ATLAS_RESOLUTION_Y)>>7; Format = RGBA16F;};
texture LaunchpadWeightAtlasL7    { Width = (ATLAS_RESOLUTION_X)>>7; Height = (ATLAS_RESOLUTION_Y)>>7; Format = RGBA16F;};
sampler sLaunchpadExposureAtlasL7 { Texture = LaunchpadExposureAtlasL7;};
sampler sLaunchpadWeightAtlasL7   { Texture = LaunchpadWeightAtlasL7;};
#endif

texture LaunchpadCollapsedExposurePyramidTex { Width = ATLAS_TILE_RESOLUTION_X; Height = ATLAS_TILE_RESOLUTION_Y; Format = RG16F;};
sampler sLaunchpadCollapsedExposurePyramidTex { Texture = LaunchpadCollapsedExposurePyramidTex;};

void InitPyramidAtlasPS(in VSOUT i, out PSOUT2 o)
{
    //figure out 1D tile ID
    int2 tile_id = floor(i.uv * float2(ATLAS_TILES_X, ATLAS_TILES_Y));
    int tile_id_1d = tile_id.y * ATLAS_TILES_X + tile_id.x;

    //now, figure out remapping values per each tile
    //x4 -> channels
    int4 curr_channel_idx = tile_id_1d * 4 + int4(0, 1, 2, 3); //0 to 23
    float num_channels = ATLAS_TILES_X * ATLAS_TILES_Y * 4;
    float exposure_spread = 1.0;    

    float4 exposure_bias = (curr_channel_idx - (num_channels - 1) * 0.5) * exposure_spread;//centered, i.e. -7.5 ... 7.5

    float2 uv = frac(i.uv * float2(ATLAS_TILES_X, ATLAS_TILES_Y));
    float3 c = sdr_to_hdr(tex2D(ColorInput, uv).rgb);

	c = 0;
	for(int x = 0; x < 2; x++)
	for(int y = 0; y < 2; y++)
	{
		float2 p = floor(i.vpos.xy) * 4 + float2(x, y) * 2;
		p += 1.0;
		p *= BUFFER_PIXEL_SIZE;
		c += tex2D(ColorInput, uv).rgb;
	}

	c /= 4.0;
	c = sdr_to_hdr(c);

    float3 exposed[4];
    exposed[0] = hdr_to_sdr(c * exp2(exposure_bias.x));
    exposed[1] = hdr_to_sdr(c * exp2(exposure_bias.y));
    exposed[2] = hdr_to_sdr(c * exp2(exposure_bias.z)); 
    exposed[3] = hdr_to_sdr(c * exp2(exposure_bias.w));  

    float4 luminances;
    luminances.x = get_sdr_luma(exposed[0]);
    luminances.y = get_sdr_luma(exposed[1]);
    luminances.z = get_sdr_luma(exposed[2]);
    luminances.w = get_sdr_luma(exposed[3]); 

    o.t0 = exposure_bias;
    o.t1 = exp(-(luminances-0.5)*(luminances-0.5)*32.0); 
}

//so apparently, no matter what filter I use, I can just go in log2 steps
//and it's fine. As the filter footprint doubles each pass, it will always make the same
//of a structure twice a given size and one pass more.
void tile_downsample(sampler s0, sampler s1, float2 uv, out float4 res0, out float4 res1)
{
    float2 num_tiles = float2(ATLAS_TILES_X, ATLAS_TILES_Y);

    float4 boundaries;
    boundaries.xy = floor(uv * num_tiles) / num_tiles;
    boundaries.zw = boundaries.xy + rcp(num_tiles);    

    float2 texelsize = rcp(tex2Dsize(s0, 0));

    const float sigma = 4.0;
    const int samples = ceil(2 * sigma);
    const float g = 0.5 * rcp(sigma * sigma);

    res0 = res1 = 0;
    float weightsum = 0;

    [loop]for(int x = -samples; x < samples; x++)
    [loop]for(int y = -samples; y < samples; y++)
    {        
        float2 offset = float2(x + 0.5, y + 0.5);//halving lands us in the middle of 2x2 texels so sample texel centers accurately
        float weight = exp(-dot(offset, offset) * g);
        float2 tap_uv = uv + offset * texelsize;

        weight = any(tap_uv <= boundaries.xy) || any(tap_uv >= boundaries.zw) ? 0 : weight;

        res0 += tex2Dlod(s0, tap_uv, 0) * weight;
        weightsum += weight;
    }

    [loop]for(int x = -samples; x < samples; x++)
    [loop]for(int y = -samples; y < samples; y++)
    {        
        float2 offset = float2(x + 0.5, y + 0.5);//halving lands us in the middle of 2x2 texels so sample texel centers accurately
        float weight = exp(-dot(offset, offset) * g);
        float2 tap_uv = uv + offset * texelsize; 

        weight = any(tap_uv <= boundaries.xy) || any(tap_uv >= boundaries.zw) ? 0 : weight;
        res1 += tex2Dlod(s1, tap_uv, 0) * weight;
    }

    res0 /= weightsum;
    res1 /= weightsum;
}


//identical, except with half the sigma but twice the sampling stride.
//using this for the higher resolutions as cross-tile bleed becomes apparent at lower resolutions (larger texels)
//and the higher resolutions are more performance critical
void tile_downsample_fast(sampler s0, sampler s1, float2 uv, out float4 res0, out float4 res1)
{
    float2 num_tiles = float2(ATLAS_TILES_X, ATLAS_TILES_Y);

    float4 boundaries;
    boundaries.xy = floor(uv * num_tiles) / num_tiles;
    boundaries.zw = boundaries.xy + rcp(num_tiles);    

    float2 texelsize = rcp(tex2Dsize(s0, 0));

    const float sigma = 2.0;
    const int samples = ceil(2 * sigma);
    const float g = 0.5 * rcp(sigma * sigma);

    res0 = res1 = 0;
    float weightsum = 0;

    [loop]for(int x = -samples; x < samples; x++)
    [loop]for(int y = -samples; y < samples; y++)
    {        
        float2 offset = float2(x + 0.5, y + 0.5);//halving lands us in the middle of 2x2 texels so sample texel centers accurately
        float weight = exp(-dot(offset, offset) * g);
        float2 tap_uv = uv + offset * texelsize * 2; 

        weight = any(tap_uv <= boundaries.xy) || any(tap_uv >= boundaries.zw) ? 0 : weight;

        res0 += tex2Dlod(s0, tap_uv, 0) * weight;
        weightsum += weight;
    }

    [loop]for(int x = -samples; x < samples; x++)
    [loop]for(int y = -samples; y < samples; y++)
    {        
        float2 offset = float2(x + 0.5, y + 0.5);//halving lands us in the middle of 2x2 texels so sample texel centers accurately
        float weight = exp(-dot(offset, offset) * g);
        float2 tap_uv = uv + offset * texelsize * 2; 

        weight = any(tap_uv <= boundaries.xy) || any(tap_uv >= boundaries.zw) ? 0 : weight;
        res1 += tex2Dlod(s1, tap_uv, 0) * weight;
    }

    res0 /= weightsum;
    res1 /= weightsum;
}


#if TARGET_MIP >= 1
void DownsamplePyramidsPS0(in VSOUT i, out PSOUT2 o){tile_downsample_fast(sLaunchpadExposureAtlasL0, sLaunchpadWeightAtlasL0, i.uv, o.t0, o.t1);}
#endif
#if TARGET_MIP >= 2
void DownsamplePyramidsPS1(in VSOUT i, out PSOUT2 o){tile_downsample_fast(sLaunchpadExposureAtlasL1, sLaunchpadWeightAtlasL1, i.uv, o.t0, o.t1);}
#endif
#if TARGET_MIP >= 3
void DownsamplePyramidsPS2(in VSOUT i, out PSOUT2 o){tile_downsample_fast(sLaunchpadExposureAtlasL2, sLaunchpadWeightAtlasL2, i.uv, o.t0, o.t1);}
#endif
#if TARGET_MIP >= 4
void DownsamplePyramidsPS3(in VSOUT i, out PSOUT2 o){tile_downsample(sLaunchpadExposureAtlasL3, sLaunchpadWeightAtlasL3, i.uv, o.t0, o.t1);}
#endif
#if TARGET_MIP >= 5
void DownsamplePyramidsPS4(in VSOUT i, out PSOUT2 o){tile_downsample(sLaunchpadExposureAtlasL4, sLaunchpadWeightAtlasL4, i.uv, o.t0, o.t1);}
#endif
#if TARGET_MIP >= 6
void DownsamplePyramidsPS5(in VSOUT i, out PSOUT2 o){tile_downsample(sLaunchpadExposureAtlasL5, sLaunchpadWeightAtlasL5, i.uv, o.t0, o.t1);}
#endif
#if TARGET_MIP >= 7
void DownsamplePyramidsPS6(in VSOUT i, out PSOUT2 o){tile_downsample(sLaunchpadExposureAtlasL6, sLaunchpadWeightAtlasL6, i.uv, o.t0, o.t1);}
#endif

void fetch_layers(sampler s, float2 uv, out float4 layers[4])
{
    const float2 num_tiles = float2(ATLAS_TILES_X, ATLAS_TILES_Y);
    float2 tile_res = int2(tex2Dsize(s, 0)) / int2(num_tiles);
    float2 texelsize = rcp(tile_res);

    uv = clamp(uv, texelsize * 0.5, 1 - texelsize * 0.5);

    [unroll]
    for(int j = 0; j < ATLAS_TILES_X * ATLAS_TILES_Y; j++)
    {
        int x = j % ATLAS_TILES_X;
        int y = j / ATLAS_TILES_X;
        float2 tile_uv = (uv + float2(x, y)) / num_tiles; 
        layers[j] = tex2Dlod(s, tile_uv, 0);
    }
}

float hadamard(float4 A[4], float4 B[4])
{
    return dot(A[0], B[0]) + dot(A[1], B[1]) + dot(A[2], B[2]) + dot(A[3], B[3]);
}

float l1norm(float4 v[4])
{
    return dot(v[0], 1) + dot(v[1], 1) + dot(v[2], 1) + dot(v[3], 1);
}

float balance(int layer)
{
    float x = float(layer)/float(TARGET_MIP);
    return exp2(-x * 3.0);
}

void CollapseTiledPyramidPS(in VSOUT i, out float2 o : SV_Target0)
{
    float collapsed = 0;

    float4 G[4];
    float4 W[4];
    float weightsum;

    float total_weightsum = 0;
    float unnormalized_weightsum = 0;
    float denom = float(TARGET_MIP);

    fetch_layers(sLaunchpadExposureAtlasL0, i.uv, G);
    fetch_layers(sLaunchpadWeightAtlasL0, i.uv, W);
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(0); //G0 x W0 add
#if TARGET_MIP >= 1 
    fetch_layers(sLaunchpadExposureAtlasL1, i.uv, G);//fetch G1
    collapsed -= hadamard(G, W) / max(1e-7, weightsum) * balance(1);  //G1 x W0 subtract, this completes the first laplacian
    fetch_layers(sLaunchpadWeightAtlasL1, i.uv, W); //fetch W1
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(1); //G1 x W1 add, this is either residual if we stop here or first half of next laplacian
#endif
#if TARGET_MIP >= 2
    fetch_layers(sLaunchpadExposureAtlasL2, i.uv, G);//fetch G2
    collapsed -= hadamard(G, W) / max(1e-7, weightsum) * balance(2); //G2 x W1 subtract, this completes the second laplacian
    fetch_layers(sLaunchpadWeightAtlasL2, i.uv, W); //fetch W2
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(2); //G2 x W2 add, this is either residual if we stop here or first half of next laplacian
#endif 
#if TARGET_MIP >= 3
    fetch_layers(sLaunchpadExposureAtlasL3, i.uv, G);//fetch G3
    collapsed -= hadamard(G, W) / max(1e-7, weightsum) * balance(3); //G3 x W2 subtract, this completes the second laplacian
    fetch_layers(sLaunchpadWeightAtlasL3, i.uv, W); //fetch W3
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(3); //G3 x W3 add, this is either residual if we stop here or first half of next laplacian
#endif 
#if TARGET_MIP >= 4
    fetch_layers(sLaunchpadExposureAtlasL4, i.uv, G);//fetch G4
    collapsed -= hadamard(G, W) / max(1e-7, weightsum) * balance(4); //G4 x W3 subtract, this completes the second laplacian
    fetch_layers(sLaunchpadWeightAtlasL4, i.uv, W); //fetch W4
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(4); //G4 x W4 add, this is either residual if we stop here or first half of next laplacian
#endif 
#if TARGET_MIP >= 5
    fetch_layers(sLaunchpadExposureAtlasL5, i.uv, G);//fetch G5
    collapsed -= hadamard(G, W) / max(1e-7, weightsum) * balance(5); //G5 x W4 subtract, this completes the second laplacian
    fetch_layers(sLaunchpadWeightAtlasL5, i.uv, W); //fetch W5
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(5); //G5 x W5 add, this is either residual if we stop here or first half of next laplacian
 #endif 
#if TARGET_MIP >= 6
    fetch_layers(sLaunchpadExposureAtlasL6, i.uv, G);//fetch G6
    collapsed -= hadamard(G, W) / max(1e-7, weightsum) * balance(6); //G6 x W5 subtract, this completes the second laplacian
    fetch_layers(sLaunchpadWeightAtlasL6, i.uv, W); //fetch W6
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(6); //G6 x W6 add, this is either residual if we stop here or first half of next laplacian
#endif 
#if TARGET_MIP >= 7
    fetch_layers(sLaunchpadExposureAtlasL7, i.uv, G);//fetch G7
    collapsed -= hadamard(G, W) / max(1e-7, weightsum) * balance(7); //G7 x W6 subtract, this completes the second laplacian
    fetch_layers(sLaunchpadWeightAtlasL7, i.uv, W); //fetch W7
    weightsum = l1norm(W);
    collapsed += hadamard(G, W) / max(1e-7, weightsum) * balance(7); //G7 x W7 add, this is either residual if we stop here or first half of next laplacian
#endif 
    o.x = collapsed; //collapsed pyramid
    o.y = get_sdr_luma(tex2D(ColorInput, i.uv).rgb);
}

void UpsampleAtlasNewPS(in VSOUT i, out float4 o : SV_Target0)
{
    float3 c = tex2D(ColorInput, i.uv).rgb;
    float luminance = get_sdr_luma(c); 

    float4 moments = 0; //guide, guide^2, guide*signal, signal
    float ws = 0.0;

    float2 texelsize = rcp(tex2Dsize(sLaunchpadCollapsedExposurePyramidTex, 0));

	float2 minmax = float2(1000, -1000);

    for (int x = -1; x <= 1; x += 1)  
    for (int y = -1; y <= 1; y += 1) 
    {
        float2 offs = float2(x, y);
        float2 t = tex2D(sLaunchpadCollapsedExposurePyramidTex, i.uv + offs * texelsize).xy;
        float w = exp(-0.5 * dot(offs, offs) / (0.7*0.7));
        moments += float4(t.y, t.y * t.y, t.y * t.x, t.x) * w;
        ws += w;

		minmax.x = min(minmax.x, t.x);
		minmax.y = max(minmax.y, t.x);
    }    

    moments /= ws;
    
    float A = (moments.z - moments.x * moments.w) / (max(moments.y - moments.x * moments.x, 0.0) + 0.00001);
    float B = moments.w - A * moments.x;

    float exposure_ratio = A * luminance + B;
	exposure_ratio = clamp(exposure_ratio, minmax.x, minmax.y);
    //exposure_ratio *= saturate(abs(INTENSITY)) * (INTENSITY > 0 ? 1 : -1);
    exposure_ratio = exp2(exposure_ratio);

    o.rgb = tex2D(ColorInput, i.uv).rgb;
    o.rgb = sdr_to_hdr(o.rgb);
    o.rgb *= exposure_ratio;

    //creates less halos this way as it creates less sharp transitions between exposure brackets if exposure target is low/high
    float3 target_hdr = sdr_to_hdr(ALBEDO_EXPOSURE_TARGET.xxx);
    float3 current_hdr = sdr_to_hdr(0.5);
    o.rgb *= target_hdr / current_hdr;
    o.rgb = hdr_to_sdr(o.rgb);    
    o.w = 1;

	{
		o.rgb = saturate(o.rgb);   
		o.rgb = o.rgb*0.283799*((2.52405+o.rgb)*o.rgb);  
		o.rgb = srgb_to_AgX(o.rgb);

		//given scene color is lighting * albedo + lighting * albedo² * k + lighting*albedo³ * k² ... due to multiple bounces
		//with a probability falloff for each consecutive bounce happening. This causes the actual albedo mostly being less saturated
		//than the apparent scene color. I'm fudging things here by assuming a constant white light source. If we invert the MacLaurin series
		//to get the actual source albedo given our assumptions, this turns out to be... a reinhard tonemap curve lmao
		float maclaurin_power = 0.8; 
		float3 albedoinversed = o.rgb / (1.0 + maclaurin_power * o.rgb);
		o.rgb = dot(o.rgb, 0.33333) * albedoinversed / (1e-6 + dot(albedoinversed, 0.33333));	
	}
}

texture RTGI_AlbedoTexV3      { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler sRTGI_AlbedoTexV3     { Texture = RTGI_AlbedoTexV3; };

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
	
	pass { ComputeShader = NormalsCS<32, 32>;DispatchSizeX = CEIL_DIV(BUFFER_WIDTH_DLSS, 32); DispatchSizeY = CEIL_DIV(BUFFER_HEIGHT_DLSS, 32);}
	//pass {VertexShader = MainVS;PixelShader = NormalsPS; RenderTarget = Deferred::NormalsTexV3; }	
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsMakeGbufPS;  RenderTarget = SmoothNormalsTempTex0;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass0PS;  RenderTarget = SmoothNormalsTempTex1;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass1PS;  RenderTarget = SmoothNormalsTempTex2;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = CopyNormalsPS; RenderTarget = Deferred::NormalsTexV3; }

	pass {VertexShader = MainVS;PixelShader = WriteDepthFeaturePS;  RenderTarget0 = DepthLowresPacked; RenderTargetWriteMask = 1 << 0;} 
    pass {VertexShader = MainVS;PixelShader = WriteFeaturePS; 	    RenderTarget0 = FeaturePyramidLevel0; RenderTargetWriteMask = 1 << 0;} 
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS1;	RenderTarget = FeaturePyramidLevel1;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS2;	RenderTarget = FeaturePyramidLevel2;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS3;	RenderTarget = FeaturePyramidLevel3;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS4;	RenderTarget = FeaturePyramidLevel4;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS5;	RenderTarget = FeaturePyramidLevel5;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS6;	RenderTarget = FeaturePyramidLevel6;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS7;	RenderTarget = FeaturePyramidLevel7;}

	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS8;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS7;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS6;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS5;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS4;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS3;	RenderTarget = MotionTexNewB;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS2;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS1;	RenderTarget = MotionTexNewB;}

	pass {VertexShader = MainVS;PixelShader = UpscaleFlowPS0;		RenderTarget = MotionTexUpscale;}
	pass {VertexShader = MainVS;PixelShader = UpscaleFlowPS1;		RenderTarget = MotionTexUpscale2;}
		
	pass {VertexShader = MainVS;PixelShader = CopyToFullres;		RenderTarget = MotionTexIntermediateTex0;}

	pass {VertexShader = MainVS;PixelShader = WritePrevLowresDepthPS; RenderTarget0 = DepthLowresPacked; RenderTargetWriteMask = 1 << 1;} 
	pass {VertexShader = MainVS;PixelShader = WriteFeaturePS2; RenderTarget0 = FeaturePyramidLevel0; RenderTargetWriteMask = 1 << 1;}	

	pass{VertexShader = MainVS; PixelShader = InitPyramidAtlasPS;   RenderTarget0 = LaunchpadExposureAtlasL0; 
                                                                	RenderTarget1 = LaunchpadWeightAtlasL0;}
#if TARGET_MIP >= 1
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS0; RenderTarget0 = LaunchpadExposureAtlasL1; RenderTarget1 = LaunchpadWeightAtlasL1; }
#if TARGET_MIP >= 2
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS1; RenderTarget0 = LaunchpadExposureAtlasL2; RenderTarget1 = LaunchpadWeightAtlasL2; }
#if TARGET_MIP >= 3 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS2; RenderTarget0 = LaunchpadExposureAtlasL3; RenderTarget1 = LaunchpadWeightAtlasL3; }
#if TARGET_MIP >= 4 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS3; RenderTarget0 = LaunchpadExposureAtlasL4; RenderTarget1 = LaunchpadWeightAtlasL4; }
#if TARGET_MIP >= 5 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS4; RenderTarget0 = LaunchpadExposureAtlasL5; RenderTarget1 = LaunchpadWeightAtlasL5; }
#if TARGET_MIP >= 6 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS5; RenderTarget0 = LaunchpadExposureAtlasL6; RenderTarget1 = LaunchpadWeightAtlasL6; }
#if TARGET_MIP >= 7 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS6; RenderTarget0 = LaunchpadExposureAtlasL7; RenderTarget1 = LaunchpadWeightAtlasL7; } 
#if TARGET_MIP >= 8 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS7; RenderTarget0 = LaunchpadExposureAtlasL8; RenderTarget1 = LaunchpadWeightAtlasL8; } 
#if TARGET_MIP >= 9 
    pass    {VertexShader = MainVS;PixelShader = DownsamplePyramidsPS8; RenderTarget0 = LaunchpadExposureAtlasL9; RenderTarget1 = LaunchpadWeightAtlasL9; } 
#endif 
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif 
    pass{VertexShader = MainVS; PixelShader = CollapseTiledPyramidPS;  RenderTarget0 = LaunchpadCollapsedExposurePyramidTex;}  
	pass{VertexShader = MainVS; PixelShader = UpsampleAtlasNewPS;  RenderTarget = RTGI_AlbedoTexV3;} 

#if LAUNCHPAD_DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	pass {VertexShader = MainVS;PixelShader  = DebugPS;  }			
#endif
}
