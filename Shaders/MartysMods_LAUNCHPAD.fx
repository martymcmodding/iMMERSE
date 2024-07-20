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
	ui_items = "Half Resolution\0Full Resolution\0";
	ui_tooltip = "Higher resolution vectors are more accurate but cost more performance.";
    ui_category = "OPTICAL FLOW";
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

uniform bool debug_key_down < source = "key"; keycode = 0x46; mode = ""; >;
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

uniform uint FRAMECOUNT < source = "framecount"; >;
uniform float FRAMETIME < source = "frametime"; >;

//don't touch, slight changes can have catastrophic effects on performance
#define POOL_RADIUS 	5.0//10 * tempF1.x //10.0 * step(0, tempF1.x)
#define UPSCALE_RADIUS  1.5//2.5 * tempF1.y//2.5 * step(0, tempF1.x)

texture MotionTexNewA               { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F;};
sampler sMotionTexNewA              { Texture = MotionTexNewA;   MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };
texture MotionTexNewB               { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F;};
sampler sMotionTexNewB              { Texture = MotionTexNewB;  MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

//Yes I know you like to optimize blue noise away in favor for some shitty PRNG function, don't.
texture BlueNoiseJitterTex     < source = "iMMERSE_bluenoise.png"; > { Width = 32; Height = 32; Format = RGBA8; };
sampler	sBlueNoiseJitterTex   { Texture = BlueNoiseJitterTex; AddressU = WRAP; AddressV = WRAP; };

#define MotionTexIntermediateTex0 			Deferred::MotionVectorsTex
#define sMotionTexIntermediateTex0 			Deferred::sMotionVectorsTex

//curr in x, prev in y
#if __RENDERER__ >= RENDERER_D3D10
texture FeaturePyramidLevel0   { Width = BUFFER_WIDTH;   	  Height = BUFFER_HEIGHT;        Format = RG8; };
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
#else //fewer textures that still need to cover a wide range, so we use a pyramid with 3x3 reduction instead of 2x2
texture FeaturePyramidLevel0   { Width = BUFFER_WIDTH;   	Height = BUFFER_HEIGHT;   Format = RG8; };
texture FeaturePyramidLevel1   { Width = BUFFER_WIDTH/3;   	Height = BUFFER_HEIGHT/3;   Format = RG16F;};
texture FeaturePyramidLevel2   { Width = BUFFER_WIDTH/9;   	Height = BUFFER_HEIGHT/9;   Format = RG16F;};
texture FeaturePyramidLevel3   { Width = BUFFER_WIDTH/27;   Height = BUFFER_HEIGHT/27;  Format = RG16F;};
texture FeaturePyramidLevel4   { Width = BUFFER_WIDTH/81;   Height = BUFFER_HEIGHT/81;  Format = RG16F;};

sampler sFeaturePyramidLevel0  { Texture = FeaturePyramidLevel0; AddressU = MIRROR; AddressV = MIRROR; }; 
sampler sFeaturePyramidLevel1  { Texture = FeaturePyramidLevel1; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel2  { Texture = FeaturePyramidLevel2; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel3  { Texture = FeaturePyramidLevel3; AddressU = MIRROR; AddressV = MIRROR; };
sampler sFeaturePyramidLevel4  { Texture = FeaturePyramidLevel4; AddressU = MIRROR; AddressV = MIRROR; };
#endif

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

static float2 block_kernel[13] = 
{
	float2(0,  0), float2( 0, -1), float2( 0,  1), float2(-1,  0),	
	float2(1,  0), float2( 0, -2), float2( 0,  2), float2(-2,  0),	
	float2(2,  0), float2(-2, -2), float2( 2,  2), float2(-2,  2),	
	float2(2, -2)/*, 
	float2(0, -4), float2( 0,  4), float2(-4,  0),	
	float2(4,  0)*/
};

/*=============================================================================
	Functions
=============================================================================*/

float get_prev_depth(float2 uv)
{
	return tex2Dlod(sDepthLowresPacked, uv, 0).y;
}

float get_curr_depth(float2 uv)
{
	return tex2Dlod(sDepthLowresPacked, uv, 0).x;
}

#if __RENDERER__ >= RENDERER_D3D10

float2 downsample_feature(sampler s, float2 uv)
{
	float2 res = 0;	
	float2 texelsize = rcp(tex2Dsize(s));
	float wsum = 0;
	for(int x = 0; x < 6; x++)
	for(int y = 0; y < 6; y++)
	{
		float2 offs = float2(x, y); //0 to 5
		offs -= 2.5; // -2.5 to 2.5
		float g = exp(-dot(offs, offs) * 0.1);
		res += g * tex2D(s, uv + offs * texelsize).rg;
		wsum += g;
	}
	return res / wsum;	
}

void DownsampleFeaturePS1(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel0, i.uv);} 
void DownsampleFeaturePS2(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel1, i.uv);} 
void DownsampleFeaturePS3(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel2, i.uv);} 
void DownsampleFeaturePS4(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel3, i.uv);} 
void DownsampleFeaturePS5(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel4, i.uv);} 
void DownsampleFeaturePS6(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel5, i.uv);}
void DownsampleFeaturePS7(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature(sFeaturePyramidLevel6, i.uv);}

#else 

float2 downsample_feature_tri(sampler s, float2 uv)
{
	float2 res = 0;	
	float2 texelsize = rcp(tex2Dsize(s));
	float wsum = 0;
	for(int x = -6; x <= 6; x++)
	for(int y = -6; y <= 6; y++)
	{
		float2 offs = float2(x, y); 
		float g = exp(-dot(offs, offs) * 0.03);
		res += g * tex2D(s, uv + offs * texelsize).rg;
		wsum += g;
	}
	return res / wsum;	
}

void DownsampleFeatureTriPS1(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature_tri(sFeaturePyramidLevel0, i.uv);}
void DownsampleFeatureTriPS2(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature_tri(sFeaturePyramidLevel1, i.uv);}
void DownsampleFeatureTriPS3(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature_tri(sFeaturePyramidLevel2, i.uv);}
void DownsampleFeatureTriPS4(in VSOUT i, out float2 o : SV_Target0){o = downsample_feature_tri(sFeaturePyramidLevel3, i.uv);}

#endif

float3 get_jitter_blue(in int2 pos)
{
	return tex2Dfetch(sBlueNoiseJitterTex, pos % 32).xyz;
}

float loss(float a, float b)
{
	float t = a - b;
	return t*t;//abs(t); //SAD	
}

struct AdamOptimizer
{
	float2 m;
	float v;
	float beta1decayed, beta2decayed;
	float beta1, beta2, epsilon;
	float lr;
};


AdamOptimizer init_adam()
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

	float2 mhat = a.m / (1 - a.beta1decayed);
	float vhat  = a.v / (1 - a.beta2decayed);

	return a.lr * mhat / (sqrt(vhat) + a.epsilon);
}

float4 gradient_block_matching(sampler s_feature, VSOUT i, int level, float4 coarse_layer, const int blocksize)
{	
	float2 texelsize = rcp(tex2Dsize(s_feature));	
	float2 search_scale = texelsize;
	float2 total_motion = coarse_layer.xy;

	float2 m = 0;
	float local_block[13];

	float3 SAD = 0; //center, +dx, +dy
	float2 deltax = texelsize * float2(0.0625, 0);
	float2 deltay = texelsize * float2(0, 0.0625);

	[unroll]
	for(uint k = 0; k < blocksize; k++)
	{
		float2 tuv = i.uv + block_kernel[k] * search_scale;

		float g = tex2Dlod(s_feature, tuv, 0).x;
		float f;	

		f = tex2Dlod(s_feature, tuv + total_motion, 0).y;
		SAD.x += loss(f, g);
		f = tex2Dlod(s_feature, tuv + total_motion + deltax, 0).y;		
		SAD.y += loss(f, g);
		f = tex2Dlod(s_feature, tuv + total_motion + deltay, 0).y;
		SAD.z += loss(f, g);

		local_block[k] = g;

	}	
	
	m /= blocksize;

	AdamOptimizer adam = init_adam();
	float2 grad = (SAD.yz - SAD.x) / float2(deltax.x, deltay.y);

	float2 local_motion = 0;
	float2 best_local_motion = 0;
	float  best_SAD = SAD.x;
	
	int num_steps = 24;
	int did_not_improve_score = 0;

	[loop]
	while(num_steps-- > 0 && did_not_improve_score < 2)
	{
		local_motion -= update_adam(adam, grad);
		SAD = 0;
		m = 0;

		[unroll]
		for(uint k = 0; k < blocksize; k++)
		{
			float2 tuv = i.uv + total_motion + local_motion + block_kernel[k] * search_scale;

			float g = local_block[k];
			float f;

			f = tex2Dlod(s_feature, tuv, 0).y;	
			SAD.x += loss(f, g);
			f = tex2Dlod(s_feature, tuv + deltax, 0).y;
			SAD.y += loss(f, g);
			f = tex2Dlod(s_feature, tuv + deltay, 0).y;
			SAD.z += loss(f, g);

			m += float2(g, g * g);
		}

		[flatten]
		if(SAD.x < best_SAD)
		{
			best_SAD = SAD.x;
			best_local_motion = local_motion;
			did_not_improve_score = 0;			
		}
		else 
		{
			did_not_improve_score++;
		}		

		grad = (SAD.yz - SAD.x) / float2(deltax.x, deltay.y);
	}

	local_motion = best_local_motion;
	total_motion += local_motion;

	float variance = abs(m.y - m.x * m.x);
	m /= blocksize;
	
	best_SAD /= blocksize;
	best_SAD /= max(0.0001, variance);

	float prev_depth_at_motion = get_prev_depth(i.uv + total_motion);
	float4 curr_layer = float4(total_motion, prev_depth_at_motion, best_SAD);
	return curr_layer;
}

float hash11(float p)
{
    p = frac(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return frac(p);
}

float4 pool_vectors_new(VSOUT i, int level, sampler motion_tex, sampler feature_tex, float radius)
{	
	float center_z = get_curr_depth(i.uv);
	if(level == 0) center_z = Depth::get_linear_depth(i.uv);
	float3 jitter = get_jitter_blue(i.vpos.xy);

	float2 texelsize = rcp(tex2Dsize(motion_tex));

	float2 kernel_scale = texelsize * radius;

	float4 flow_filtered = 0;
	float4 flow_reservoir = 0;

	float wsum_filtered = 0;
	float wsum_reservoir = 0;
	float fseed = jitter.z;

	float4 center = tex2Dlod(motion_tex, i.uv, 0);
	flow_filtered = center * 0.001;
	wsum_filtered = 0.001;

	[loop]for(int x = -2; x <= 1; x++)
	[loop]for(int y = -2; y <= 1; y++)
	{
		float2 offs = float2(x, y) + jitter.xy;
		float2 sample_uv = i.uv + offs * kernel_scale;		

		if(!Math::inside_screen(sample_uv)) continue;

		float4 flow = tex2Dlod(motion_tex, sample_uv, 0);

		float match_error        = flow.w * 8000.0;
		float flow_length_pixels = dot(flow.xy * BUFFER_ASPECT_RATIO, flow.xy * BUFFER_SCREEN_SIZE);

		float sample_z = get_curr_depth(sample_uv);
		float dzc = abs(center_z - sample_z) / max(1e-5, min(center_z, sample_z));
		float dzp = abs(center_z - flow.z) / max(1e-5, min(center_z, flow.z));

		float wfactor = (dzc * dzc * 4 + dzp * dzp) * 32.0 + match_error * match_error * 4;
		float w = exp2(-wfactor);
		w = max(w, 0.001);

		flow_filtered += flow * w;
		wsum_filtered += w;	

		//clean weight
		w = exp2(-wfactor);
		w *= w;
	
		wsum_reservoir += w;			

		float rand = hash11(fseed * 161.88 + sin(w * 142.33));
		fseed = rand;

		if(rand * wsum_reservoir < w)
			flow_reservoir = flow;
	}

	flow_filtered /= max(1e-3, wsum_filtered);

	if(level > 0.5)
	{	
		texelsize = rcp(tex2Dsize(feature_tex));	
		float loss_avg = 0;	
		float loss_res = 0;	
		int num_samples = 9;

		[loop]
		for(uint k = 0; k < num_samples; k++)
		{
			float2 kernel_uv = i.uv + block_kernel[k] * texelsize;
			float f = tex2Dlod(feature_tex, kernel_uv, 0).x;
			float g1 = tex2Dlod(feature_tex, kernel_uv + flow_filtered.xy, 0).y;
			float g2 = tex2Dlod(feature_tex, kernel_uv + flow_reservoir.xy, 0).y;
			loss_avg += loss(f, g1);
			loss_res += loss(f, g2);
		}

		if(loss_res < loss_avg)
			flow_filtered = flow_reservoir;
	}

	return flow_filtered;
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(log(1 + dist * 400.0  / FRAMETIME)));//normalize by frametime such that we don't need to adjust visualization intensity all the time
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
	float4 feature_data = 0;
#if MIN_MIP > 0
	const float4 radius = float4(0.7577, -0.7577, 2.907, 0);
	const float2 weight = float2(0.37487566, -0.12487566);
	feature_data.rgb =  weight.x * tex2D(ColorInput, i.uv + radius.xx * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.xy * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.yx * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.yy * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv + radius.zw * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv - radius.zw * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv + radius.wz * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv - radius.wz * BUFFER_PIXEL_SIZE).xyz;	
#else	
	feature_data.rgb = tex2D(ColorInput, i.uv).rgb;
#endif	

	o = dot(0.3333, feature_data.rgb);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x + 1) discard;
}

void WritePrevLowresDepthPS(in VSOUT i, out float2 o : SV_Target0)
{
	o = Depth::get_linear_depth(i.uv);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

void WriteFeaturePS2(in VSOUT i, out float4 o : SV_Target0)
{	
	float4 feature_data = 0;
#if MIN_MIP > 0
	const float4 radius = float4(0.7577, -0.7577, 2.907, 0);
	const float2 weight = float2(0.37487566, -0.12487566);
	feature_data.rgb =  weight.x * tex2D(ColorInput, i.uv + radius.xx * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.xy * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.yx * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.x * tex2D(ColorInput, i.uv + radius.yy * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv + radius.zw * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv - radius.zw * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv + radius.wz * BUFFER_PIXEL_SIZE).xyz;
	feature_data.rgb += weight.y * tex2D(ColorInput, i.uv - radius.wz * BUFFER_PIXEL_SIZE).xyz;	
#else	
	feature_data.rgb = tex2D(ColorInput, i.uv).rgb;
#endif	
	feature_data.w = Depth::get_linear_depth(i.uv);	

	o = dot(0.3333, feature_data.rgb);
	//if(FRAMECOUNT > tex2Dfetch(sStateCounterTex, int2(0, 0)).x) discard;
}

#if __RENDERER__ >= RENDERER_D3D10

void BlockMatchingPassPS8(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel7, i, 8, 0.0.xxxx, 13);}
void FilterPass8(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 8, sMotionTexNewA, sFeaturePyramidLevel7, POOL_RADIUS);}
void BlockMatchingPassPS7(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel6, i, 7, pool_vectors_new(i, 7, sMotionTexNewB, sFeaturePyramidLevel6, POOL_RADIUS), 13);}
void FilterPass7(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 7, sMotionTexNewA, sFeaturePyramidLevel6, POOL_RADIUS);}
void BlockMatchingPassPS6(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel5, i, 6, pool_vectors_new(i, 5, sMotionTexNewB, sFeaturePyramidLevel5, POOL_RADIUS), 13);}
void FilterPass6(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 5, sMotionTexNewA, sFeaturePyramidLevel5, POOL_RADIUS);}
void BlockMatchingPassPS5(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel4, i, 5, pool_vectors_new(i, 4, sMotionTexNewB, sFeaturePyramidLevel4, POOL_RADIUS), 13);}
void FilterPass5(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 4, sMotionTexNewA, sFeaturePyramidLevel4, POOL_RADIUS);}
void BlockMatchingPassPS4(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel3, i, 4, pool_vectors_new(i, 3, sMotionTexNewB, sFeaturePyramidLevel3, POOL_RADIUS), 13);}
void FilterPass4(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 3, sMotionTexNewA, sFeaturePyramidLevel3, POOL_RADIUS);}
void BlockMatchingPassPS3(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel2, i, 3, pool_vectors_new(i, 2, sMotionTexNewB, sFeaturePyramidLevel2, POOL_RADIUS), 13);}
void FilterPass3(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 2, sMotionTexNewA, sFeaturePyramidLevel2, POOL_RADIUS);}
void BlockMatchingPassPS2(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel1, i, 2, pool_vectors_new(i, 1, sMotionTexNewB, sFeaturePyramidLevel1, POOL_RADIUS), 13);}
void FilterPass2(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 1, sMotionTexNewA, sFeaturePyramidLevel1, POOL_RADIUS);}
void BlockMatchingPassPS1(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel0, i, 1, pool_vectors_new(i, 0, sMotionTexNewB, sFeaturePyramidLevel0, POOL_RADIUS), 13);}
void FilterPass1(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 0, sMotionTexNewA, sFeaturePyramidLevel0, POOL_RADIUS);}

void CopyToFullres(in VSOUT i, out float4 o : SV_Target0)
{
	o = pool_vectors_new(i, 0, sMotionTexNewB, sFeaturePyramidLevel0, UPSCALE_RADIUS);
	[branch]
	if(OPTICAL_FLOW_RES == 1) 
		o = gradient_block_matching(sFeaturePyramidLevel0, i, 0, o, 9);
}

#else

void BlockMatchingPassPS5(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel4, i, 5, 0.0.xxxx, 13);}
void FilterPass5(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 4, sMotionTexNewA, sFeaturePyramidLevel4, POOL_RADIUS);}
void BlockMatchingPassPS4(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel3, i, 4, pool_vectors_new(i, 3, sMotionTexNewB, sFeaturePyramidLevel3, POOL_RADIUS), 13);}
void FilterPass4(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 3, sMotionTexNewA, sFeaturePyramidLevel3, POOL_RADIUS);}
void BlockMatchingPassPS3(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel2, i, 3, pool_vectors_new(i, 2, sMotionTexNewB, sFeaturePyramidLevel2, POOL_RADIUS), 13);}
void FilterPass3(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 2, sMotionTexNewA, sFeaturePyramidLevel2, POOL_RADIUS);}
void BlockMatchingPassPS2(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel1, i, 2, pool_vectors_new(i, 1, sMotionTexNewB, sFeaturePyramidLevel1, POOL_RADIUS), 13);}
void FilterPass2(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 1, sMotionTexNewA, sFeaturePyramidLevel1, POOL_RADIUS);}
void BlockMatchingPassPS1(in VSOUT i, out float4 o : SV_Target0){o = gradient_block_matching(sFeaturePyramidLevel0, i, 1, pool_vectors_new(i, 0, sMotionTexNewB, sFeaturePyramidLevel0, POOL_RADIUS), 13);}
void FilterPass1(in VSOUT i, out float4 o : SV_Target0){o = pool_vectors_new(i, 0, sMotionTexNewA, sFeaturePyramidLevel0, POOL_RADIUS);}

void CopyToFullres(in VSOUT i, out float4 o : SV_Target0)
{
	o = pool_vectors_new(i, 0, sMotionTexNewB, sFeaturePyramidLevel0, UPSCALE_RADIUS);
	[branch]
	if(OPTICAL_FLOW_RES == 1) 
		o = gradient_block_matching(sFeaturePyramidLevel0, i, 0, o, 9);
}

#endif

/*=============================================================================
	Shader Entry Points - Normals
=============================================================================*/

void NormalsPS(in VSOUT i, out float4 o : SV_Target0)
{
	//TODO optimize with tex2Dgather? Compute? What about scaled depth buffers? oh man
	const float2 dirs[9] = 
	{
		BUFFER_PIXEL_SIZE * float2(-1,-1),//TL
		BUFFER_PIXEL_SIZE * float2(0,-1),//T
		BUFFER_PIXEL_SIZE * float2(1,-1),//TR
		BUFFER_PIXEL_SIZE * float2(1,0),//R
		BUFFER_PIXEL_SIZE * float2(1,1),//BR
		BUFFER_PIXEL_SIZE * float2(0,1),//B
		BUFFER_PIXEL_SIZE * float2(-1,1),//BL
		BUFFER_PIXEL_SIZE * float2(-1,0),//L
		BUFFER_PIXEL_SIZE * float2(-1,-1)//TL first duplicated at end cuz it might be best pair	
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
texture SmoothNormalsTempTex0  { Width = BUFFER_WIDTH/2;   Height = BUFFER_HEIGHT/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex0 { Texture = SmoothNormalsTempTex0; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT; };
//gbuffer halfres for fast filtering
texture SmoothNormalsTempTex1  { Width = BUFFER_WIDTH/2;   Height = BUFFER_HEIGHT/2;   Format = RGBA16F;  };
sampler sSmoothNormalsTempTex1 { Texture = SmoothNormalsTempTex1; MinFilter = POINT; MagFilter = POINT; MipFilter = POINT;  };
//high res copy back so we can fetch center tap at full res always
texture SmoothNormalsTempTex2  < pooled = true; > { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16;  };
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

		float3 e_y = (p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE * float2(0, 2)));
		float3 e_x = (p - Camera::uv_to_proj(i.uv + BUFFER_PIXEL_SIZE * float2(2, 0)));
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
	//pass{PrimitiveTopology = POINTLIST;VertexCount = 1;VertexShader = FrameWriteVS;PixelShader  = FrameWritePS;RenderTarget = StateCounterTex;} 
	pass {VertexShader = MainVS;PixelShader = NormalsPS; RenderTarget = Deferred::NormalsTexV2; }		
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsMakeGbufPS;  RenderTarget = SmoothNormalsTempTex0;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass0PS;  RenderTarget = SmoothNormalsTempTex1;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = SmoothNormalsPass1PS;  RenderTarget = SmoothNormalsTempTex2;}
	pass {VertexShader = SmoothNormalsVS;PixelShader = CopyNormalsPS; RenderTarget = Deferred::NormalsTexV2; }
	
	pass {VertexShader = MainVS;PixelShader = WriteDepthFeaturePS; RenderTarget0 = DepthLowresPacked; RenderTargetWriteMask = 1 << 0;} 
    pass {VertexShader = MainVS;PixelShader = WriteFeaturePS; RenderTarget0 = FeaturePyramidLevel0; RenderTargetWriteMask = 1 << 0;} 

#if __RENDERER__ >= RENDERER_D3D10
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS1;	RenderTarget = FeaturePyramidLevel1;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS2;	RenderTarget = FeaturePyramidLevel2;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS3;	RenderTarget = FeaturePyramidLevel3;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS4;	RenderTarget = FeaturePyramidLevel4;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS5;	RenderTarget = FeaturePyramidLevel5;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS6;	RenderTarget = FeaturePyramidLevel6;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeaturePS7;	RenderTarget = FeaturePyramidLevel7;}

	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS8;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass8;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS7;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass7;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS6;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass6;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS5;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass5;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS4;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass4;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS3;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass3;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS2;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass2;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS1;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass1;		    RenderTarget = MotionTexNewB;}
#else 
	pass {VertexShader = MainVS;PixelShader = DownsampleFeatureTriPS1;	RenderTarget = FeaturePyramidLevel1;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeatureTriPS2;	RenderTarget = FeaturePyramidLevel2;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeatureTriPS3;	RenderTarget = FeaturePyramidLevel3;}
	pass {VertexShader = MainVS;PixelShader = DownsampleFeatureTriPS4;	RenderTarget = FeaturePyramidLevel4;}
	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS5;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass5;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS4;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass4;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS3;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass3;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS2;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass2;		    RenderTarget = MotionTexNewB;}	
	pass {VertexShader = MainVS;PixelShader = BlockMatchingPassPS1;	RenderTarget = MotionTexNewA;}
	pass {VertexShader = MainVS;PixelShader = FilterPass1;		    RenderTarget = MotionTexNewB;}
		
	pass {VertexShader = MainVS;PixelShader = CopyToFullres;		RenderTarget = MotionTexIntermediateTex0;}
#endif
		
	pass {VertexShader = MainVS;PixelShader = CopyToFullres;		RenderTarget = MotionTexIntermediateTex0;}

	pass {VertexShader = MainVS;PixelShader = WritePrevLowresDepthPS; RenderTarget0 = DepthLowresPacked; RenderTargetWriteMask = 1 << 1;} 
	pass {VertexShader = MainVS;PixelShader = WriteFeaturePS2; RenderTarget0 = FeaturePyramidLevel0; RenderTargetWriteMask = 1 << 1;}	

#if LAUNCHPAD_DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	pass {VertexShader = MainVS;PixelShader  = DebugPS;  }			
#endif
}