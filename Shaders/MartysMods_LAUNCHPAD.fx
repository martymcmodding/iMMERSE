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

#ifndef OPTICAL_FLOW_MATCHING_LAYERS 
 #define OPTICAL_FLOW_MATCHING_LAYERS 	1		//[0-2] 0=luma, 1=luma + depth, 2 = rgb + depth
#endif

#ifndef OPTICAL_FLOW_RESOLUTION
 #define OPTICAL_FLOW_RESOLUTION 		1		//[0-2] 0=fullres, 1=halfres, 2=quarter res
#endif

#ifndef LAUNCHPAD_DEBUG_OUTPUT
 #define LAUNCHPAD_DEBUG_OUTPUT 	  	0		//[0 or1] 1: enables debug output of the motion vectors
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float FILTER_RADIUS <
	ui_type = "drag";
	ui_label = "Optical Flow Filter Smoothness";
	ui_min = 0.0;
	ui_max = 6.0;	
> = 4.0;

#if LAUNCHPAD_DEBUG_OUTPUT != 0
uniform int DEBUG_MODE < 
    ui_type = "combo";
	ui_items = "All\0Optical Flow\0Normals\0Depth\0";
	ui_label = "Debug Output";
> = 0;
#endif

uniform int UIHELP <
	ui_type = "radio";
	ui_label = " ";	
	ui_text ="\nDescription for preprocessor definitions:\n"
	"\n"
	"OPTICAL_FLOW_MATCHING_LAYERS\n"
	"\n"
	"Determines which data to use for optical flow\n"
	"0: luma (fastest)\n"
	"1: luma + depth (more accurate, slower, recommended)\n"
	"2: YCoCg color + depth (most accurate, slowest)\n"
	"\n"
	"OPTICAL_FLOW_RESOLUTION\n"
	"\n"
	"Resolution factor for optical flow\n"
	"0: full resolution (slowest)\n"
	"1: half resolution (faster, recommended)\n"
	"2: quarter resolution (fastest)\n"
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
#include ".\MartysMods\mmx_camera.fxh"
#include ".\MartysMods\mmx_deferred.fxh"

#define INTERP 			LINEAR
#define FILTER_WIDE	 	true 
#define FILTER_NARROW 	false

#define BLOCK_SIZE 					2
#define SEARCH_OCTAVES              2
#define OCTAVE_SAMPLES             	4

uniform uint FRAME_COUNT < source = "framecount"; >;

#define MAX_MIP  	6 //do not change, tied to textures
#define MIN_MIP 	OPTICAL_FLOW_RESOLUTION

//texture texMotionVectors          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
//sampler sMotionVectorTex         { Texture = texMotionVectors;  };

texture MotionTexIntermediate6               { Width = BUFFER_WIDTH >> 6;   Height = BUFFER_HEIGHT >> 6;   Format = RGBA16F;  };
sampler sMotionTexIntermediate6              { Texture = MotionTexIntermediate6; };
texture MotionTexIntermediate5               { Width = BUFFER_WIDTH >> 5;   Height = BUFFER_HEIGHT >> 5;   Format = RGBA16F;  };
sampler sMotionTexIntermediate5              { Texture = MotionTexIntermediate5; };
texture MotionTexIntermediate4               { Width = BUFFER_WIDTH >> 4;   Height = BUFFER_HEIGHT >> 4;   Format = RGBA16F;  };
sampler sMotionTexIntermediate4              { Texture = MotionTexIntermediate4; };
texture MotionTexIntermediate3               { Width = BUFFER_WIDTH >> 3;   Height = BUFFER_HEIGHT >> 3;   Format = RGBA16F;  };
sampler sMotionTexIntermediate3              { Texture = MotionTexIntermediate3; };
texture MotionTexIntermediate2               { Width = BUFFER_WIDTH >> 2;   Height = BUFFER_HEIGHT >> 2;   Format = RGBA16F;  };
sampler sMotionTexIntermediate2              { Texture = MotionTexIntermediate2; };
texture MotionTexIntermediate1               { Width = BUFFER_WIDTH >> 1;   Height = BUFFER_HEIGHT >> 1;   Format = RGBA16F;  };
sampler sMotionTexIntermediate1              { Texture = MotionTexIntermediate1; };

#define MotionTexIntermediate0 				Deferred::MotionVectorsTex
#define sMotionTexIntermediate0 			Deferred::sMotionVectorsTex

#if OPTICAL_FLOW_MATCHING_LAYERS == 0
 #define FEATURE_FORMAT 	R8 
 #define FEATURE_TYPE 		float
 #define FEATURE_COMPS 		x
#elif OPTICAL_FLOW_MATCHING_LAYERS == 1
 #define FEATURE_FORMAT 	RG16F
 #define FEATURE_TYPE		float2
 #define FEATURE_COMPS 		xy
#else 
 #define FEATURE_FORMAT 	RGBA16F
 #define FEATURE_TYPE		float4
 #define FEATURE_COMPS 		xyzw
#endif

texture FeaturePyramid          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = FEATURE_FORMAT; MipLevels = 1 + MAX_MIP - MIN_MIP; };
sampler sFeaturePyramid         { Texture = FeaturePyramid; MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; }; //MIRROR helps with out of frame disocclusions
texture FeaturePyramidPrev          { Width = BUFFER_WIDTH>>MIN_MIP;   Height = BUFFER_HEIGHT>>MIN_MIP;   Format = FEATURE_FORMAT; MipLevels = 1 + MAX_MIP - MIN_MIP; };
sampler sFeaturePyramidPrev         { Texture = FeaturePyramidPrev;MipFilter=INTERP; MagFilter=INTERP; MinFilter=INTERP; AddressU = MIRROR; AddressV = MIRROR; };

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

/*=============================================================================
	Functions
=============================================================================*/

float4 get_curr_feature(float2 uv, int mip)
{
	mip = max(0, mip - MIN_MIP);
	return tex2Dlod(sFeaturePyramid, saturate(uv), mip);
}

float4 get_prev_feature(float2 uv, int mip)
{
	mip = max(0, mip - MIN_MIP);
	return tex2Dlod(sFeaturePyramidPrev, saturate(uv), mip);
}

float4 find_best_residual_motion(VSOUT i, int level, float4 coarse_layer, const int blocksize)
{	
	level = max(level - 1, 0); //sample one higher res for better quality
	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(level));
	FEATURE_TYPE local_block[16];

	float2 total_motion = coarse_layer.xy;
	float coarse_sim = coarse_layer.w;

	FEATURE_TYPE m1_local = 0;
	FEATURE_TYPE m2_local = 0;
	FEATURE_TYPE m2_search = 0;
	FEATURE_TYPE m_cov = 0;

	i.uv -= texelsize * (blocksize / 2); //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	[unroll] //array index not natively addressable bla...
	for(uint k = 0; k < blocksize * blocksize; k++)
	{
		float2 offs = float2(k % blocksize, k / blocksize);
		float2 tuv = i.uv + offs * texelsize;
		FEATURE_TYPE t_local = get_curr_feature(tuv, level).FEATURE_COMPS; 	
		FEATURE_TYPE t_search = get_prev_feature(tuv + total_motion, level).FEATURE_COMPS;		

		local_block[k] = t_local;

		m1_local += t_local;
		m2_local += t_local * t_local;
		m2_search += t_search * t_search;
		m_cov += t_local * t_search;
	}

	float best_sim = minc(m_cov * rsqrt(m2_local * m2_search));

	float phi = radians(360.0 / OCTAVE_SAMPLES);
	float4 rotator = Math::get_rotator(phi);
	float randseed = (((dot(uint2(i.vpos.xy) % 5, float2(1, 5)) * 17) % 25) + 0.5) / 25.0; //prime shuffled, similar spectral properties to bayer but faster to compute and unique values within 5x5
	randseed = frac(randseed + level * 0.6180339887498);

	float2 randdir; sincos(randseed * phi, randdir.x, randdir.y);
	int _octaves = SEARCH_OCTAVES + (level >= 1 ? 2 : 0);

	while(_octaves-- > 0)
	{
		_octaves = best_sim < 0.999999 ? _octaves : 0;
		float2 local_motion = 0;

		int _samples = OCTAVE_SAMPLES;
		while(_samples-- > 0)		
		{
			_samples = best_sim < 0.999999 ? _samples : 0;
			randdir = Math::rotate_2D(randdir, rotator);
			float2 search_offset = randdir * texelsize;
			float2 search_center = i.uv + total_motion + search_offset;			 

			m2_search = 0;
			m_cov = 0;

			[loop]
			for(uint k = 0; k < blocksize * blocksize; k++)
			{
				FEATURE_TYPE t = get_prev_feature(search_center + float2(k % blocksize, k / blocksize) * texelsize, level).FEATURE_COMPS;
				m2_search += t * t;
				m_cov += local_block[k] * t;
			}

			float sim = minc(m_cov * rsqrt(m2_local * m2_search));

			[flatten]
			if(sim > best_sim)
			{
				best_sim = sim;
				local_motion = search_offset;	
			}				
		}
		total_motion += local_motion;
		randdir *= 0.5;
	}

	m1_local /= BLOCK_SIZE * BLOCK_SIZE;	
	m2_local /= BLOCK_SIZE * BLOCK_SIZE;
	float variance = dot(1, sqrt(abs(m2_local - m1_local * m1_local)));
	float4 curr_layer = float4(total_motion, variance, saturate(1 - acos(best_sim) / (PI * 0.5)));  //delayed sqrt for variance -> stddev
	return curr_layer;
}

float4 atrous_upscale(VSOUT i, int level, sampler sMotionLow, int filter_size)
{	
    float2 texelsize = rcp(tex2Dsize(sMotionLow));
	float rand = frac(level * 0.2114 + (FRAME_COUNT % 16) * 0.6180339887498) * 3.1415927*0.5;
	float2x2 rotm = float2x2(cos(rand), -sin(rand), sin(rand), cos(rand)) * FILTER_RADIUS;
	const float4 gauss = float4(1, 0.85, 0.65, 0.45);

	float4 gbuffer_sum = 0;
	float wsum = 1e-6;
	int rad = filter_size;

	[loop]for(int x = -rad; x <= rad; x++)
	[loop]for(int y = -rad; y <= rad; y++)
	{
		float2 offs = mul(float2(x * abs(x), y * abs(y)), rotm) * texelsize;
		float2 sample_uv = i.uv + offs;	

		float4 sample_gbuf = tex2Dlod(sMotionLow, sample_uv, 0);
		float2 sample_mv = sample_gbuf.xy;
		float sample_var = sample_gbuf.z;
		float sample_sim = sample_gbuf.w;

		float vv = dot(sample_mv, sample_mv);

		float ws = saturate(1.0 - sample_sim);
		float wf = saturate(1 - sample_var * 128.0);

		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4;

		float weight = exp2(-(ws + wm + wf) * 4) * gauss[abs(x)] * gauss[abs(y)];

		weight *= all(saturate(sample_uv - sample_uv * sample_uv));
		gbuffer_sum += sample_gbuf * weight;
		wsum += weight;		
	}

	gbuffer_sum /= wsum;
	return gbuffer_sum;	
}

float4 atrous_upscale_temporal(VSOUT i, int level, sampler sMotionLow, int filter_size)
{	
    float2 texelsize = rcp(tex2Dsize(sMotionLow));
	float rand = frac(level * 0.2114 + (FRAME_COUNT % 16) * 0.6180339887498) * 3.1415927*0.5;
	float2x2 rotm = float2x2(cos(rand), -sin(rand), sin(rand), cos(rand)) * FILTER_RADIUS;
	float4 gauss = float4(1, 0.85, 0.65, 0.45);

	float4 gbuffer_sum = 0;
	float wsum = 1e-6;
	int rad = filter_size;

	[loop]for(int x = -rad; x <= rad; x++)
	[loop]for(int y = -rad; y <= rad; y++)
	{
		float2 offs = mul(float2(x * abs(x), y * abs(y)), rotm) * texelsize;
		float2 sample_uv = i.uv + offs;	

		float4 sample_gbuf = tex2Dlod(sMotionLow, sample_uv, 0);
		float2 sample_mv = sample_gbuf.xy;
		float sample_var = sample_gbuf.z;
		float sample_sim = sample_gbuf.w;

		float vv = dot(sample_mv, sample_mv);

		float2 prev_mv = tex2Dlod(sMotionTexIntermediate0, sample_uv + sample_gbuf.xy, 0).xy;
		float2 mvdelta = prev_mv - sample_mv;
		float diff = dot(mvdelta, mvdelta) * rcp(1e-8 + max(vv, dot(prev_mv, prev_mv)));

		float wd = 3.0 * diff;		
		float ws = saturate(1.0 - sample_sim);
		float wf = saturate(1 - sample_var * 128.0);

		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4;

		float weight = exp2(-(ws + wm + wf + wd) * 4) * gauss[abs(x)] * gauss[abs(y)];

		weight *= all(saturate(sample_uv - sample_uv * sample_uv));
		gbuffer_sum += sample_gbuf * weight;
		wsum += weight;
	}

	gbuffer_sum /= wsum;
	return gbuffer_sum;	
}

float4 motion_pass(in VSOUT i, sampler sMotionLow, int level, int filter_size)
{
	float4 prior_motion = tex2Dlod(sMotionLow, i.uv, 0) * 0.95;
    if(level < MAX_MIP)
    	prior_motion = atrous_upscale(i, level, sMotionLow, filter_size);	

	if(level < MIN_MIP)
		return prior_motion;

	return find_best_residual_motion(i, level, prior_motion, 2);	
}

float4 motion_pass_with_temporal_filter(in VSOUT i, sampler sMotionLow, int level, int filter_size)
{
	float4 prior_motion = tex2Dlod(sMotionLow, i.uv, 0) * 0.95;
    if(level < MAX_MIP)
    	prior_motion = atrous_upscale_temporal(i, level, sMotionLow, filter_size);	

	if(level < MIN_MIP)
		return prior_motion;

	return find_best_residual_motion(i, level, prior_motion, 3);	
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(dist * 100));
}

float3 linear_to_ycocg(float3 color)
{
    float Y  = dot(color, float3(0.25, 0.5, 0.25));
    float Co = dot(color, float3(0.5, 0.0, -0.5));
    float Cg = dot(color, float3(-0.25, 0.5, -0.25));
    return float3(Y, Co, Cg);
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

void WriteFeaturePS(in VSOUT i, out FEATURE_TYPE o : SV_Target0)
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

#if OPTICAL_FLOW_MATCHING_LAYERS == 0
	o = dot(float3(0.299, 0.587, 0.114), feature_data.rgb);
#elif OPTICAL_FLOW_MATCHING_LAYERS == 1
	o.x = dot(float3(0.299, 0.587, 0.114), feature_data.rgb);
	o.y = feature_data.w;
#else 
	float3 ycocg = linear_to_ycocg(feature_data.rgb);
	o = float4(ycocg.x, feature_data.w, ycocg.yz*0.5+0.5);
#endif
}

void MotionPS6(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate2, 6, 2);}
void MotionPS5(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate6, 5, 2);}
void MotionPS4(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate5, 4, 2);}
void MotionPS3(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate4, 3, 2);}
void MotionPS2(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate3, 2, 2);}
void MotionPS1(in VSOUT i, out float4 o : SV_Target0){o = motion_pass(i, sMotionTexIntermediate2, 1, 1);}
void MotionPS0(in VSOUT i, out float4 o : SV_Target0){o = motion_pass(i, sMotionTexIntermediate1, 0, 1);}

void NormalsPS(in VSOUT i, out float2 o : SV_Target0)
{
	float3 delta = float3(BUFFER_PIXEL_SIZE, 0);
    //similar system to Intel ASSAO/AMD CACAO/XeGTAO and friends with improved weighting and less ALU
    float3 center = Camera::uv_to_proj(i.uv);
    float3 deltaL = Camera::uv_to_proj(i.uv - delta.xz) - center;
    float3 deltaR = Camera::uv_to_proj(i.uv + delta.xz) - center;   
    float3 deltaT = Camera::uv_to_proj(i.uv - delta.zy) - center;
    float3 deltaB = Camera::uv_to_proj(i.uv + delta.zy) - center;
    
    float4 zdeltaLRTB = abs(float4(deltaL.z, deltaR.z, deltaT.z, deltaB.z));
    float4 w = zdeltaLRTB.xzyw + zdeltaLRTB.zywx;
    w = rcp(0.001 + w * w); //inverse weighting, larger delta -> lesser weight

    float3 n0 = cross(deltaT, deltaL);
    float3 n1 = cross(deltaR, deltaT);
    float3 n2 = cross(deltaB, deltaR);
    float3 n3 = cross(deltaL, deltaB);

    float4 finalweight = w * rsqrt(float4(dot(n0, n0), dot(n1, n1), dot(n2, n2), dot(n3, n3)));
    float3 normal = n0 * finalweight.x + n1 * finalweight.y + n2 * finalweight.z + n3 * finalweight.w;
    normal *= rsqrt(dot(normal, normal) + 1e-8);    

	o = Math::octahedral_enc(normal);
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
			if(qq == 1) o = Depth::get_linear_depth(tuv);
			if(qq == 2) o = showmotion(Deferred::get_motion(tuv));	
			if(qq == 3) o = tex2Dlod(ColorInput, tuv, 0).rgb;	
			break;			
		}
		case 1: o = showmotion(Deferred::get_motion(i.uv)); break;
		case 2: o = Deferred::get_normals(i.uv) * 0.5 + 0.5; break;
		case 3: o = Depth::get_linear_depth(i.uv); break;
	}	
}
#endif 

/*=============================================================================
	Techniques
=============================================================================*/

technique MartysMods_Launchpad
<
    ui_label = "iMMERSE Launchpad (enable and move to the top!)";
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
    pass {VertexShader = MainVS;PixelShader = WriteFeaturePS; RenderTarget = FeaturePyramid; } 
	pass {VertexShader = MainVS;PixelShader = MotionPS6;RenderTarget = MotionTexIntermediate6;}
    pass {VertexShader = MainVS;PixelShader = MotionPS5;RenderTarget = MotionTexIntermediate5;}
    pass {VertexShader = MainVS;PixelShader = MotionPS4;RenderTarget = MotionTexIntermediate4;}
    pass {VertexShader = MainVS;PixelShader = MotionPS3;RenderTarget = MotionTexIntermediate3;}
    pass {VertexShader = MainVS;PixelShader = MotionPS2;RenderTarget = MotionTexIntermediate2;}
    pass {VertexShader = MainVS;PixelShader = MotionPS1;RenderTarget = MotionTexIntermediate1;}
    pass {VertexShader = MainVS;PixelShader = MotionPS0;RenderTarget = MotionTexIntermediate0;}
	pass {VertexShader = MainVS;PixelShader = WriteFeaturePS; RenderTarget = FeaturePyramidPrev; }
	pass {VertexShader = MainVS;PixelShader = NormalsPS; RenderTarget = Deferred::NormalsTex; }
#if LAUNCHPAD_DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	pass {VertexShader = MainVS;PixelShader  = DebugPS;  }		
#endif 
}