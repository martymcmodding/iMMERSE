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

#ifndef MATCHING_LAYERS 
 #define MATCHING_LAYERS 	1		//[0-2] 0=luma, 1=luma + depth, 2 = rgb + depth
#endif

#ifndef RESOLUTION_SCALE
 #define RESOLUTION_SCALE 	2		//[0-2] 0=fullres, 1=halfres, 2=quarter res
#endif

#ifndef DEBUG_OUTPUT
 #define DEBUG_OUTPUT 	  	0		//[0 or1] 1: enables debug output of the motion vectors
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float FILTER_RADIUS <
	ui_type = "drag";
	ui_label = "Filter Smoothness";
	ui_min = 0.0;
	ui_max = 6.0;	
> = 4.0;

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

#define INTERP 			LINEAR
#define FILTER_WIDE	 	true 
#define FILTER_NARROW 	false

#define BLOCK_SIZE 					3
#define SEARCH_OCTAVES              2
#define OCTAVE_SAMPLES             	4

uniform uint FRAME_COUNT < source = "framecount"; >;

#define MAX_MIP  	6 //do not change, tied to textures
#define MIN_MIP 	RESOLUTION_SCALE

texture texMotionVectors          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
sampler sMotionVectorTex         { Texture = texMotionVectors;  };

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

#define MotionTexIntermediate0 				texMotionVectors
#define sMotionTexIntermediate0 			sMotionVectorTex

#if MATCHING_LAYERS == 0
 #define FEATURE_FORMAT 	R8 
 #define FEATURE_TYPE 		float
 #define FEATURE_COMPS 		x
#elif MATCHING_LAYERS == 1
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

float get_min_cossim(FEATURE_TYPE cs)
{
#if MATCHING_LAYERS == 0
	return cs.x;
#elif MATCHING_LAYERS == 1
	//return dot(cs, 0.5);
	cs = saturate(cs);
	//return cs.x * cs.y;
	return min(cs.x, cs.y);
#else 
	//return dot(cs, 0.25);
	cs = saturate(cs);
	//return cs.x * cs.y * cs.z * cs.w;
	return min(min(cs.x, cs.y), min(cs.z, cs.w));
#endif
}

float4 find_best_residual_motion(VSOUT i, int level, float4 coarse_layer)
{	
	float2 texelsize = rcp(BUFFER_SCREEN_SIZE / exp2(level));
	FEATURE_TYPE local_block[BLOCK_SIZE * BLOCK_SIZE];

	float2 total_motion = coarse_layer.xy;
	float coarse_sim = coarse_layer.w;

	FEATURE_TYPE m1_local = 0;
	FEATURE_TYPE m2_local = 0;
	FEATURE_TYPE m1_search = 0;
	FEATURE_TYPE m2_search = 0;
	FEATURE_TYPE m_cov = 0;

	i.uv -= texelsize * (BLOCK_SIZE / 2); //since we only use to sample the blocks now, offset by half a block so we can do it easier inline

	float4 coeff_local = 0;
	float4 coeff_search = 0;

	[unroll] //array index not natively addressable bla...
	for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
	{
		float2 offs = float2(k / BLOCK_SIZE, k % BLOCK_SIZE);
		float2 tuv = i.uv + offs * texelsize;
		FEATURE_TYPE t_local = get_curr_feature(tuv, level).FEATURE_COMPS; 	
		FEATURE_TYPE t_search = get_prev_feature(tuv + total_motion, level).FEATURE_COMPS;		

		local_block[k] = t_local;

		m1_local += t_local;
		m2_local += t_local * t_local;
		m1_search += t_search;
		m2_search += t_search * t_search;
		m_cov += t_local * t_search;

		//float2 index_rc = (offs.xy * BLOCK_SIZE + offs.yx + 0.5) / (BLOCK_SIZE * BLOCK_SIZE);
		//float4 wavelet = cos(index_rc.xyxy * PI * float4(1.0, 1.0, 2.0, 2.0)); 

		//float4 wavelet = cos((k + 0.5) / (BLOCK_SIZE * BLOCK_SIZE) * PI * float4(1.0,2.0,4.0,8.0));

		//coeff_local += t_local.x * wavelet;//pow(offs.xy * BLOCK_SIZE + offs.yx, 4);
		//coeff_search += t_search.x * wavelet;//pow(offs.xy * BLOCK_SIZE + offs.yx, 4);
	}

	FEATURE_TYPE cossim = m_cov * rsqrt(m2_local * m2_search);
	float best_sim = get_min_cossim(cossim);

	float best_error = maxc(abs(coeff_local - coeff_search));

	FEATURE_TYPE best_m1 = m1_search;
	FEATURE_TYPE best_m2 = m2_search;

	float phi = radians(360.0 / OCTAVE_SAMPLES);
	float2x2 rotsector = float2x2(cos(phi), -sin(phi), sin(phi), cos(phi));
	float randseed = (((dot(uint2(i.vpos.xy) % 5, float2(1, 5)) * 17) % 25) + 0.5) / 25.0; //prime shuffled, similar spectral properties to bayer but faster to compute and unique values within 5x5
	randseed = frac(randseed + SEARCH_OCTAVES * 0.6180339887498);
	float2 randdir; sincos(randseed * phi, randdir.x, randdir.y);

	int _octaves = SEARCH_OCTAVES;	

	while(_octaves-- > 0)
	{
		_octaves = best_sim < 0.999999 ? _octaves : 0;
		float2 local_motion = 0;

		int _samples = OCTAVE_SAMPLES;
		while(_samples-- > 0)		
		{
			_samples = best_sim < 0.999999 ? _samples : 0;
			randdir = mul(randdir, rotsector);		
			float2 search_offset = randdir * texelsize;
			float2 search_center = i.uv + total_motion + search_offset;			 

			m1_search = 0;
			m2_search = 0;
			m_cov = 0;

			coeff_search = 0;

			[loop]
			for(uint k = 0; k < BLOCK_SIZE * BLOCK_SIZE; k++)
			{
				float2 offs = float2(k / BLOCK_SIZE, k % BLOCK_SIZE);
				FEATURE_TYPE t = get_prev_feature(search_center + float2(k / BLOCK_SIZE, k % BLOCK_SIZE) * texelsize, level).FEATURE_COMPS;

				m1_search += t;
				m2_search += t * t;
				m_cov += local_block[k] * t;

				//coeff_search += t.x * pow(offs.xy * BLOCK_SIZE + offs.yx, 4);
				//float2 index_rc = (offs.xy * BLOCK_SIZE + offs.yx + 0.5) / (BLOCK_SIZE * BLOCK_SIZE);
				//float4 wavelet = cos(index_rc.xyxy * PI * float4(1.0, 1.0, 2.0, 2.0)); 
				//float4 wavelet = cos((k + 0.5) / (BLOCK_SIZE * BLOCK_SIZE) * PI * float4(1.0,2.0,4.0,8.0));				
				//coeff_search += t.x * wavelet;//pow(offs.xy * BLOCK_SIZE + offs.yx, 4);
			}
			cossim = m_cov * rsqrt(m2_local * m2_search);
			float sim = get_min_cossim(cossim);

			//float error = maxc(abs(coeff_local - coeff_search));

			[branch]
			if(sim > best_sim)	
			//if(best_error > error)				
			{
				best_sim = sim;
				//best_error = error;
				local_motion = search_offset;
				best_m1 = m1_search;
	            best_m2 = m2_search;		
			}		
		}
		total_motion += local_motion;
		randdir *= 0.5;
	}

	best_m1 /= BLOCK_SIZE * BLOCK_SIZE;	
	best_m2 /= BLOCK_SIZE * BLOCK_SIZE;
	float variance = dot(1, sqrt(abs(best_m2 - best_m1 * best_m1)));
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
		//float sample_z = get_curr_feature(sample_uv, level).y;
		float4 sample_gbuf = tex2Dlod(sMotionLow, sample_uv, 0);
		float ws = saturate(10 - sample_gbuf.w * 10);
		float wf = saturate(1 - sample_gbuf.b * 128.0);
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4;
		wm *= wm;

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
	const float4 gauss = float4(1, 0.85, 0.65, 0.45);

	float4 gbuffer_sum = 0;
	float wsum = 1e-6;
	int rad = filter_size;

	[loop]for(int x = -rad; x <= rad; x++)
	[loop]for(int y = -rad; y <= rad; y++)
	{
		float2 offs = mul(float2(x * abs(x), y * abs(y)), rotm) * texelsize;
		float2 sample_uv = i.uv + offs;	

		//float sample_z = get_curr_feature(sample_uv, level).y;
		float4 sample_gbuf = tex2Dlod(sMotionLow, sample_uv, 0);
		float2 prev_mv = tex2Dlod(sMotionTexIntermediate0, sample_uv + sample_gbuf.xy, 0).xy;

		float wd = (dot(prev_mv, sample_gbuf.xy) + 1e-7) * rcp(1e-7 + sqrt(dot(prev_mv, prev_mv) * dot(sample_gbuf.xy, sample_gbuf.xy)));
		wd = saturate(wd);
		wd = 1 - wd * wd;

		float ws = saturate(10 - sample_gbuf.w * 10);
		float wf = saturate(1 - sample_gbuf.b * 128.0);
		float wm = dot(sample_gbuf.xy, sample_gbuf.xy) * 4;
		wm *= wm;

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

	return find_best_residual_motion(i, level, prior_motion);	
}

float4 motion_pass_with_temporal_filter(in VSOUT i, sampler sMotionLow, int level, int filter_size)
{
	float4 prior_motion = tex2Dlod(sMotionLow, i.uv, 0) * 0.95;
    if(level < MAX_MIP)
    	prior_motion = atrous_upscale_temporal(i, level, sMotionLow, filter_size);	

	if(level < MIN_MIP)
		return prior_motion;

	return find_best_residual_motion(i, level, prior_motion);	
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(dist * 100));
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

float3 linear_to_ycocg(float3 color)
{
    float Y  = dot(color, float3(0.25, 0.5, 0.25));
    float Co = dot(color, float3(0.5, 0.0, -0.5));
    float Cg = dot(color, float3(-0.25, 0.5, -0.25));
    return float3(Y, Co, Cg);
}

void PSWriteFeature(in VSOUT i, out FEATURE_TYPE o : SV_Target0)
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

#if MATCHING_LAYERS == 0
	o = dot(float3(0.299, 0.587, 0.114), feature_data.rgb);
#elif MATCHING_LAYERS == 1
	o.x = dot(float3(0.299, 0.587, 0.114), feature_data.rgb);
	o.y = feature_data.w;
#else 
	float3 ycocg = linear_to_ycocg(feature_data.rgb);
	o = float4(ycocg.x, feature_data.w, ycocg.yz*0.5+0.5);
#endif
}

void PSMotion6(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate2, 6, 2);}
void PSMotion5(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate6, 5, 2);}
void PSMotion4(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate5, 4, 2);}
void PSMotion3(in VSOUT i, out float4 o : SV_Target0){o = motion_pass_with_temporal_filter(i, sMotionTexIntermediate4, 3, 2);}
void PSMotion2(in VSOUT i, out float4 o : SV_Target0){o = motion_pass(i, sMotionTexIntermediate3, 2, 2);}
void PSMotion1(in VSOUT i, out float4 o : SV_Target0){o = motion_pass(i, sMotionTexIntermediate2, 1, 1);}
void PSMotion0(in VSOUT i, out float4 o : SV_Target0){o = motion_pass(i, sMotionTexIntermediate1, 0, 1);}

void PSOut(in VSOUT i, out float3 o : SV_Target0)
{	
	o = showmotion(-tex2D(sMotionTexIntermediate0, i.uv).xy);
}

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
    pass {VertexShader = MainVS;PixelShader  = PSWriteFeature; RenderTarget = FeaturePyramid; } 

	pass {VertexShader = MainVS;PixelShader = PSMotion6;RenderTarget = MotionTexIntermediate6;}
    pass {VertexShader = MainVS;PixelShader = PSMotion5;RenderTarget = MotionTexIntermediate5;}
    pass {VertexShader = MainVS;PixelShader = PSMotion4;RenderTarget = MotionTexIntermediate4;}
    pass {VertexShader = MainVS;PixelShader = PSMotion3;RenderTarget = MotionTexIntermediate3;}
    pass {VertexShader = MainVS;PixelShader = PSMotion2;RenderTarget = MotionTexIntermediate2;}
    pass {VertexShader = MainVS;PixelShader = PSMotion1;RenderTarget = MotionTexIntermediate1;}
    pass {VertexShader = MainVS;PixelShader = PSMotion0;RenderTarget = MotionTexIntermediate0;}

	pass {VertexShader = MainVS;PixelShader  = PSWriteFeature; RenderTarget = FeaturePyramidPrev; }
#if DEBUG_OUTPUT != 0 //why waste perf for this pass in normal mode
	pass {VertexShader = MainVS;PixelShader  = PSOut;  } 
#endif 
}