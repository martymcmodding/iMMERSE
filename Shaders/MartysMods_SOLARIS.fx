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

    Solaris Bloom

    Author:         Pascal Gilcher

    More info:      https://martysmods.com
                    https://patreon.com/mcflypg
                    https://github.com/martymcmodding  	

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef ENABLE_SOLARIS_REGRADE_PARITY
 #define ENABLE_SOLARIS_REGRADE_PARITY                  0   //[0 or 1]    If enabled, ReGrade takes HDR input from SOLARIS as color buffer instead. This allows HDR exposure, bloom and color grading to work nondestructively
#endif

#ifndef SOLARIS_PERF_MODE
 #define SOLARIS_PERF_MODE                              0
#endif 

#ifndef SOLARIS_ARTISTIC_MODE
 #define SOLARIS_ARTISTIC_MODE                          0
#endif 

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float HDR_EXPOSURE <
	ui_type = "drag";
	ui_min = -5.0; ui_max = 5.0;
	ui_label = "Log Exposure Bias";
> = 0.0;

uniform float HDR_WHITEPOINT <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 12.0;
    ui_label = "Log HDR Whitepoint";
> = 7.0;

uniform float HDR_BLOOM_INT <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 1.0;
	ui_label = "Bloom Intensity";
> = 0.3;

uniform float HDR_BLOOM_RADIUS <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 1.0;
	ui_label = "Bloom Radius";
> = 1.0;

#if SOLARIS_ARTISTIC_MODE != 0
uniform int HDR_BLOOM_BLEND <
	ui_type = "combo";
    ui_label = "Blend Mode";
	ui_items = "Energy Conserving (physically accurate)\0HDR Drama\0Orton\0Dreamy\0Depth Blend\0Screen\0";
> = 0;
#else 
uniform float HDR_BLOOM_HAZYNESS <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 1.0;
	ui_label = "Bloom Hazyness";
> = 0.9;
#endif

uniform bool BLOOM_HQ_DOWNSAMPLING <
    ui_label = "High Resolution Input";
> = false;

uniform bool BLOOM_DEPTH_MASK <
    ui_label = "Mask by Depth";
> = true;

uniform float BLOOM_DEPTH_MASK_STRENGTH <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 1.0;
	ui_label = "Depth Mask Strength";
> = 0.5;

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

texture ColorInputTex : COLOR;
texture DepthInputTex : DEPTH;
sampler ColorInput 	{ Texture = ColorInputTex;};
sampler ColorInputPoint 	{ Texture = ColorInputTex; MinFilter = POINT; MipFilter = POINT; MagFilter = POINT; };
sampler DepthInput 	{ Texture = DepthInputTex;};

#if ENABLE_SOLARIS_REGRADE_PARITY != 0
texture2D ColorInputHDRTex			    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
sampler2D sColorInputHDR			    { Texture = ColorInputHDRTex;  };
#endif

#include ".\MartysMods\mmx_global.fxh"
#include ".\MartysMods\mmx_depth.fxh"
#include ".\MartysMods\mmx_texture.fxh"

#if SOLARIS_PERF_MODE > 2
 #undef SOLARIS_PERF_MODE
 #define SOLARIS_PERF_MODE 2
#endif
#if SOLARIS_PERF_MODE < 0
 #undef SOLARIS_PERF_MODE
 #define SOLARIS_PERF_MODE 0
#endif

//max screen width allowed, everything above will get downscaled
#define LOG_MAX_RES     (11  - SOLARIS_PERF_MODE)   //0: 2048, 1: 1024, 2: 512
#define BASE_SIZE_X     (1 << LOG_MAX_RES)
#define BASE_SIZE_Y     (BASE_SIZE_X * BUFFER_HEIGHT) / BUFFER_WIDTH

texture SolarisBloomOut { Width = BASE_SIZE_X; Height = BASE_SIZE_Y; Format = RGBA16F;  }; 
sampler sSolarisBloomOut { Texture = SolarisBloomOut; };

texture2D SolarisBloom0 { Width = BASE_SIZE_X >> 0; Height = BASE_SIZE_Y >> 0; Format = RGBA16F;  }; sampler2D sSolarisBloom0 { Texture = SolarisBloom0;  };  
texture2D SolarisBloom1 { Width = BASE_SIZE_X >> 1; Height = BASE_SIZE_Y >> 1; Format = RGBA16F;  }; sampler2D sSolarisBloom1 { Texture = SolarisBloom1;  };  
texture2D SolarisBloom2 { Width = BASE_SIZE_X >> 2; Height = BASE_SIZE_Y >> 2; Format = RGBA16F;  }; sampler2D sSolarisBloom2 { Texture = SolarisBloom2;  };  
texture2D SolarisBloom3 { Width = BASE_SIZE_X >> 3; Height = BASE_SIZE_Y >> 3; Format = RGBA16F;  }; sampler2D sSolarisBloom3 { Texture = SolarisBloom3;  };  
texture2D SolarisBloom4 { Width = BASE_SIZE_X >> 4; Height = BASE_SIZE_Y >> 4; Format = RGBA16F;  }; sampler2D sSolarisBloom4 { Texture = SolarisBloom4;  };  
texture2D SolarisBloom5 { Width = BASE_SIZE_X >> 5; Height = BASE_SIZE_Y >> 5; Format = RGBA16F;  }; sampler2D sSolarisBloom5 { Texture = SolarisBloom5;  };  
#if SOLARIS_PERF_MODE <= 1
texture2D SolarisBloom6 { Width = BASE_SIZE_X >> 6; Height = BASE_SIZE_Y >> 6; Format = RGBA16F;  }; sampler2D sSolarisBloom6 { Texture = SolarisBloom6;  }; 
#endif
#if SOLARIS_PERF_MODE <= 0
texture2D SolarisBloom7 { Width = BASE_SIZE_X >> 7; Height = BASE_SIZE_Y >> 7; Format = RGBA16F;  }; sampler2D sSolarisBloom7 { Texture = SolarisBloom7;  }; 
#endif

texture2D BlueNoiseDitherTex     < source = "iMMERSE_bluenoise_opt.png"; > { Width = 256; Height = 256; Format = RGBA8; };
sampler2D	sBlueNoiseDitherTex   { Texture = BlueNoiseDitherTex; AddressU = WRAP; AddressV = WRAP; };

static const float3 kernel_downsample[12] = 
{
    float3(-1, -1, 4.0),        
    float3( 1, -1, 4.0),
    float3(-1,  1, 4.0),
    float3( 1,  1, 4.0),
    float3( 0, -2, 2.0),
    float3(-2,  0, 2.0),
    float3( 2,  0, 2.0),
    float3( 0,  2, 2.0),             
    float3( 2,  2, 1.0),
    float3(-2,  2, 1.0),
    float3( 2, -2, 1.0),
    float3(-2, -2, 1.0)
};

static const float3 kernel_upsample[8] = 
{
    float3(-0.707, -0.707, 0.153873),
    float3(     0,     -1, 0.153873),
    float3( 0.707, -0.707, 0.153873),
    float3(     1,      0, 0.153873),
    float3( 0.707,  0.707, 0.153873),
    float3(     0,      1, 0.153873),
    float3(-0.707,  0.707, 0.153873),
    float3(    -1,      0, 0.153873)
};

/*=============================================================================
	Functions
=============================================================================*/

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

float3 sdr_to_hdr(float3 c, float w)
{ 
    c = cone_overlap(c);
    c *= c;
    c /= exp2(w) * (1 - c) + 1;
    c *= 128.0; 
    return c;   
}

float3 hdr_to_sdr(float3 c, float w)
{    
    c /= 128.0;
    c = c * (1 + exp2(-w)) / (c + exp2(-w));
    c = sqrt(c);   
    c = cone_overlap_inv(c);
    return c;   
}

//depth masking works on relative values so even though
//this scales alpha as well, it does not interfere
float layerweight(float i)
{
    return exp2(-i * 0.5 * (1 - sqrt(HDR_BLOOM_RADIUS)));
}

/*=============================================================================
	Shader Entry Points
=============================================================================*/

struct VSOUT
{
	float4 vpos : SV_Position;
    float2 uv   : UV;    
};

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv);
    return o;
}

void PrepassPS(in VSOUT i, out float4 o : SV_Target0) 
{
    float2 target_texelsize = rcp(float2(BASE_SIZE_X, BASE_SIZE_Y));
    float2 uv_lo = i.uv - target_texelsize * 0.5;
    float2 uv_hi = i.uv + target_texelsize * 0.5;

    float2 texel_boundary_lo = uv_lo * BUFFER_SCREEN_SIZE;
    float2 texel_boundary_hi = uv_hi * BUFFER_SCREEN_SIZE;

    int2 texel_lo = floor(texel_boundary_lo);
    int2 texel_hi = ceil(texel_boundary_hi);

    o = 0;

    //out of all variants to branch this, this seems to be the fastest one... weird.

    [branch]
    if(BLOOM_HQ_DOWNSAMPLING)
    {
        float wsum = 1e-7; 
        [loop]for(int x = texel_lo.x; x <= texel_hi.x; x += 2)
        [loop]for(int y = texel_lo.y; y <= texel_hi.y; y += 2)
        {
            float4 weights = saturate(texel_boundary_hi.xxyy - float4(x, x + 1, y, y + 1)) - saturate(texel_boundary_lo.xxyy - float4(x, x + 1, y, y + 1));
            weights = weights.xyxy * weights.zzww;
            float quad_weight = sum(weights);        
            float2 subpixel_pos = float2(x, y) + (weights.yz + weights.ww) / quad_weight;   
            float2 subpixel_uv = (subpixel_pos + 0.5) * BUFFER_PIXEL_SIZE;   

            float4 qR = tex2DgatherR(ColorInput, subpixel_uv).wzxy;
            float4 qG = tex2DgatherG(ColorInput, subpixel_uv).wzxy;
            float4 qB = tex2DgatherB(ColorInput, subpixel_uv).wzxy;
            o.rgb += sdr_to_hdr(float3(qR.x, qG.x, qB.x), HDR_WHITEPOINT) * weights.x;
            o.rgb += sdr_to_hdr(float3(qR.y, qG.y, qB.y), HDR_WHITEPOINT) * weights.y;
            o.rgb += sdr_to_hdr(float3(qR.z, qG.z, qB.z), HDR_WHITEPOINT) * weights.z;
            o.rgb += sdr_to_hdr(float3(qR.w, qG.w, qB.w), HDR_WHITEPOINT) * weights.w;  

            [branch] 
            if(BLOOM_DEPTH_MASK)
            {            
                o.w += Depth::get_linear_depth(subpixel_uv) * quad_weight;                           
            }

            wsum += quad_weight;
        }
        o /= wsum; 
    }
    else 
    {
        float wsum = 1e-7; 
        [loop]for(int x = texel_lo.x; x <= texel_hi.x; x += 2)
        [loop]for(int y = texel_lo.y; y <= texel_hi.y; y += 2)
        {
            float4 weights = saturate(texel_boundary_hi.xxyy - float4(x, x + 1, y, y + 1)) - saturate(texel_boundary_lo.xxyy - float4(x, x + 1, y, y + 1));
            weights = weights.xyxy * weights.zzww;
            float quad_weight = sum(weights);        
            float2 subpixel_pos = float2(x, y) + (weights.yz + weights.ww) / quad_weight;   
            float2 subpixel_uv = (subpixel_pos + 0.5) * BUFFER_PIXEL_SIZE;   

            o.rgb += tex2Dlod(ColorInput, subpixel_uv, 0).rgb * quad_weight;

            [branch] 
            if(BLOOM_DEPTH_MASK)
            {            
                o.w += Depth::get_linear_depth(subpixel_uv) * quad_weight;                           
            }
            wsum += quad_weight;
        }
        o /= wsum;
        o.rgb = sdr_to_hdr(o.rgb, HDR_WHITEPOINT); 
    }   
      
    o.rgb *= exp2(HDR_EXPOSURE);
}

/*=============================================================================
	Downsampling
=============================================================================*/

float4 downsample(sampler s, float2 uv, const int pass_id)
{    
    static const float2 texelsize = rcp(int2(BASE_SIZE_X, BASE_SIZE_Y) / int(exp2(pass_id)));

    float4 sum = tex2Dlod(s, uv, 0);
    float centerdepth = sum.w;

    float wsum = 4.0;
    sum *= wsum; 

    float2 wmadd = float2(-rcp(centerdepth) * BLOOM_DEPTH_MASK_STRENGTH * 0.5, BLOOM_DEPTH_MASK_STRENGTH * 0.5);  

    [unroll]
    for(int j = 0; j < 12; j++)
    {
        float4 tap = tex2Dlod(s, uv + kernel_downsample[j].xy * texelsize, 0);
        float w = BLOOM_DEPTH_MASK ? saturate(exp2(tap.w * wmadd.x + wmadd.y)) : 1;
        w *= kernel_downsample[j].z;
        sum += tap * w;
        wsum += w;
    }

    return sum / wsum * layerweight(pass_id);  
}

void Downsample01PS(  in VSOUT i, out float4 o : SV_Target0) { o = downsample(sSolarisBloom0, i.uv, 0); }
void Downsample12PS(  in VSOUT i, out float4 o : SV_Target0) { o = downsample(sSolarisBloom1, i.uv, 1); }
void Downsample23PS(  in VSOUT i, out float4 o : SV_Target0) { o = downsample(sSolarisBloom2, i.uv, 2); }
void Downsample34PS(  in VSOUT i, out float4 o : SV_Target0) { o = downsample(sSolarisBloom3, i.uv, 3); }
void Downsample45PS(  in VSOUT i, out float4 o : SV_Target0) { o = downsample(sSolarisBloom4, i.uv, 4); }
#if SOLARIS_PERF_MODE <= 1
void Downsample56PS(  in VSOUT i, out float4 o : SV_Target0) { o = downsample(sSolarisBloom5, i.uv, 5); }
#if SOLARIS_PERF_MODE <= 0
void Downsample67PS(  in VSOUT i, out float4 o : SV_Target0) { o = downsample(sSolarisBloom6, i.uv, 6); }
#endif
#endif

/*=============================================================================
	Upsampling
=============================================================================*/

float4 upsample(sampler2D s, float2 uv, const int pass_id)
{   
    static const float2 texelsize = rcp(int2(BASE_SIZE_X, BASE_SIZE_Y) / int(exp2(pass_id))) * 1.5;
    
    float4 sum = tex2Dlod(s, uv, 0);
    float centerdepth = sum.w;   

    float wsum = 1;
    sum *= wsum;    
    float2 wmadd = float2(-rcp(centerdepth) * BLOOM_DEPTH_MASK_STRENGTH * 0.5, BLOOM_DEPTH_MASK_STRENGTH * 0.5);  

    [unroll]    
    for(int j = 0; j < 8; j++)
    {
        float4 tap = tex2Dlod(s, uv + kernel_upsample[j].xy * texelsize, 0);
        float w = BLOOM_DEPTH_MASK ? saturate(exp2(tap.w * wmadd.x + wmadd.y)) : 1;
        w *= kernel_upsample[j].z;
        sum += tap * w;
        wsum += w;
    }

    return sum / wsum;
}

#if SOLARIS_PERF_MODE <= 1
#if SOLARIS_PERF_MODE <= 0
void Upsample76PS(   in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom7, i.uv, 7); }
#endif
void Upsample65PS(   in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom6, i.uv, 6); }
#endif
void Upsample54PS(   in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom5, i.uv, 5); }
void Upsample43PS(   in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom4, i.uv, 4); }
void Upsample32PS(   in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom3, i.uv, 3); }
void Upsample21PS(   in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom2, i.uv, 2); }
void Upsample10PS(   in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom1, i.uv, 1); }
void UpsampleFinalPS(in VSOUT i, out float4 o : SV_Target0) { o = upsample(sSolarisBloom0, i.uv, 0); }

/*=============================================================================
	Blend
=============================================================================*/

struct VSOUT_Blend
{
	float4 vpos : SV_Position;
    float2 uv   : UV; 
    float  w    : WEIGHT;   
};

VSOUT_Blend BlendVS(in uint id : SV_VertexID)
{
    VSOUT_Blend o;
    FullscreenTriangleVS(id, o.vpos, o.uv);

    float l1 = layerweight(1);
	float l2 = layerweight(2);
	float l3 = layerweight(3);
	float l4 = layerweight(4);
    float l5 = layerweight(5);
    float l6 = layerweight(6);
    float l7 = layerweight(7);
#if SOLARIS_PERF_MODE <= 0
    o.w = 1 + (l1 * (1 + l2 * (1 + l3 * (1 + l4 * (1 + l5 * (1 + l6 * (1 + l7)))))));
#elif SOLARIS_PERF_MODE <= 1
    o.w = 1 + (l1 * (1 + l2 * (1 + l3 * (1 + l4 * (1 + l5 * (1 + l6))))));
#else 
    o.w = 1 + (l1 * (1 + l2 * (1 + l3 * (1 + l4 * (1 + l5)))));
#endif
    o.w = 1/o.w;

    return o;
}

void BlendPS(in VSOUT_Blend i, out float3 o : SV_Target0)
{
    float4 bloomt = Texture::sample2D_bspline_auto(sSolarisBloomOut, i.uv); 
    float3 bloom = bloomt.rgb * i.w;  

    [branch]
    if(BLOOM_DEPTH_MASK)
    {
        float depth = Depth::get_linear_depth(i.uv);
        float bloomdepth = bloomt.w;
        float depthw = saturate(exp2(-(bloomdepth - depth) / depth * BLOOM_DEPTH_MASK_STRENGTH * 0.5));
        depthw = lerp(depthw, 1, 0.3333); //only here, lift up a little to avoid weird looking bloom reappearing inside close objects
        bloom *= depthw;
    } 

    bloom *= exp2(-HDR_WHITEPOINT * 0.25); //visually normalize so observed bloom intensity is agnostic of whitepoint setting    
    float3 col = tex2D(ColorInput, i.uv).rgb;

    col = sdr_to_hdr(col, HDR_WHITEPOINT);    
    col *= exp2(HDR_EXPOSURE); 

#if SOLARIS_ARTISTIC_MODE != 0
    switch(HDR_BLOOM_BLEND)
    {
        case 0:
        {
            col = lerp(col, bloom, saturate(HDR_BLOOM_INT * HDR_BLOOM_INT) * 0.5);
            //col += col * bloom.rgb * HDR_BLOOM_INT * HDR_BLOOM_INT * 128.0; //not energy conserving lmao but try to explain to users why the screen gets darker with high bloom intensity
            col = hdr_to_sdr(col, HDR_WHITEPOINT);
            break;
        }        
        case 1:
        {            
            bloom = hdr_to_sdr(bloom, HDR_WHITEPOINT);        
            col = hdr_to_sdr(col, HDR_WHITEPOINT);

            float3 blended = (1 - 2 * bloom) * col * col + 2 * col * bloom;
            col = lerp(col, blended, HDR_BLOOM_INT * HDR_BLOOM_INT);
            break;          
        }
        case 2:
        {
            col = hdr_to_sdr(col, HDR_WHITEPOINT);
            bloom = hdr_to_sdr(bloom, HDR_WHITEPOINT);           
            float3 blended =  (1 - 2 * col) * bloom * bloom + 2 * col * bloom;
            col = lerp(col, blended, HDR_BLOOM_INT * HDR_BLOOM_INT);
            break;          
        }
        case 3:
        {
            col = col * exp2((log2(1 + bloom * HDR_BLOOM_INT * 6)));
            col = hdr_to_sdr(col, HDR_WHITEPOINT);
            break;          
        }
        case 4: 
        {
            bloom = hdr_to_sdr(bloom, HDR_WHITEPOINT);        
            col = hdr_to_sdr(col, HDR_WHITEPOINT);

            float3 blendedA = (1 - 2 * bloom) * col * col + 2 * col * bloom;
            float3 blendedB = (1 - 2 * col) * bloom * bloom + 2 * col * bloom;   

            float z = Depth::get_linear_depth(i.uv);

            float k = 0.1;
            float depthblend = (1 + k) * z / (k + z);

            float3 blended = lerp(blendedA, blendedB, depthblend); 
            col = lerp(col, blended, HDR_BLOOM_INT * HDR_BLOOM_INT);

            break;
        }
        case 5: 
        {
            bloom = hdr_to_sdr(bloom, HDR_WHITEPOINT);        
            col = hdr_to_sdr(col, HDR_WHITEPOINT);

            col = 1- (1 - col) * (1 - bloom * HDR_BLOOM_INT * HDR_BLOOM_INT);
            break;
        }
    }  
#else 
    col += lerp(col, 0.05, HDR_BLOOM_HAZYNESS * 0.5 + 0.5) * bloom.rgb * HDR_BLOOM_INT * HDR_BLOOM_INT * 128.0;
    col = hdr_to_sdr(col, HDR_WHITEPOINT);
#endif

#if !ENABLE_SOLARIS_REGRADE_PARITY 
    float3 dither = tex2Dfetch(sBlueNoiseDitherTex, int2(i.vpos.xy) % 256).xyz;
    dither = dither - 0.5;
    dither *= 0.999;
    dither *= exp2(-8.0);
    col = saturate(col + dither);  
#endif
    o = col;
}

/*=============================================================================
	Techniques
=============================================================================*/

technique MartysMods_SOLARIS
<
    ui_label = "iMMERSE: Solaris";
    ui_tooltip =       
        "                               MartysMods - Solaris                               \n"
        "                     MartysMods Epic ReShade Effects (iMMERSE)                    \n"
        "               Official versions only via https://patreon.com/mcflypg             \n"
        "__________________________________________________________________________________\n"
        "\n"
        "SOLARIS is a novel approach for bloom and exposure control.                       \n"
        "It can be paired with REGRADE, an all-purpose color grading effect.\n"
        "Make sure to place this effect right before ReGrade.                              \n"
        "\n"
        "\n"
        "Visit https://martysmods.com for more information.                                \n"
        "\n"       
        "__________________________________________________________________________________\n";
>
{
    pass{VertexShader = MainVS;   PixelShader = PrepassPS; RenderTarget = SolarisBloom0; }
    
    pass{VertexShader = MainVS;   PixelShader = Downsample01PS;   RenderTarget = SolarisBloom1; }
    pass{VertexShader = MainVS;   PixelShader = Downsample12PS;   RenderTarget = SolarisBloom2; }
    pass{VertexShader = MainVS;   PixelShader = Downsample23PS;   RenderTarget = SolarisBloom3; }
    pass{VertexShader = MainVS;   PixelShader = Downsample34PS;   RenderTarget = SolarisBloom4; }  
    pass{VertexShader = MainVS;   PixelShader = Downsample45PS;   RenderTarget = SolarisBloom5; }   
#if SOLARIS_PERF_MODE <= 1   
    pass{VertexShader = MainVS;   PixelShader = Downsample56PS;   RenderTarget = SolarisBloom6; }
    
#if SOLARIS_PERF_MODE <= 0
    pass{VertexShader = MainVS;   PixelShader = Downsample67PS;   RenderTarget = SolarisBloom7; }
    pass{VertexShader = MainVS;   PixelShader = Upsample76PS;   RenderTarget = SolarisBloom6; BlendEnable = true;BlendOp = ADD;SrcBlend = ONE;DestBlend = ONE; }
#endif
    pass{VertexShader = MainVS;   PixelShader = Upsample65PS;   RenderTarget = SolarisBloom5; BlendEnable = true;BlendOp = ADD;SrcBlend = ONE;DestBlend = ONE; }
#endif
    pass{VertexShader = MainVS;   PixelShader = Upsample54PS;   RenderTarget = SolarisBloom4; BlendEnable = true;BlendOp = ADD;SrcBlend = ONE;DestBlend = ONE; }
    pass{VertexShader = MainVS;   PixelShader = Upsample43PS;   RenderTarget = SolarisBloom3; BlendEnable = true;BlendOp = ADD;SrcBlend = ONE;DestBlend = ONE; }
    pass{VertexShader = MainVS;   PixelShader = Upsample32PS;   RenderTarget = SolarisBloom2; BlendEnable = true;BlendOp = ADD;SrcBlend = ONE;DestBlend = ONE; }
    pass{VertexShader = MainVS;   PixelShader = Upsample21PS;   RenderTarget = SolarisBloom1; BlendEnable = true;BlendOp = ADD;SrcBlend = ONE;DestBlend = ONE; }
    pass{VertexShader = MainVS;   PixelShader = Upsample10PS;   RenderTarget = SolarisBloom0; BlendEnable = true;BlendOp = ADD;SrcBlend = ONE;DestBlend = ONE; }    
    pass{VertexShader = MainVS;   PixelShader = UpsampleFinalPS; RenderTarget = SolarisBloomOut; } 
#if ENABLE_SOLARIS_REGRADE_PARITY != 0
    pass{VertexShader = BlendVS;   PixelShader = BlendPS; RenderTarget = ColorInputHDRTex; } 
#else 
    pass{VertexShader = BlendVS;   PixelShader = BlendPS;    }
#endif   
}
