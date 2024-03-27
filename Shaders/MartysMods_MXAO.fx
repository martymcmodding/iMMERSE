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

    MXAO v1.1

    Author:         Pascal Gilcher

    More info:      https://martysmods.com
                    https://patreon.com/mcflypg
                    https://github.com/martymcmodding  	

=============================================================================*/

//TODO: fix black lines in bottom and right for DX9 (require threads outside view if not 1:1 mapping)

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef MXAO_AO_TYPE
 #define MXAO_AO_TYPE       0
#endif 

#ifndef MXAO_USE_LAUNCHPAD_NORMALS
 #define MXAO_USE_LAUNCHPAD_NORMALS       0
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform int MXAO_GLOBAL_SAMPLE_QUALITY_PRESET <
	ui_type = "combo";
    ui_label = "Sample Quality";
	ui_items = "Low\0Medium\0High\0Very High\0Ultra\0Extreme\0IDGAF\0";
	ui_tooltip = "Global quality control, main performance knob. Higher radii might require higher quality.";
    ui_category = "Global";
> = 1;

uniform int SHADING_RATE <
	ui_type = "combo";
    ui_label = "Shading Rate";
	ui_items = "Full Rate\0Half Rate\0Quarter Rate\0";
	ui_tooltip = "0: render all pixels each frame\n1: render only 50% of pixels each frame\n2: render only 25% of pixels each frame";
    ui_category = "Global";
> = 1;

uniform float MXAO_SAMPLE_RADIUS <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 10.0;
    ui_label = "Sample Radius";
	ui_tooltip = "Sample radius of MXAO, higher means more large-scale occlusion with less fine-scale details.";  
    ui_category = "Global";      
> = 2.5;

uniform bool MXAO_WORLDSPACE_ENABLE <
    ui_label = "Increase Radius with Distance";
    ui_category = "Global";
> = false;

uniform float MXAO_SSAO_AMOUNT <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 1.0;
    ui_label = "Ambient Occlusion Amount";        
	ui_tooltip = "Intensity of AO effect. Can cause pitch black clipping if set too high.";
    ui_category = "Blending";
> = 0.8;

uniform float MXAO_FADE_DEPTH <
	ui_type = "drag";
    ui_label = "Fade Out Distance";
	ui_min = 0.0; ui_max = 1.0;
	ui_tooltip = "Fadeout distance for MXAO. Higher values show MXAO in farther areas.";
    ui_category = "Blending";
> = 0.25;

uniform int MXAO_FILTER_SIZE <
	ui_type = "slider";
    ui_label = "Filter Quality";
    ui_min = 0; ui_max = 2;	
    ui_category = "Blending";
> = 1;

uniform bool MXAO_DEBUG_VIEW_ENABLE <
    ui_label = "Show Raw AO";
    ui_category = "Debug";
> = false;

#define TOKENIZE(s) #s

uniform int HELP1 <
ui_type = "radio";
    ui_label = " ";
    ui_category = "Preprocessor definition Documentation";
    ui_category_closed = false;
    ui_text = 
            "\n"
            TOKENIZE(MXAO_AO_TYPE)
            ":\n\n0: Ground Truth Ambient Occlusion (high contrast, fast)\n"
                 "1: Solid Angle (smoother, fastest)\n"
                 "2: Visibility Bitmask (DX11+ only, highest quality, slower)\n"
                 "3: Visibility Bitmask w/ Solid Angle (like 2, only smoother)\n"
            "\n"
            TOKENIZE(MXAO_USE_LAUNCHPAD_NORMALS)
            ":\n\n0: Compute normal vectors on the fly (fast)\n"
                 "1: Use normals from iMMERSE Launchpad (far slower)\n"
                 "   This allows to use Launchpad's smooth normals feature.";
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

texture AOTex1 { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F;  };
texture AOTex2 { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F;  };

#if !_COMPUTE_SUPPORTED
texture AOTexRaw { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F;  };
sampler sAOTexRaw { Texture = AOTexRaw;  MinFilter=POINT; MipFilter=POINT; MagFilter=POINT; };
#endif

sampler sAOTex1 { Texture = AOTex1; };
sampler sAOTex2 { Texture = AOTex2; };

#include ".\MartysMods\mmx_global.fxh"
#include ".\MartysMods\mmx_depth.fxh"
#include ".\MartysMods\mmx_math.fxh"
#include ".\MartysMods\mmx_camera.fxh"

#if MXAO_USE_LAUNCHPAD_NORMALS 
 #include ".\MartysMods\mmx_deferred.fxh"
#endif 

//#undef _COMPUTE_SUPPORTED

//integer divide, rounding up
#define CEIL_DIV(num, denom) (((num - 1) / denom) + 1)


#if ((BUFFER_WIDTH/4)*4) == BUFFER_WIDTH
 #define DEINTERLEAVE_HIGH       0
 #define DEINTERLEAVE_TILE_COUNT 4u
#else 
 #define DEINTERLEAVE_HIGH       1
 #define DEINTERLEAVE_TILE_COUNT 5u
#endif

uniform uint FRAMECOUNT < source = "framecount"; >;

#if _COMPUTE_SUPPORTED
storage stAOTex1       { Texture = AOTex1;        };
storage stAOTex2       { Texture = AOTex2;        };

texture3D ZSrc3D 
{ 
    Width = BUFFER_WIDTH/DEINTERLEAVE_TILE_COUNT;   
    Height = BUFFER_HEIGHT/DEINTERLEAVE_TILE_COUNT;   
    Depth = DEINTERLEAVE_TILE_COUNT * DEINTERLEAVE_TILE_COUNT;   
    Format = R16F;
};
sampler3D sZSrc3D { Texture = ZSrc3D; MinFilter=POINT; MipFilter=POINT; MagFilter=POINT;};
storage3D stZSrc3D { Texture = ZSrc3D; };
#else 
texture ZSrc { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F; };
sampler sZSrc { Texture = ZSrc; MinFilter=POINT; MipFilter=POINT; MagFilter=POINT;};
#endif

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

static const uint2 samples_per_preset[7] = 
{
//  slices/steps    preset            samples 
    uint2(2, 2),    //Low             8
    uint2(2, 4),    //Medium          16
    uint2(2, 10),   //High            40
    uint2(3, 12),   //Very High       72
    uint2(4, 14),   //Ultra           112
    uint2(6, 16),   //Extreme         192
    uint2(8, 20)    //IDGAF           320
};

/*=============================================================================
	Functions
=============================================================================*/

float2 pixel_idx_to_uv(float2 pos, float2 texture_size)
{
    float2 inv_texture_size = rcp(texture_size);
    return pos * inv_texture_size + 0.5 * inv_texture_size;
}

bool check_boundaries(uint2 pos, uint2 dest_size)
{
    return pos.x < dest_size.x && pos.y < dest_size.y; //>= because dest size e.g. 1920, pos [0, 1919]
}

uint2 deinterleave_pos(uint2 pos, uint2 tiles, uint2 gridsize)
{
    int2 tilesize = CEIL_DIV(gridsize, tiles); //gridsize / tiles;
    int2 tile_idx    = pos % tiles;
    int2 pos_in_tile = pos / tiles;
    return tile_idx * tilesize + pos_in_tile;
}

uint2 reinterleave_pos(uint2 pos, uint2 tiles, uint2 gridsize)
{
    int2 tilesize = CEIL_DIV(gridsize, tiles); //gridsize / tiles;
    int2 tile_idx    = pos / tilesize;
    int2 pos_in_tile = pos % tilesize;
    return pos_in_tile * tiles + tile_idx;
}

float2 deinterleave_uv(float2 uv)
{
    float2 splituv = uv * DEINTERLEAVE_TILE_COUNT;
    float2 splitoffset = floor(splituv) - DEINTERLEAVE_TILE_COUNT * 0.5 + 0.5;
    splituv = frac(splituv) + splitoffset * BUFFER_PIXEL_SIZE;
    return splituv;
}

float2 reinterleave_uv(float2 uv)
{
    uint2 whichtile = floor(uv / BUFFER_PIXEL_SIZE) % DEINTERLEAVE_TILE_COUNT;
    float2 newuv = uv + whichtile;
    newuv /= DEINTERLEAVE_TILE_COUNT;
    return newuv;
}

float3 get_normals(in float2 uv, out float edge_weight)
{
    float3 delta = float3(BUFFER_PIXEL_SIZE, 0);
    //similar system to Intel ASSAO/AMD CACAO/XeGTAO and friends with improved weighting and less ALU
    float3 center = Camera::uv_to_proj(uv);
    float3 deltaL = Camera::uv_to_proj(uv - delta.xz) - center;
    float3 deltaR = Camera::uv_to_proj(uv + delta.xz) - center;   
    float3 deltaT = Camera::uv_to_proj(uv - delta.zy) - center;
    float3 deltaB = Camera::uv_to_proj(uv + delta.zy) - center;
    
    float4 zdeltaLRTB = abs(float4(deltaL.z, deltaR.z, deltaT.z, deltaB.z));
    float4 w = zdeltaLRTB.xzyw + zdeltaLRTB.zywx;
    w = rcp(0.001 + w * w); //inverse weighting, larger delta -> lesser weight

    edge_weight = saturate(1.0 - dot(w, 1));

#if MXAO_USE_LAUNCHPAD_NORMALS //this is a bit hacky, we need the edge weight for filtering but Launchpad doesn't give them to us, so we compute the data till here and read launchpad normals
    float3 normal = Deferred::get_normals(uv);
#else 

    float3 n0 = cross(deltaT, deltaL);
    float3 n1 = cross(deltaR, deltaT);
    float3 n2 = cross(deltaB, deltaR);
    float3 n3 = cross(deltaL, deltaB);

    float4 finalweight = w * rsqrt(float4(dot(n0, n0), dot(n1, n1), dot(n2, n2), dot(n3, n3)));
    float3 normal = n0 * finalweight.x + n1 * finalweight.y + n2 * finalweight.z + n3 * finalweight.w;
    normal *= rsqrt(dot(normal, normal) + 1e-8);
#endif 
    return normal;  
}

float get_jitter(uint2 p)
{
    uint tiles = DEINTERLEAVE_TILE_COUNT;
    uint jitter_idx = dot(p % tiles, uint2(1, tiles));
    jitter_idx *= DEINTERLEAVE_HIGH ? 17u : 11u;
    return ((jitter_idx % (tiles * tiles)) + 0.5) / (tiles * tiles);
}

float get_fade_factor(float depth)
{
    float fade = saturate(1 - depth * depth); //fixed fade that smoothly goes to 0 at depth = 1
    depth /= MXAO_FADE_DEPTH;
    return fade * saturate(exp2(-depth * depth)); //overlaying regular exponential fade
}

//=============================================================================   
#if _COMPUTE_SUPPORTED
//=============================================================================  

static uint occlusion_bitfield;

void bitfield_init()
{
    occlusion_bitfield = 0xFFFFFFFF;
}

void process_horizons(float2 h)
{
    uint a = uint(h.x * 32);
    uint b = ceil(saturate(h.y - h.x) * 32); //ceil? using half occlusion here, this attenuates effect when an occluder is so far away that can't cover half a sector
    uint occlusion = ((1 << b) - 1) << a;
    occlusion_bitfield &= ~occlusion; //somehow "and" is faster than "or" based occlusion
}

float integrate_sectors()
{
    return saturate(countbits(occlusion_bitfield) / 32.0);
}

//read from deinterleave volume
float read_z(float2 uv, float w)
{
    return tex3Dlod(sZSrc3D, float4(uv, w, 0)).x;
}

bool shading_rate(uint2 tile_idx)
{
    bool skip_pixel = false;
    switch(SHADING_RATE)
    {
        case 1: skip_pixel = ((tile_idx.x + tile_idx.y) & 1) ^ (FRAMECOUNT & 1); break;     
        case 2: skip_pixel = (tile_idx.x & 1 + (tile_idx.y & 1) * 2) ^ (FRAMECOUNT & 3); break; 
    }
    return skip_pixel;
}

//=============================================================================                  
#else //Needs this because DX9 is a jackass and doesn't have bitwise ops... so emulate them with floats
//=============================================================================   

bool bitfield_is_set(float bitfield, int bit)
{
    float state = floor(bitfield * exp2(-bit)); //>>
    return frac(state * 0.5) > 0.25; //& 1
}

void bitfield_set(inout float bitfield, int bit, bool value)
{
    bitfield += exp2(bit) * (value - bitfield_is_set(bitfield, bit));    
}

float bitfield_set_bits(float bitfield, int start, int stride)
{ 
    [loop]
    for(int bit = start; bit < start + stride; bit++)
        bitfield_set(bitfield, bit, 1);       
    return bitfield;
}

static float occlusion_bitfield;

void bitfield_init()
{
    occlusion_bitfield = 0;
}

float integrate_sectors()
{  
    float sum = 0;
    [loop]
    for(int bit = 0; bit < 24; bit++)
        sum += bitfield_is_set(occlusion_bitfield, bit);
    return saturate(1.0 - sum / 25.0);
}
                    
void process_horizons(float2 h)
{
    uint a = floor(h.x * 24);
    uint b = floor(saturate(h.y - h.x) * 25.0); //haven't figured out why this needs to be one more (gives artifacts otherwise) but whatever, somethingsomething float inaccuracy
    occlusion_bitfield = bitfield_set_bits(occlusion_bitfield, a, b);
}

//read from tiled texture
float read_z(float2 uv, float w)
{
    return tex2Dlod(sZSrc, uv, 0).x;
}

bool shading_rate(uint2 tile_idx)
{
    bool skip_pixel = false;
    switch(SHADING_RATE)
    { 
        case 1: skip_pixel = ((tile_idx.x + tile_idx.y) % 2) != (FRAMECOUNT % 2); break;     
        case 2: skip_pixel = (tile_idx.x % 2 + (tile_idx.y % 2) * 2) != (FRAMECOUNT % 4); break;
    }
    return skip_pixel;
}

//=============================================================================   
#endif //_COMPUTE_SUPPORTED
//=============================================================================   

/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
    return o;
}

#if _COMPUTE_SUPPORTED
void Deinterleave3DCS(in CSIN i)
{
    if(!check_boundaries(i.dispatchthreadid.xy * 2, BUFFER_SCREEN_SIZE)) return;

    float2 uv = pixel_idx_to_uv(i.dispatchthreadid.xy * 2, BUFFER_SCREEN_SIZE);
    float2 corrected_uv = Depth::correct_uv(uv); //fixed for lookup 

#if RESHADE_DEPTH_INPUT_IS_UPSIDE_DOWN
    corrected_uv.y -= BUFFER_PIXEL_SIZE.y * 0.5;    //shift upwards since gather looks down and right
    float4 depth_texels = tex2DgatherR(DepthInput, corrected_uv).wzyx;  
#else
    float4 depth_texels = tex2DgatherR(DepthInput, corrected_uv);
#endif

    depth_texels = Depth::linearize(depth_texels);
    depth_texels.x = Camera::depth_to_z(depth_texels.x);
    depth_texels.y = Camera::depth_to_z(depth_texels.y);
    depth_texels.z = Camera::depth_to_z(depth_texels.z);
    depth_texels.w = Camera::depth_to_z(depth_texels.w);

    //offsets for xyzw components
    const uint2 offsets[4] = {uint2(0, 1), uint2(1, 1), uint2(1, 0), uint2(0, 0)};

    [unroll]
    for(uint j = 0; j < 4; j++)
    {
        uint2 screenpos = i.dispatchthreadid.xy * 2 + offsets[j];

        const uint tilecount = DEINTERLEAVE_TILE_COUNT;

        uint3 write_pos;
        write_pos.xy = screenpos / tilecount;
        uint2 tile_idx = screenpos - write_pos.xy * tilecount;
        write_pos.z = tile_idx.x + tile_idx.y * tilecount;

        tex3Dstore(stZSrc3D, write_pos, depth_texels[j]);
    }
}
#else 
void DepthInterleavePS(in VSOUT i, out float o : SV_Target0)
{ 
    float2 get_uv = deinterleave_uv(i.uv);
    o = Camera::depth_to_z(Depth::get_linear_depth(get_uv));
}
#endif

float2 MXAOFused(uint2 screenpos, float4 uv, float depth_layer)
{ 	
    float z = read_z(uv.xy, depth_layer);
    float d = Camera::z_to_depth(z);

    [branch]
    if(get_fade_factor(d) < 0.001) return float2(1, d);

    float3 p = Camera::uv_to_proj(uv.zw, z); 
    float edge_weight;  
    float3 n = get_normals(uv.zw, edge_weight);
    p = p * 0.996;
    float3 v = normalize(-p);  

#if _COMPUTE_SUPPORTED
    static const float4 texture_scale = BUFFER_ASPECT_RATIO.xyxy;
#else
    static const float4 texture_scale = float2(1.0 / DEINTERLEAVE_TILE_COUNT, 1.0).xxyy * BUFFER_ASPECT_RATIO.xyxy;
#endif

    uint slice_count  = samples_per_preset[MXAO_GLOBAL_SAMPLE_QUALITY_PRESET].x;    
    uint sample_count = samples_per_preset[MXAO_GLOBAL_SAMPLE_QUALITY_PRESET].y; 

    float jitter = get_jitter(screenpos);    
    float3 slice_dir = 0; sincos(jitter * PI * (6.0/slice_count), slice_dir.x, slice_dir.y);    
    float2x2 rotslice; sincos(PI / slice_count, rotslice._21, rotslice._11); rotslice._12 = -rotslice._21; rotslice._22 = rotslice._11;    

    float worldspace_radius = MXAO_SAMPLE_RADIUS * 0.5;
    float screenspace_radius = worldspace_radius / p.z * 0.5;

    [flatten]
    if(MXAO_WORLDSPACE_ENABLE)
    {
        screenspace_radius = MXAO_SAMPLE_RADIUS * 0.03;
        worldspace_radius = screenspace_radius * p.z * 2.0;
    }

    float visibility = 0;
    float slicesum = 0;  
    float T = log(1 + worldspace_radius) * 0.3333;//arbitrary thickness that looks good relative to sample radius  

    float falloff_factor = rcp(worldspace_radius);
    falloff_factor *= falloff_factor;

    //terms for the GTAO slice weighting logic, math has been extremely simplified but is
    //entirely unrecognizable now. 26 down to 19 instructions though :yeahboiii:
    float2 vcrossn_xy = float2(v.yz * n.zx - v.zx * n.yz);//cross(v, n).xy;
    float ndotv = dot(n, v);

    while(slice_count-- > 0) //1 less register and a bit faster
    {        
        slice_dir.xy = mul(slice_dir.xy, rotslice);
        float4 scaled_dir = (slice_dir.xy * screenspace_radius).xyxy * texture_scale; 
        
        float sdotv = dot(slice_dir.xy, v.xy);
        float sdotn = dot(slice_dir.xy, n.xy); 
        float ndotns = dot(slice_dir.xy, vcrossn_xy) * rsqrt(saturate(1 - sdotv * sdotv));
     
        float sliceweight = sqrt(saturate(1 - ndotns * ndotns));//length of projected normal on slice
        float cosn = saturate(ndotv * rcp(sliceweight));
        float normal_angle = Math::fast_acos(cosn);
        normal_angle = sdotn < sdotv * ndotv ? -normal_angle : normal_angle;

        float2 maxhorizoncos = sin(normal_angle); maxhorizoncos.y = -maxhorizoncos.y; //cos(normal_angle -+ pi/2)  
        bitfield_init();

        [unroll]
        for(int side = 0; side < 2; side++)
        {            
            maxhorizoncos = maxhorizoncos.yx; //can't trust Vulkan to unroll, so make indices natively addressable for that little more efficiency
            float lowesthorizoncos = maxhorizoncos.x; //much better falloff than original GTAO :)

            [loop]         
            for(int _sample = 0; _sample < sample_count; _sample += 2)
            {
                float2 s = (_sample + float2(0, 1) + jitter) / sample_count; s *= s;  

                float4 tap_uv[2] = {uv + s.x * scaled_dir, 
                                    uv + s.y * scaled_dir};

                if(!all(saturate(tap_uv[1].zw - tap_uv[1].zw * tap_uv[1].zw))) break;                       

                float2 zz; //https://developer.nvidia.com/blog/the-peak-performance-analysis-method-for-optimizing-any-gpu-workload/
                zz.x = read_z(tap_uv[0].xy, depth_layer); 
                zz.y = read_z(tap_uv[1].xy, depth_layer);               

                [unroll] //less VGPR by splitting
                for(uint pair = 0; pair < 2; pair++)
                {
                    float3 deltavec = Camera::uv_to_proj(tap_uv[pair].zw, zz[pair]) - p;
#if MXAO_AO_TYPE < 2
                    float ddotd = dot(deltavec, deltavec);    
                    float samplehorizoncos = dot(deltavec, v) * rsqrt(ddotd);
                    float falloff = rcp(1 + ddotd * falloff_factor);
                    samplehorizoncos = lerp(lowesthorizoncos, samplehorizoncos, falloff);
                    maxhorizoncos.x = max(maxhorizoncos.x, samplehorizoncos);  
#else      
                    float ddotv = dot(deltavec, v);
                    float ddotd = dot(deltavec, deltavec);
                    float2 h_frontback = float2(ddotv, ddotv - T) * rsqrt(float2(ddotd, ddotd - 2 * T * ddotv + T * T));

                    h_frontback = Math::fast_acos(h_frontback);
                    h_frontback = side ? h_frontback : -h_frontback.yx;//flip sign and sort in the same cmov, efficiency baby!
                    h_frontback = saturate((h_frontback + normal_angle) / PI + 0.5);
#if MXAO_AO_TYPE == 2
                    //this almost perfectly approximates inverse transform sampling for cosine lobe
                    h_frontback = h_frontback * h_frontback * (3.0 - 2.0 * h_frontback); 
#endif                   
                    process_horizons(h_frontback);
#endif  //MXAO_AO_TYPE        
                }              
            }
            scaled_dir = -scaled_dir; //unroll kills that :)                                  
        }
#if MXAO_AO_TYPE == 0
        float2 max_horizon_angle = Math::fast_acos(maxhorizoncos);
        float2 h = float2(-max_horizon_angle.x, max_horizon_angle.y); //already clamped at init
        visibility += dot(cosn + 2.0 * h * sin(normal_angle) - cos(2.0 * h - normal_angle), sliceweight);
        slicesum++;
#elif MXAO_AO_TYPE == 1
        float2 max_horizon_angle = Math::fast_acos(maxhorizoncos);
        visibility += dot(max_horizon_angle, sliceweight);
        slicesum += sliceweight;
#else
        visibility += integrate_sectors() * sliceweight;
        slicesum += sliceweight;         
#endif
    }

#if MXAO_AO_TYPE == 0
    visibility /= slicesum * 4;
#elif MXAO_AO_TYPE == 1
    visibility /= slicesum * PI;
#else 
    visibility /= slicesum;
#endif
    return float2(saturate(visibility), edge_weight > 0.5 ? -d : d);//store depth negated for pixels with low normal confidence to drive the filter
}

#if _COMPUTE_SUPPORTED
void OcclusionWrap3DCS(in CSIN i)
{    
    const uint tilecount = DEINTERLEAVE_TILE_COUNT;
    const uint2 tilesize = BUFFER_SCREEN_SIZE / tilecount;    

    uint2 tile_idx;
    tile_idx.y = i.dispatchthreadid.z / tilecount;
    tile_idx.x = i.dispatchthreadid.z - tile_idx.y * tilecount; 

    if(!check_boundaries(i.dispatchthreadid.xy, tilesize) || shading_rate(tile_idx)) return;

    uint2 screen_pos = i.dispatchthreadid.xy * tilecount + tile_idx;
    float4 uv;
    uv.xy = pixel_idx_to_uv(i.dispatchthreadid.xy, tilesize);
    uv.zw = pixel_idx_to_uv(screen_pos, BUFFER_SCREEN_SIZE);    

    float depth_layer = i.dispatchthreadid.z * rcp(tilecount * tilecount);    
    float2 ao_and_guide = MXAOFused(screen_pos, uv, depth_layer);

    tex2Dstore(stAOTex1, screen_pos, ao_and_guide.xyyy);
}
#else 
void OcclusionWrap1PS(in VSOUT i, out float2 o : SV_Target0) //writes to AOTex2
{
    uint2 dispatchthreadid = floor(i.vpos.xy);
    uint2 write_pos = reinterleave_pos(dispatchthreadid, DEINTERLEAVE_TILE_COUNT, BUFFER_SCREEN_SIZE);
    uint2 tile_idx = dispatchthreadid / CEIL_DIV(BUFFER_SCREEN_SIZE, DEINTERLEAVE_TILE_COUNT);

    if(shading_rate(tile_idx)) discard;   

    float4 uv;
    uv.xy = pixel_idx_to_uv(dispatchthreadid, BUFFER_SCREEN_SIZE);
    //uv.zw = pixel_idx_to_uv(write_pos, BUFFER_SCREEN_SIZE);
    uv.zw = deinterleave_uv(uv.xy); //no idea why _this_ works but the other doesn't but that's just DX9 being a jackass I guess
    o = MXAOFused(write_pos, uv, 0.0);
}

void OcclusionWrap2PS(in VSOUT i, out float2 o : SV_Target0) 
{
	uint2 dispatchthreadid = floor(i.vpos.xy);
    uint2 read_pos = deinterleave_pos(dispatchthreadid, DEINTERLEAVE_TILE_COUNT, BUFFER_SCREEN_SIZE);
    uint2 tile_idx = dispatchthreadid / CEIL_DIV(BUFFER_SCREEN_SIZE, DEINTERLEAVE_TILE_COUNT);
    
    //need to do it here again because the AO pass writes to AOTex2, which is also intermediate for filter
    //so we only take the new texels and transfer them to AOTex1, so AOTex1 contains unfiltered, reconstructed data
    if(shading_rate(tile_idx)) discard;
    o = tex2Dfetch(sAOTexRaw, read_pos).xy;    
}
#endif
/*
float2 filter_crossbilateral(float2 uv, sampler sAO, int iter)
{
    float2 center = tex2Dlod(sAO, uv, 0).xy;
    float2 axis = float2(iter, !iter) * BUFFER_PIXEL_SIZE;

    int k = 5;
    float sigma = (k + 1.0) * 0.5;
    float falloff = rcp(2 * sigma * sigma);

    float4 mv = float4(center.y, center.y * center.y, center.x, center.x * center.y);
    float wsum = 1;

    [unroll]
    for(int j = 1; j < k; j++)
    {      
        float2 tap = tex2Dlod(sAO, uv + axis * j, 0).xy;
        float w = exp2(-j*j*falloff);
        mv += float4(tap.y, tap.y * tap.y, tap.x, tap.x * tap.y) * w;     
        tap = tex2Dlod(sAO, uv - axis * j, 0).xy;   
        mv += float4(tap.y, tap.y * tap.y, tap.x, tap.x * tap.y) * w;
        wsum += 2.0 * w;
    }

    mv /= wsum;

    float b = (mv.w - mv.x * mv.z) / max(mv.y - mv.x * mv.x, exp2(-28));
    float a = mv.z - b * mv.x;
    return float2(saturate(b * center.y + a), center.y);
}
*/
//todo add direct sample method for DX9
float2 filter(float2 uv, sampler sAO, int iter)
{ 
    float g = tex2D(sAO, uv).y;
    bool blurry = g < 0;
    float flip = iter ? -1 : 1;

    float4 ao, depth, mv;
    ao = tex2DgatherR(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(-0.5, -0.5));
    depth = abs(tex2DgatherG(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(-0.5, -0.5))); //abs because sign flip for edge pixels!
    mv = float4(dot(depth, 1), dot(depth, depth), dot(ao, 1), dot(ao, depth));

    ao = tex2DgatherR(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(1.5, -0.5));
    depth = abs(tex2DgatherG(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(1.5, -0.5)));
    mv += float4(dot(depth, 1), dot(depth, depth), dot(ao, 1), dot(ao, depth));

    ao = tex2DgatherR(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(-0.5, 1.5));
    depth = abs(tex2DgatherG(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(-0.5, 1.5)));
    mv += float4(dot(depth, 1), dot(depth, depth), dot(ao, 1), dot(ao, depth));
    
    ao = tex2DgatherR(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(1.5, 1.5));
    depth = abs(tex2DgatherG(sAO, uv + flip * BUFFER_PIXEL_SIZE * float2(1.5, 1.5)));
    mv += float4(dot(depth, 1), dot(depth, depth), dot(ao, 1), dot(ao, depth));

    mv /= 16.0;

    float b = (mv.w - mv.x * mv.z) / max(mv.y - mv.x * mv.x, exp2(blurry ? -12 : -30));
    float a = mv.z - b * mv.x;
    return float2(saturate(b * abs(g) + a), g); //abs because sign flip for edge pixels!
}

void Filter1PS(in VSOUT i, out float2 o : SV_Target0)
{    
    if(MXAO_FILTER_SIZE < 2) discard;
    o = filter(i.uv, sAOTex1, 0);
}

void Filter2PS(in VSOUT i, out float3 o : SV_Target0)
{    
    float2 t;
    [branch]
    if(MXAO_FILTER_SIZE == 2)
        t = filter(i.uv, sAOTex2, 1);
    else if(MXAO_FILTER_SIZE == 1)
        t = filter(i.uv, sAOTex1, 1);
    else 
        t = tex2Dlod(sAOTex1, i.uv, 0).xy;

    float mxao = t.x, d = abs(t.y);  //abs because sign flip for edge pixels!

    mxao = lerp(1, mxao, saturate(MXAO_SSAO_AMOUNT)); 
    if(MXAO_SSAO_AMOUNT > 1) mxao = lerp(mxao, mxao * mxao, saturate(MXAO_SSAO_AMOUNT - 1)); //if someone _MUST_ use a higher intensity, switch to gamma
    mxao = lerp(1, mxao, get_fade_factor(d));

    float3 color = tex2D(ColorInput, i.uv).rgb;

    color *= color;
    color = color * rcp(1.1 - color);
    color *= mxao;
    color = 1.1 * color * rcp(color + 1.0); 
    color = sqrt(color); 

    o = MXAO_DEBUG_VIEW_ENABLE ? mxao : color;
}

/*=============================================================================
	Techniques
=============================================================================*/

technique MartysMods_MXAO
<
    ui_label = "iMMERSE: MXAO";
    ui_tooltip =        
        "                              MartysMods - MXAO                               \n"
        "                   MartysMods Epic ReShade Effects (iMMERSE)                  \n"
        "______________________________________________________________________________\n"
        "\n"

        "MXAO is a high quality, high performance Screen-Space Ambient Occlusion (SSAO)\n"
        "effect which accurately simulates diffuse shadows in dark corners and crevices\n"
        "\n"
        "\n"
        "Visit https://martysmods.com for more information.                            \n"
        "\n"       
        "______________________________________________________________________________";
>
{ 
#if _COMPUTE_SUPPORTED
    pass 
    { 
        ComputeShader = Deinterleave3DCS<32, 32>;
        DispatchSizeX = CEIL_DIV(BUFFER_WIDTH, 64); 
        DispatchSizeY = CEIL_DIV(BUFFER_HEIGHT, 64);
    }
    pass 
    { 
        ComputeShader = OcclusionWrap3DCS<16, 16, 1>;
        DispatchSizeX = CEIL_DIV((BUFFER_WIDTH/DEINTERLEAVE_TILE_COUNT), 16); 
        DispatchSizeY = CEIL_DIV((BUFFER_HEIGHT/DEINTERLEAVE_TILE_COUNT), 16);
        DispatchSizeZ = DEINTERLEAVE_TILE_COUNT * DEINTERLEAVE_TILE_COUNT;
    }
#else 
    pass { VertexShader = MainVS; PixelShader = DepthInterleavePS; RenderTarget = ZSrc; }
    pass { VertexShader = MainVS; PixelShader = OcclusionWrap1PS;  RenderTarget = AOTexRaw; }
    pass { VertexShader = MainVS; PixelShader = OcclusionWrap2PS;  RenderTarget = AOTex1; }
#endif
    pass { VertexShader = MainVS; PixelShader = Filter1PS; RenderTarget = AOTex2; }
    pass { VertexShader = MainVS; PixelShader = Filter2PS; }
}