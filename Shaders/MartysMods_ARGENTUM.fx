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

    Argentum Film Grain Shader

    Author:         Pascal Gilcher

    More info:      https://martysmods.com
                    https://patreon.com/mcflypg
                    https://github.com/martymcmodding  	

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float GRAIN_SIZE <
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
> = 0.3;

uniform int NUM_GRAINS <
	ui_type = "slider";
    ui_min = 1; ui_max = 128;
> = 10;

uniform int FILM_TYPE <
    ui_type = "combo";
    ui_items = "Monochrome\0Color\0";
> = 0;

#define FILM_TYPE_MONOCHROME 0
#define FILM_TYPE_COLOR      1

uniform bool USE_FILM_CURVE <
    ui_label = "Use Filmic Response Curve";
    ui_category = "Film Curve";
> = false;

uniform float FILM_CURVE_GAMMA <
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
    ui_label = "Film Gamma";
    ui_category = "Film Curve";
> = 0.0;

uniform float FILM_CURVE_TOE <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_label = "Film Shadow Emphasis";
    ui_category = "Film Curve";
> = 0.0;

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

/*=============================================================================
	Textures, Samplers, Globals, Structs
=============================================================================*/

//do NOT change anything here. "hurr durr I changed this and now it works"
//you ARE breaking things down the line, if the shader does not work without changes
//here, it's by design.

texture ColorInputTex : COLOR;
sampler ColorInput 	{ Texture = ColorInputTex; };

#include ".\MartysMods\mmx_global.fxh"
#include ".\MartysMods\mmx_math.fxh"

#define CEIL_DIV(num, denom) ((((num) - 1) / (denom)) + 1)

struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

/*=============================================================================
	Functions
=============================================================================*/

uint lowbias32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

#define to_linear(x)    ((x)*0.283799*((2.52405+(x))*(x)))
#define from_linear(x)  (1.14374*(-0.126893*(x)+sqrt((x))))

float get_grey_value(int2 p)
{
    float3 color = tex2Dfetch(ColorInput, p).rgb;
    color = to_linear(color);
    return dot(color, float3(0.299, 0.587, 0.114));
}

//hand crafted response curve that mimics exposure adjustment pre-tonemap with toe
float3 filmic_curve(float3 x, float toe_strength, float gamma)
{
    //input is [-1, 1]
    gamma = gamma < 0.0 ? gamma * 0.5 : gamma * 6.0;

    x = saturate(x);
    float3 toe = saturate(1 - x);
    toe *= toe;//2
    toe *= toe;//4
    toe *= toe;//16    
    x = saturate(x - toe_strength * toe);
    float3 gx = x * gamma;
    return (gx + x) / (gx + 1);
}

float get_adjusted_grey_value(int2 p)
{
    float3 color = tex2Dfetch(ColorInput, p).rgb;
    color = dot(color, float3(0.299, 0.587, 0.114));
    return filmic_curve(color, FILM_CURVE_TOE, FILM_CURVE_GAMMA).x;
}

float4 next_rand(inout uint rng)
{
    float4 rand01;
    //need 16 bit rand here to avoid quantization of the output (N values for rand seed == N possible binomial outcomes)
    rng = lowbias32(rng);      
    rand01.xy = frac(((rng >> uint2(0, 16)) & 0xFFFF) * rcp(0xFFFF));
    rng = lowbias32(rng);      
    rand01.zw = frac(((rng >> uint2(0, 16)) & 0xFFFF) * rcp(0xFFFF));
    return rand01;
}

float next_rand_single(inout uint rng)
{   
    rng = lowbias32(rng);      
    return float(rng) * exp2(-32.0);
}

/*=============================================================================
	Shader Entry Points
=============================================================================*/

struct CSIN 
{
    uint3 groupthreadid     : SV_GroupThreadID;         //XYZ idx of thread inside group
    uint3 groupid           : SV_GroupID;               //XYZ idx of group inside dispatch
    uint3 dispatchthreadid  : SV_DispatchThreadID;      //XYZ idx of thread inside dispatch
    uint threadid           : SV_GroupIndex;            //flattened idx of thread inside group
};


texture GrainResultTex { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;  };
sampler sGrainResultTex { Texture = GrainResultTex; MinFilter=POINT; MipFilter=POINT; MagFilter=POINT;};
storage stGrainResultTex       { Texture = GrainResultTex;        };


#define GRP_SIZE            16
#define FIXED_POINT_SCALE   4096 //atomic add on floats no worky

groupshared uint grain_sum_tgsm[(GRP_SIZE+4) * (GRP_SIZE+4)];

void FilmGrainGroundTruthCS(in CSIN i)
{   
    uint idx = i.threadid;
    while(idx < (GRP_SIZE+4) * (GRP_SIZE+4))
    {
        grain_sum_tgsm[idx] = 0;
        idx += GRP_SIZE * GRP_SIZE;
    } 

    barrier(); 
    
    //rename threads to 1 pixel additional size
    idx = i.threadid;
    const uint block_size = (GRP_SIZE + 2);
    while(idx < block_size * block_size)
    {
        int2 grain_sector = int2(idx % block_size, idx / block_size);
        int2 grain_sector_screen = int2(i.groupid.xy) * GRP_SIZE + grain_sector - 1;

        float grey = get_grey_value(grain_sector_screen);

        uint rng = grain_sector_screen.x + grain_sector_screen.y * 3776;
        float4 grain_contributions_ABCD = 0; 

        for(int g = 0; g < NUM_GRAINS; g++) 
        {
            float4 rand01 = next_rand(rng);
            float2 grain_center = rand01.xy;
            float grain_size    = GRAIN_SIZE + GRAIN_SIZE * 0.25 * (rand01.z - 0.5);

            bool2 LorR_TorB = step(0.5, grain_center);
            float2 overlap_with_neighbour = max(0, grain_size * 0.5  - lerp(grain_center, 1.0.xx - grain_center, LorR_TorB));
            float2 overlap_percentage = saturate(overlap_with_neighbour / (1e-7 + grain_size));

            float weight_center = (1 - overlap_percentage.x) * (1 - overlap_percentage.y);
            float weight_horizo = overlap_percentage.x * (1 - overlap_percentage.y);
            float weight_vertic = (1 - overlap_percentage.x) * overlap_percentage.y;
            float weight_diagon = overlap_percentage.x * overlap_percentage.y;

            float is_activated = step(rand01.w, grey);

            int2 neighbour_horizo = int2(grain_sector.x + (LorR_TorB.x ? 1 : -1), grain_sector.y);
            int2 neighbour_vertic = int2(grain_sector.x,                          grain_sector.y + (LorR_TorB.y ? 1 : -1));
            int2 neighbour_diagon = int2(grain_sector.x + (LorR_TorB.x ? 1 : -1), grain_sector.y + (LorR_TorB.y ? 1 : -1));

            atomicAdd(grain_sum_tgsm[grain_sector.x + grain_sector.y * (GRP_SIZE + 2)], uint(FIXED_POINT_SCALE * is_activated * weight_center));    

            if(all(neighbour_horizo >= 0) && all(neighbour_horizo < block_size.xx))
                atomicAdd(grain_sum_tgsm[neighbour_horizo.x + neighbour_horizo.y * (GRP_SIZE + 2)], uint(FIXED_POINT_SCALE * is_activated * weight_horizo));        
            if(all(neighbour_vertic >= 0) && all(neighbour_vertic < block_size.xx))
                atomicAdd(grain_sum_tgsm[neighbour_vertic.x + neighbour_vertic.y * (GRP_SIZE + 2)], uint(FIXED_POINT_SCALE * is_activated * weight_vertic));   
            if(all(neighbour_diagon >= 0) && all(neighbour_diagon < block_size.xx))
                atomicAdd(grain_sum_tgsm[neighbour_diagon.x + neighbour_diagon.y * (GRP_SIZE + 2)], uint(FIXED_POINT_SCALE * is_activated * weight_diagon));
        }
        idx += GRP_SIZE * GRP_SIZE;
    }

    barrier();

    int2 fetch_coord = i.groupthreadid.xy + 1; //group is centered in processed block
    int flat_idx = fetch_coord.x + fetch_coord.y * (GRP_SIZE + 2);

    float final_grey = (float(grain_sum_tgsm[flat_idx]) / FIXED_POINT_SCALE) / NUM_GRAINS;

    final_grey = from_linear(final_grey.xxx).x;

    if(all(i.dispatchthreadid.xy < BUFFER_SCREEN_SIZE))
        tex2Dstore(stGrainResultTex, i.dispatchthreadid.xy, final_grey.xxxx);
}

//to create a LUT that does the same, we need to know how a bernoulli process on each pixel
//influences the neighbourhood. Each final pixel is the result of lots of random trials 
//in the 3x3 neighbourhood around it. 

//Or differently expressed, each pixel influences the 3x3 neighbourhood in a specific way
//This means, simulating N grains on a single pixel gives us 9 weights, one for each pixel

#define NUM_COLORS 256
#define NUM_TRIALS 1024

texture GrainMonteCarloTex { Width = NUM_COLORS;   Height = NUM_TRIALS;   Format = RGBA32F;  };
sampler sGrainMonteCarloTex { Texture = GrainMonteCarloTex; MinFilter=POINT; MipFilter=POINT; MagFilter=POINT;};
storage stGrainMonteCarloTex       { Texture = GrainMonteCarloTex;        };  

void FilmGrainLUTGenCS(in CSIN i)
{ 
    if(tempF3.x > 0) return;
    float grey = float(i.dispatchthreadid.x) / (NUM_COLORS - 1.0);
    grey = to_linear(grey).x;
    uint rng = i.dispatchthreadid.y + 1337;

    float weights[9] = 
    {
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    };

    for(int g = 0; g < NUM_GRAINS; g++) 
    {
        float4 rand01 = next_rand(rng);
        float2 grain_center = rand01.xy;
        float grain_size    = GRAIN_SIZE + GRAIN_SIZE * 0.25 * (rand01.z - 0.5);

        bool2 LorR_TorB = step(0.5, grain_center);
        float2 overlap_with_neighbour = max(0, grain_size * 0.5  - lerp(grain_center, 1.0.xx - grain_center, LorR_TorB));
        float2 overlap_percentage = saturate(overlap_with_neighbour / (1e-7 + grain_size));

        float weight_center = (1 - overlap_percentage.x) * (1 - overlap_percentage.y);
        float weight_horizo = overlap_percentage.x * (1 - overlap_percentage.y);
        float weight_vertic = (1 - overlap_percentage.x) * overlap_percentage.y;
        float weight_diagon = overlap_percentage.x * overlap_percentage.y;

        float wsum = weight_center + weight_horizo + weight_vertic + weight_diagon;

        float is_activated = step(rand01.w, grey);
        is_activated /= wsum;

        weights[4] += is_activated * weight_center;
        weights[4 + (LorR_TorB.x ? 1 : -1)] += is_activated * weight_horizo;
        weights[4 + (LorR_TorB.y ? 3 : -3)] += is_activated * weight_vertic;
        weights[4 + (LorR_TorB.x ? 1 : -1) + (LorR_TorB.y ? 3 : -3)] += is_activated * weight_diagon;
    }

    for(int j = 0; j < 9; j++) weights[j] = saturate(weights[j] / NUM_GRAINS);

    uint4 packed_output;
    packed_output.x = uint(weights[0] * 1023.99) | (uint(weights[1] * 1023.99) << 10) | (uint(weights[2] * 1023.99) << 20);
    packed_output.y = uint(weights[3] * 1023.99) | (uint(weights[4] * 1023.99) << 10) | (uint(weights[5] * 1023.99) << 20);
    packed_output.z = uint(weights[6] * 1023.99) | (uint(weights[7] * 1023.99) << 10) | (uint(weights[8] * 1023.99) << 20);
    packed_output.w = 1;
    tex2Dstore(stGrainMonteCarloTex, i.dispatchthreadid.xy, asfloat(packed_output));
}


VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); //use original fullscreen triangle VS
    return o;
}

void ApplyGroundTruthPS(in VSOUT i, out float3 o : SV_Target0)
{  
    o = tex2D(sGrainResultTex, i.uv).x;
}

void MainPS(in VSOUT i, out float3 o : SV_Target0)
{  
    //xy offset, channel, encoding offset
    //this reads the offsets to the center texel from the POV of the neighbour
    const int4 contribs[9] = 
    {
        int4(-1, -1, 2, 20),
        int4( 0, -1, 2, 10),
        int4( 1, -1, 2,  0),
        int4(-1,  0, 1, 20),
        int4( 0,  0, 1, 10),
        int4( 1,  0, 1,  0),
        int4(-1,  1, 0, 20),
        int4( 0,  1, 0, 10),
        int4( 1,  1, 0,  0)
    };

    o = 0;

    [branch]
    if(FILM_TYPE == FILM_TYPE_COLOR)
    {
        [unroll]
        for(int j = 0; j < 9; j++)
        {
            int2 p = int2(i.vpos.xy) + contribs[j].xy;
            float3 color = tex2Dfetch(ColorInput, p).rgb;
            color = min(NUM_COLORS - 1, floor(NUM_COLORS * color));

            uint seed = lowbias32(lowbias32(p.y) + p.x);
            uint trial = seed % NUM_TRIALS;    
            
                o.r += float((asuint(tex2Dfetch(sGrainMonteCarloTex, uint2(color.r, trial))[contribs[j].z]) >> contribs[j].w) & 0x3FF) / 1023.0; 
                trial = lowbias32(seed) % NUM_TRIALS;   
                o.g += float((asuint(tex2Dfetch(sGrainMonteCarloTex, uint2(color.g, trial))[contribs[j].z]) >> contribs[j].w) & 0x3FF) / 1023.0;    
                trial = lowbias32(seed) % NUM_TRIALS;   
                o.b += float((asuint(tex2Dfetch(sGrainMonteCarloTex, uint2(color.b, trial))[contribs[j].z]) >> contribs[j].w) & 0x3FF) / 1023.0;    
               
        }

      
        o = saturate(o);
        o = from_linear(o);             
    }
    else 
    {
        [unroll]
        for(int j = 0; j < 9; j++)
        {
            int2 p = int2(i.vpos.xy) + contribs[j].xy;
            float3 color = tex2Dfetch(ColorInput, p).rgb;

            float grey = from_linear(dot(to_linear(color), float3(0.2126729, 0.7151522, 0.072175)));
            grey = floor(NUM_COLORS * 0.999 * grey);

            uint seed = lowbias32(lowbias32(p.y) + p.x);
            uint trial = seed % NUM_TRIALS;
            o += float((asuint(tex2Dfetch(sGrainMonteCarloTex, uint2(grey, trial))[contribs[j].z]) >> contribs[j].w) & 0x3FF) / 1023.0; 
        }

        o = saturate(o);
        o = from_linear(o);   
    }    
}

texture PoissonLookupTex            { Width = NUM_COLORS;   Height = NUM_TRIALS;   Format = RGBA8;  };
sampler sPoissonLookupTex           { Texture = PoissonLookupTex; };

void PoissonLUTPS(in VSOUT i, out float4 o : SV_Target0)
{ 
    float p = uint(i.vpos.x) / (NUM_COLORS - 1.0);
    p = to_linear(p);
    uint rng = uint(i.vpos.y) + 1337;

    o = 0;    
    [loop]for(int g = 0; g < NUM_GRAINS; g++) 
    {
        o += step(next_rand(rng), p);
    }
        
    o /= NUM_GRAINS;
}

float4 hash42(float2 p)
{
	float4 p4 = frac(p.xyxy * float4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return frac((p4.xxyz+p4.yzzw)*p4.zywx);
}

float2 boxmuller(float2 u)
{
    float2 g; sincos(u.x * TAU, g.x, g.y);
    return g * sqrt(max(0, -2.0 * log(u.y + 1e-6)));
}

void ApplyPoissonPS(in VSOUT i, out float3 o : SV_Target0)
{  
    uint2 p = uint2(i.vpos.xy); 
    o = 0;

    float2 gaussian = float2(1, 0.5 * GRAIN_SIZE);
    float sigma = rsqrt(NUM_GRAINS);
     
    float wsum = 0;
    [loop]for(int x = -1; x <= 1; x++)
    [loop]for(int y = -1; y <= 1; y++)
    {
        uint2 tp = p + int2(x, y);
#if _COMPUTE_SUPPORTED
        uint rng = lowbias32(lowbias32(tp.y) + tp.x);
        float4 rand01 = next_rand(rng);
#else 
        float4 rand01 = hash42(tp);
#endif
        float3 tcol = tex2Dfetch(ColorInput, tp).rgb;
        float3 poisson = 0;

        [branch]
        if(FILM_TYPE == FILM_TYPE_COLOR)
        {
           poisson.x = tex2Dlod(sPoissonLookupTex, float2(tcol.x, rand01.x), 0).x; 
           poisson.y = tex2Dlod(sPoissonLookupTex, float2(tcol.y, rand01.x), 0).y; 
           poisson.z = tex2Dlod(sPoissonLookupTex, float2(tcol.z, rand01.x), 0).z; 
        }
        else 
        {
            float tgrey = from_linear(dot(to_linear(tcol), float3(0.2126729, 0.7151522, 0.072175)));
            poisson = tex2Dlod(sPoissonLookupTex, float2(tgrey, rand01.x), 0).x;
        }   
        
        //random displacement to approximate average displacement of grains (gets lower as grains increase, until it converges to a regular lowpass)
        float2 offs = float2(x, y) + boxmuller(rand01.zw) * sigma;
        float w = exp(-dot(offs, offs));   
        //lowpass weight    
        w *= gaussian[abs(x)] * gaussian[abs(y)];

        o += poisson * w;
        wsum += w;
    }

    o /= wsum;
    o = saturate(o);
    o = from_linear(o);
}

/*=============================================================================
	Techniques
=============================================================================*/

technique MartyMods_Argentum_GroundTruth
{    
    pass 
    { 
        ComputeShader = FilmGrainGroundTruthCS<16, 16>;
        DispatchSizeX = CEIL_DIV(BUFFER_WIDTH, 16); 
        DispatchSizeY = CEIL_DIV(BUFFER_HEIGHT, 16);
    }
    pass
	{
		VertexShader = MainVS;
		PixelShader  = ApplyGroundTruthPS;  
	}      
}

technique MartyMods_Argentum
{    
    pass 
    { 
        ComputeShader = FilmGrainLUTGenCS<16, 16>;
        DispatchSizeX = CEIL_DIV(NUM_COLORS, 16); 
        DispatchSizeY = CEIL_DIV(NUM_TRIALS, 16);
    }
    
    pass
	{
		VertexShader = MainVS;
		PixelShader  = MainPS;  
	}      
}

technique MartyMods_Argentum_Lowpass
{ 
    pass
	{
		VertexShader = MainVS;
		PixelShader  = PoissonLUTPS;  
        RenderTarget = PoissonLookupTex;
	}      
    pass
	{
		VertexShader = MainVS;
		PixelShader  = ApplyPoissonPS;  
	} 
}