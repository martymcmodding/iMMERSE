/*=============================================================================
                                                           
 888b     d888 8888888888 88888888888 8888888888 .d88888b.  8888888b.  
 8888b   d8888 888            888     888       d88P" "Y88b 888   Y88b 
 88888b.d88888 888            888     888       888     888 888    888 
 888Y88888P888 8888888        888     8888888   888     888 888   d88P 
 888 Y888P 888 888            888     888       888     888 8888888P"  
 888  Y8P  888 888            888     888       888     888 888 T88b   
 888   "   888 888            888     888       Y88b. .d88P 888  T88b  
 888       888 8888888888     888     8888888888 "Y88888P"  888   T88b 

  Marty's Extra Effects for ReShade                                                          
                                                                            
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

    Film Grain

    Author:         Pascal Gilcher

    More info:      https://martysmods.com
                    https://patreon.com/mcflypg
                    https://github.com/martymcmodding  	

=============================================================================*/

/*
    TODO: backport to DX9, iCDF formulation doesn't need compute shaders inherently   
*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform int GRAIN_TYPE <
    ui_type = "combo";
    ui_label = "Type";
    ui_items = "Analog Film Grain\0Digital ISO Noise\0";
    ui_category = "Global";
> = 0;

#define GRAIN_TYPE_ANALOG        0
#define GRAIN_TYPE_DIGITAL       1

uniform int FILM_MODE <
    ui_type = "combo";
    ui_label = "Film Mode";
    ui_items = "Monochrome\0Color\0";
    ui_category = "Global";
> = 0;

#define FILM_MODE_MONOCHROME 0
#define FILM_MODE_COLOR      1

uniform int GRAIN_FPS <
    ui_type = "combo";
    ui_label = "Animation Speed";
    ui_items = "Static\0"
               "10 FPS\0"  
               "25 FPS\0" 
               "30 FPS\0" 
               "60 FPS\0" 
               "Native\0";
    ui_tooltip = "Modulate grain at different framerates to reduce excessive flicker.\n"
                 "This may affect brightness slightly, depending on display response rates.";
    ui_category = "Global";
> = 0;

uniform float GRAIN_INTENSITY_ANALOG < 
    ui_label = "Intensity"; 
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_category = "Parameters for Analog Film Grain";
     ui_tooltip = "Analog film grain simulates a specific amount of grains per pixel\n"
                  "Lower intensity means more grains, which average to a more uniform pixel value.";
> = 0.3;

uniform float GRAIN_DISPERSION_ANALOG < 
    ui_label = "Dispersion"; 
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_category = "Parameters for Analog Film Grain";
    ui_tooltip = "At 0, every pixel simulates the same number of grain particles, yielding uniform noise intensity.\n"
                  "Higher values make the dye cloud sizes uneven, causing a spikier noise profile.";
> = 0.5;

uniform float GRAIN_SIZE <
    ui_type = "drag";
    ui_label = "Halide Crystal Size";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_tooltip = "Analog photo paper contains randomly distributed silver halide crystals,\n"
                 "whose dye clouds can overlap pixel boundaries when scanned digitally.\n\n" 
                 "This shader applies a slight diffusion to reproduce this, larger grains diffuse more.";
    ui_category = "Parameters for Analog Film Grain";
> = 0.3;

uniform float FILM_CURVE_GAMMA <
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
    ui_label = "Film Shoulder";
    ui_category = "Parameters for Analog Film Grain";
    ui_tooltip = "Simple filmic tone curve, baked into the process. Higher values make the image brighter\n";
> = 0.0;

uniform float FILM_CURVE_TOE <
    ui_type = "drag";
    ui_min = -1.0; ui_max = 1.0;
    ui_label = "Film Toe";
    ui_category = "Parameters for Analog Film Grain";
    ui_tooltip = "Simple filmic tone curve, baked into the process. Controls the brightness of the dark regions.\n";
> = 0.0;

uniform float GRAIN_INTENSITY_DIGITAL < 
    ui_label = "Intensity"; 
    ui_type = "drag";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_category = "Parameters for ISO Noise";
> = 0.85;

uniform float GRAIN_SAT <
    ui_type = "drag";
    ui_label = "Noise Saturation";
    ui_min = 0.0;
    ui_max = 1.0;
    ui_category = "Parameters for ISO Noise";
> = 1.0;

uniform bool GRAIN_USE_BAYER <
    ui_label = "Bayer Matrix RGB Weighting";
    ui_tooltip = "Camera Sensors allocate twice as much subpixel area to green,\n"
                 "thus reducing the noise sigma by sqrt(2) for this channel.   \n"
                 "This causes the grain to adopt a pink hue in dark areas      \n\n"
                 "This feature is inactive when using monochrome grain.";
    ui_category = "Parameters for ISO Noise";
> = true;

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
sampler ColorInput 	{ Texture = ColorInputTex; };

texture2D GrainIntermediateTex  < pooled = true; > { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;  };
sampler2D sGrainIntermediateTex                    { Texture = GrainIntermediateTex; };
storage2D stGrainIntermediateTex                { Texture = GrainIntermediateTex; };

#define TEXTURE_RES 1024 //allow more than one seed per pixel
#define GREY_LEVELS 256

texture2D FilmGrainInverseCDFTex            { Width = TEXTURE_RES;   Height = GREY_LEVELS;   Format = R8;  };
sampler2D sFilmGrainInverseCDFTex           { Texture = FilmGrainInverseCDFTex; };
storage2D stFilmGrainInverseCDFTex          { Texture = FilmGrainInverseCDFTex; };

texture1D FilmGrainStateTex                  { Width = 1; Format = R32U;  };
sampler1D<uint> sFilmGrainStateTex           { Texture = FilmGrainStateTex; };
storage1D<uint> stFilmGrainStateTex          { Texture = FilmGrainStateTex; };

uniform uint   FRAMECOUNT < source = "framecount"; >;
uniform float  TIMER < source = "timer"; >;

#include ".\MartysMods\mmx_global.fxh"
#include ".\MartysMods\mmx_math.fxh"
#include ".\MartysMods\mmx_hash.fxh"
#include ".\MartysMods\mmx_bxdf.fxh"
#include ".\MartysMods\mmx_sfc.fxh"

#if !_COMPUTE_SUPPORTED
 #error "DirectX 9 not supported, please use a wrapper."
 #error "See https://guides.martysmods.com/additionalguides/apiwrappers/dxvk for more information."
#endif

struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

struct CSIN 
{
    uint3 groupthreadid     : SV_GroupThreadID;         
    uint3 groupid           : SV_GroupID;            
    uint3 dispatchthreadid  : SV_DispatchThreadID;     
    uint threadid           : SV_GroupIndex;
};

/*=============================================================================
	Functions
=============================================================================*/

#define WHITE_POINT 15.0

float3 to_hdr(float3 c)
{
    float w = 1 + rcp(1e-6 + WHITE_POINT); 
    c = c / (w - c);    
    return c;
}
float3 from_hdr(float3 c)
{
    float w = 1 + rcp(1e-6 + WHITE_POINT);      
    c = w * c * rcp(1 + c);
    return c;
}

#define to_linear(x)    ((x)*0.283799*((2.52405+(x))*(x)))
#define from_linear(x)  (1.14374*(-0.126893*(x)+sqrt((x))))

//hand crafted response curve that mimics exposure adjustment pre-tonemap with toe
float3 filmic_curve(float3 x, float toe_strength, float gamma)
{
    //input is [-1, 1]
    gamma = gamma < 0.0 ? gamma * 0.5 : gamma * 6.0;

    x = saturate(x);
    float3 toe = saturate(1 - x);
    toe *= toe;//2
    toe *= toe;//4  
    x = saturate(x + x * toe_strength * toe);
    float3 gx = x * gamma;
    return (gx + x) / (gx + 1);
}


float4 next_rand_lq(inout uint rng)
{
    //need higher precision than 4x8b
    float4 res;
    res.xy = Hash::next2D(rng);
    res.zw = Hash::next2D(rng);
    return res;    
}

//grain intensity is more intuitive, however halide crystal count is what we need for the simulation
//we simulate up to 128 grains per pixel, lerping to original color for grain intensity < 0.5
//using sqrt for GUI control to make the perceived intensity proportional to slider value
uint grain_intensity_to_halide_count()
{
    return uint(1 + 255 * saturate(2.0 -(1-(1-GRAIN_INTENSITY_ANALOG)*(1-GRAIN_INTENSITY_ANALOG)) * 2.0));
}

float grain_intensity_to_blend()
{
    return saturate((1-(1-GRAIN_INTENSITY_ANALOG)*(1-GRAIN_INTENSITY_ANALOG)) * 2.0);
}

/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); //use original fullscreen triangle VS
    return o;
}

void TrackStateCS(in CSIN i)
{    
    uint prev_state = tex1Dfetch(stFilmGrainStateTex, 0);
    //only register states that would affect the LUT being required
    uint curr_state_hash = uint(GRAIN_TYPE);
    Hash::hash_combine(curr_state_hash, asuint(GRAIN_INTENSITY_ANALOG));
    Hash::hash_combine(curr_state_hash, asuint(GRAIN_DISPERSION_ANALOG));
    Hash::hash_combine(curr_state_hash, asuint(FILM_CURVE_GAMMA));
    Hash::hash_combine(curr_state_hash, asuint(FILM_CURVE_TOE));

    uint m = ~(1 << 31);

    curr_state_hash &= m;
    uint res = (((prev_state & m) != curr_state_hash) ? ~m : 0) | curr_state_hash;
    tex1Dstore(stFilmGrainStateTex, 0, res);
}

bool regenerate_icdf()
{
    return tex1Dfetch(sFilmGrainStateTex, 0) >> 31;
}

//this is relatively inefficient with 1 thread per row but I haven't found a good way to handle varying N
void BuildBinomialiCDFCS(in CSIN i)
{
    if(!regenerate_icdf()) return;

    float p = i.dispatchthreadid.y / float(GREY_LEVELS-1);

    //apply this here since 50% grey resulting in 50% of black pixels perceptually doesn't look like
    //50% grey anymore, so at some place we need to skew the curve to handle the gamma. I do this here.   
    p = filmic_curve(p.xxx, FILM_CURVE_TOE, FILM_CURVE_GAMMA).x;  
    p = to_linear(p);

    float rand01 = i.dispatchthreadid.x / float(TEXTURE_RES - 1);
    rand01       = 0.5 - cos(PI * rand01) * 0.5;   //nonlinear map to increase resolution on extremes

    uint N = grain_intensity_to_halide_count();

    if(GRAIN_DISPERSION_ANALOG > 0)
    {
        //need something completely uncorrelated to the original rand01 seed
        //only jitter along X pos such that noise remains more stable when not animated
        float graincountrng = Hash::uhash(i.dispatchthreadid.x) * exp2(-32.0); 
        float k = saturate(GRAIN_DISPERSION_ANALOG) * 10.0;
        N = clamp(ceil(N * exp2(k * graincountrng) / exp2(k)), 1, N);
    }
   
    //future me, don't optimize this, it needs to be this order
    if(i.dispatchthreadid.y == 0)               { tex2Dstore(stFilmGrainInverseCDFTex, i.dispatchthreadid.xy, 0); return; }
    if(i.dispatchthreadid.y == (GREY_LEVELS-1)) { tex2Dstore(stFilmGrainInverseCDFTex, i.dispatchthreadid.xy, 1); return; }
    if(i.dispatchthreadid.x == 0)               { tex2Dstore(stFilmGrainInverseCDFTex, i.dispatchthreadid.xy, 0); return; }
    if(i.dispatchthreadid.x == (TEXTURE_RES-1)) { tex2Dstore(stFilmGrainInverseCDFTex, i.dispatchthreadid.xy, 1); return; }

    float log_p   = log(p);
    float log_1mp = log(1 - p);

    uint3 bounds = uint3(0, N, 0);

    [unroll]
    for(uint j = 0; j < 16; ++j)
    {
        bounds.z = (bounds.x + bounds.y) / 2;

        //compute CDF at mid by summing PMF from 0 to mid
        float cdf       = 0;
        float log_binom = 0;

        [loop]
        for(uint k = 0; k <= bounds.z; ++k)
        {
            if(k > 0) log_binom += log(N - k + 1) - log(k);          
            cdf += exp(log_binom + k * log_p + (N - k) * log_1mp);
        }
        bounds.xy = cdf <= rand01 ? int2(bounds.z + 1, bounds.y) : int2(bounds.x, bounds.z);
    }

    float res = bounds.x / float(N);    
    tex2Dstore(stFilmGrainInverseCDFTex, i.dispatchthreadid.xy, res);
}


float sample_binomial_icdf(float p, float rand01)
{
    float2 uv;
    uv.x = rand01;
    uv.x = 1 - acos(rand01 * 2 - 1) / PI;
    uv.x = lerp(0.5 / TEXTURE_RES, 1.0 - 0.5 / TEXTURE_RES, saturate(uv.x));
    uv.x = clamp(uv.x, 1.5 / TEXTURE_RES, 1.0 - 1.5 / TEXTURE_RES);
    uv.y = (floor(p * GREY_LEVELS) + 0.5) / GREY_LEVELS;
    //return tex2Dlod(sFilmGrainInverseCDFTexPoint, uv, 0).x; //more correct but also produces much worse image histograms, worse for screenshot compression.
    return tex2Dlod(sFilmGrainInverseCDFTex, uv, 0).x;
}

uint get_seed(uint2 p)
{
    uint seed = Hash::uhash(Hash::uhash(p.x) ^ p.y); 

    float timer_s = TIMER * 0.001;    
    switch(GRAIN_FPS)
    {
        case 0: break;
        case 1: timer_s *= 10.0; seed += uint(timer_s); break;
        case 2: timer_s *= 25.0; seed += uint(timer_s); break;
        case 3: timer_s *= 30.0; seed += uint(timer_s); break;
        case 4: timer_s *= 60.0; seed += uint(timer_s); break;
        case 5: seed += FRAMECOUNT; break;
    }  
    return seed;
}

void ApplyGrainCS(in CSIN i)
{
    if(GRAIN_TYPE == GRAIN_TYPE_DIGITAL) return;

    uint2 p = i.dispatchthreadid.xy;
    p = i.groupid.xy * 16 + SFC::morton_i_to_xy(i.threadid);

    float3 tcol = tex2Dfetch(ColorInput, p).rgb;
    uint seed = get_seed(p);

    float4 o = 0; 

    float tgrey = from_linear(dot(to_linear(tcol), float3(0.2126729, 0.7151522, 0.072175)));
    uint N = grain_intensity_to_halide_count();  
    
    if(FILM_MODE == FILM_MODE_MONOCHROME)
    {        
        o.rgb = sample_binomial_icdf(tgrey, Hash::next1D(seed));
    }
    else //FILM_MODE_COLOR
    {
        o.r = sample_binomial_icdf(tcol.r, Hash::next1D(seed));
        o.g = sample_binomial_icdf(tcol.g, Hash::next1D(seed));
        o.b = sample_binomial_icdf(tcol.b, Hash::next1D(seed));            
    }
    o.w = seed & 1023u; // needed for the film diffusion pass
    tex2Dstore(stGrainIntermediateTex, p, o);    
}

//more registers this way but only 1 PS overhead
void ResolvePS(in VSOUT i, out float3 o : SV_Target0)
{ 
    uint2 p = uint2(i.vpos.xy); 
    o = 0;

    if(GRAIN_TYPE == GRAIN_TYPE_ANALOG)
    {
        if(GRAIN_SIZE > 0)
        {
            float2 gaussian = float2(1, 0.5 * lerp(0.1, 1.0, GRAIN_SIZE));
            float sigma = rsqrt(grain_intensity_to_halide_count());
            
            float wsum = 0;       

            [unroll]for(int x = -1; x <= 1; x++)
            [unroll]for(int y = -1; y <= 1; y++)
            {
                uint2 tp = p + int2(x, y);
                float4 texel = tex2Dfetch(sGrainIntermediateTex, tp);
                float3 tcol = texel.rgb;
                uint rng = uint(texel.w);

                float2 rand01 = float2(rng & 63u, rng >> 6u) / 64.0; //demux random bitfield into 2 shitty rng numbers        
                //random displacement to approximate average displacement of grains (gets lower as grains increase, until it converges to a regular lowpass)
                float2 offs = float2(x, y) + BXDF::boxmuller(rand01) * sigma;
                float w = exp(-dot(offs, offs));   
                //lowpass weight    
                w *= gaussian[abs(x)] * gaussian[abs(y)];

                o += tcol * w;
                wsum += w;
            }

            o /= wsum;
        }
        else 
        {
            o = tex2Dfetch(sGrainIntermediateTex, p).rgb;
        }

        float3 center = tex2Dfetch(ColorInput, p).rgb;  
        center = filmic_curve(center, FILM_CURVE_TOE, FILM_CURVE_GAMMA);  

        [branch]
        if(FILM_MODE == FILM_MODE_COLOR)
        {
            center = to_linear(center);
            o.rgb = lerp(center, o.rgb, grain_intensity_to_blend());
        }
        else 
        {
            float grey = dot(to_linear(center), float3(0.2126729, 0.7151522, 0.072175));
            o.rgb = lerp(grey, o.rgb, grain_intensity_to_blend());
        }
        
        o.rgb = from_linear(o.rgb);   
    }
    else //DIGITAL_SENSOR_NOISE
    {
        o = tex2Dfetch(ColorInput, p).rgb;  
        o = to_linear(o);
        
        uint seed = get_seed(p);  

        //3D box muller for 3 uncorrelated gaussian distributed noise values
        float3 gaussian;     
        gaussian.z  = BXDF::boxmuller(Hash::next2D(seed)).x;

        float intensity = GRAIN_INTENSITY_DIGITAL * GRAIN_INTENSITY_DIGITAL * 0.35;

        [branch]
        if(FILM_MODE == FILM_MODE_COLOR)
        {     
            gaussian.xy = BXDF::boxmuller(Hash::next2D(seed));   
            gaussian.y *= GRAIN_USE_BAYER ? 0.7071 : 1; //monte carlo
            gaussian = lerp(gaussian.xxx, gaussian, GRAIN_SAT);
            o = to_hdr(o);
            o += gaussian * intensity;            
        }
        else 
        {
            o = dot(o, float3(0.2126729, 0.7151522, 0.072175));
            o = to_hdr(o);
            o += gaussian.z * intensity;
        }

        o = from_hdr(o);
        o = from_linear(o);
    }    
}

/*=============================================================================
	Techniques
=============================================================================*/

technique MartyMods_FilmGrain
<
    ui_label = "iMMERSE: Film Grain";
    ui_tooltip =        
        "                            MartysMods - Film Grain                           \n"
        "                   MartysMods Epic ReShade Effects (iMMERSE)                  \n"
        "______________________________________________________________________________\n"
        "\n"

        "iMMERSE Film Grain is a physically based film grain emulation effect. Modeled \n"
        "after extensive offline simulations to produce results as seen in the real world.\n"
        "\n"
        "\n"
        "Visit https://martysmods.com for more information.                            \n"
        "\n"       
        "______________________________________________________________________________";
>
{  
    pass TrackState     { ComputeShader = TrackStateCS<1, 1>;DispatchSizeX = 1; DispatchSizeY = 1;}
    pass UpdateiCDF     { ComputeShader = BuildBinomialiCDFCS<TEXTURE_RES, 1>; DispatchSizeX = 1; DispatchSizeY = GREY_LEVELS;}
    pass AnalogGrain    { ComputeShader = ApplyGrainCS<16, 16>;DispatchSizeX = CEIL_DIV(BUFFER_WIDTH, 16); DispatchSizeY = CEIL_DIV(BUFFER_HEIGHT, 16);}
    pass Resolve        { VertexShader = MainVS; PixelShader  = ResolvePS; }
}