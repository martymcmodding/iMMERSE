/*=============================================================================

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
 
=============================================================================*/

#pragma once 

//Helpers for all sorts of device queries

#define GPU_VENDOR_NVIDIA      0x10DE
#define GPU_VENDOR_AMD         0x1002
#define GPU_VENDOR_INTEL       0x8086

#define RENDERER_D3D9          0x9000
#define RENDERER_D3D10         0xA000 //>=
#define RENDERER_D3D11         0xB000 //>=
#define RENDERER_D3D12         0xC000 //>=
#define RENDERER_OPENGL       0x10000 //>=
#define RENDERER_VULKAN       0x20000 //>=

#if __RENDERER__ >= RENDERER_D3D11
 #define _COMPUTE_SUPPORTED          1
#else 
 #define _COMPUTE_SUPPORTED          0
#endif

#if __RENDERER__ >= RENDERER_D3D10
 #define _BITWISE_SUPPORTED          1
#else 
 #define _BITWISE_SUPPORTED          0
#endif

//Frequently used things / ReShade FX extensions

static const float2 BUFFER_PIXEL_SIZE = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
static const uint2 BUFFER_SCREEN_SIZE = uint2(BUFFER_WIDTH, BUFFER_HEIGHT);
static const float2 BUFFER_ASPECT_RATIO = float2(1.0, BUFFER_WIDTH * BUFFER_RCP_HEIGHT);

//DLSS/FSR/XeSS/TAAU compatibility features

#define DLSS_QUALITY           0.66666666 //    1 / 1.5
#define DLSS_BALANCED          0.58000000 //    1 / 1.72
#define DLSS_PERFORMANCE       0.50000000 //    1 / 2.0 
#define DLSS_ULTRA_PERFORMANCE 0.33333333 //    1 / 3.0

#define FSR_ULTRA_QUALITY      0.77000000 //    1 / 1.3
#define FSR_QUALITY            0.66666666 //    1 / 1.5
#define FSR_BALANCED           0.58823529 //    1 / 1.7
#define FSR_PERFORMANCE        0.50000000 //    1 / 2.0

//if we write it this way instead of ifdef, ReShade won't add this to the GUI
#ifdef _MARTYSMODS_TAAU_SCALE //this works with both the "enum" above and actual literals like 0.5
       
    //use the shit we have
    #define BUFFER_WIDTH_DLSS       int(BUFFER_WIDTH  * _MARTYSMODS_TAAU_SCALE + 0.5)
    #define BUFFER_HEIGHT_DLSS      int(BUFFER_HEIGHT * _MARTYSMODS_TAAU_SCALE + 0.5)
    #define BUFFER_RCP_WIDTH_DLSS   (1.0 / (BUFFER_WIDTH_DLSS))
    #define BUFFER_RCP_HEIGHT_DLSS  (1.0 / (BUFFER_HEIGHT_DLSS))   

#else 

    #define BUFFER_WIDTH_DLSS       BUFFER_WIDTH 
    #define BUFFER_HEIGHT_DLSS      BUFFER_HEIGHT
    #define BUFFER_RCP_WIDTH_DLSS   BUFFER_RCP_WIDTH
    #define BUFFER_RCP_HEIGHT_DLSS  BUFFER_RCP_HEIGHT

#endif

static const float2 BUFFER_PIXEL_SIZE_DLSS   = float2(BUFFER_RCP_WIDTH_DLSS, BUFFER_RCP_HEIGHT_DLSS);
static const uint2 BUFFER_SCREEN_SIZE_DLSS   = uint2(BUFFER_WIDTH_DLSS, BUFFER_HEIGHT_DLSS);
static const float2 BUFFER_ASPECT_RATIO_DLSS = float2(1.0, BUFFER_WIDTH_DLSS * BUFFER_RCP_HEIGHT_DLSS);

void FullscreenTriangleVS(in uint id : SV_VertexID, out float4 vpos : SV_Position, out float2 uv : TEXCOORD)
{
	uv = id.xx == uint2(2, 1) ? -1.0.xx : 1.0.xx; 
	vpos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

struct PSOUT1 {float4 t0 : SV_Target0;};
struct PSOUT2 {float4 t0 : SV_Target0, t1 : SV_Target1;};
struct PSOUT3 {float4 t0 : SV_Target0, t1 : SV_Target1, t2 : SV_Target2;};
struct PSOUT4 {float4 t0 : SV_Target0, t1 : SV_Target1, t2 : SV_Target2, t3 : SV_Target3;};

/*=============================================================================
	Things that should be intrinsics
    Declaring all these overloads with preprocessor is common practice but seems
    to compile a lot slower on some platforms. So I'd rather do things naively.
=============================================================================*/

#if _BITWISE_SUPPORTED

uint f32tof16x2(float2 f){return (f32tof16(f.x) << 16) | f32tof16(f.y);}
float2 f16tof32x2(uint u){return f16tof32(uint2(u >> 16, u));}

uint2 f32tof16x4(float4 f){return (f32tof16(f.xz) << 16) | f32tof16(f.yw);}
float4 f16tof32x4(uint2 u){return f16tof32(uint4(u >> 16, u)).xzyw;}

#endif

//log2 macro for uints up to 16 bit, inefficient in runtime but preprocessor doesn't care
#define T1(x,n) ((uint(x)>>(n))>0)
#define T2(x,n) (T1(x,n)+T1(x,n+1))
#define T4(x,n) (T2(x,n)+T2(x,n+2))
#define T8(x,n) (T4(x,n)+T4(x,n+4))
#define LOG2(x) (T8(x,0)+T8(x,8))

#define CEIL_DIV(num, denom) ((((num) - 1) / (denom)) + 1)

//why is smoothstep a thing but not this also...
#define linearstep(_a, _b, _x) saturate(((_x) - (_a)) * rcp((_b) - (_a)))

//why is log10 a thing but not this also...
#define exp10(_x) pow(10.0, (_x))

//why 1e-8? On some platforms the compiler truncates smaller constants? idfk, caused lots of trouble before...
#define safenormalize(_x) ((_x) * rsqrt(max(1e-8, dot((_x), (_x)))))

//SM5 syntax is so much better for sampling, why this needs to be a float4...
float4 tex2Dlod(sampler s, float2 uv, float mip) {return tex2Dlod(s, float4(uv, 0, mip));}

//sometimes makes code more elegant
float dot2(float2 a) {return dot(a, a);}
float dot2(float3 a) {return dot(a, a);}
float dot2(float4 a) {return dot(a, a);}

//same thing
float  square(float  x) {return x * x;}
float2 square(float2 x) {return x * x;}
float3 square(float3 x) {return x * x;}
float4 square(float4 x) {return x * x;}
int    square(int    x) {return x * x;}
int2   square(int2   x) {return x * x;}
int3   square(int3   x) {return x * x;}
int4   square(int4   x) {return x * x;}
uint   square(uint   x) {return x * x;}
uint2  square(uint2  x) {return x * x;}
uint3  square(uint3  x) {return x * x;}
uint4  square(uint4  x) {return x * x;}

float  max3(float  a, float  b, float  c) {return max(max(a, b), c);}
float2 max3(float2 a, float2 b, float2 c) {return max(max(a, b), c);}
float3 max3(float3 a, float3 b, float3 c) {return max(max(a, b), c);}
float4 max3(float4 a, float4 b, float4 c) {return max(max(a, b), c);}
int    max3(int    a, int    b, int    c) {return max(max(a, b), c);}
int2   max3(int2   a, int2   b, int2   c) {return max(max(a, b), c);}
int3   max3(int3   a, int3   b, int3   c) {return max(max(a, b), c);}
int4   max3(int4   a, int4   b, int4   c) {return max(max(a, b), c);}
uint   max3(uint   a, uint   b, uint   c) {return max(max(a, b), c);}
uint2  max3(uint2  a, uint2  b, uint2  c) {return max(max(a, b), c);}
uint3  max3(uint3  a, uint3  b, uint3  c) {return max(max(a, b), c);}
uint4  max3(uint4  a, uint4  b, uint4  c) {return max(max(a, b), c);}

float  min3(float  a, float  b, float  c) {return min(min(a, b), c);}
float2 min3(float2 a, float2 b, float2 c) {return min(min(a, b), c);}
float3 min3(float3 a, float3 b, float3 c) {return min(min(a, b), c);}
float4 min3(float4 a, float4 b, float4 c) {return min(min(a, b), c);}
int    min3(int    a, int    b, int    c) {return min(min(a, b), c);}
int2   min3(int2   a, int2   b, int2   c) {return min(min(a, b), c);}
int3   min3(int3   a, int3   b, int3   c) {return min(min(a, b), c);}
int4   min3(int4   a, int4   b, int4   c) {return min(min(a, b), c);}
uint   min3(uint   a, uint   b, uint   c) {return min(min(a, b), c);}
uint2  min3(uint2  a, uint2  b, uint2  c) {return min(min(a, b), c);}
uint3  min3(uint3  a, uint3  b, uint3  c) {return min(min(a, b), c);}
uint4  min3(uint4  a, uint4  b, uint4  c) {return min(min(a, b), c);}

float  max4(float  a, float  b, float  c, float  d) {return max(max(a, b), max(c, d));}
float2 max4(float2 a, float2 b, float2 c, float2 d) {return max(max(a, b), max(c, d));}
float3 max4(float3 a, float3 b, float3 c, float3 d) {return max(max(a, b), max(c, d));}
float4 max4(float4 a, float4 b, float4 c, float4 d) {return max(max(a, b), max(c, d));}
int    max4(int    a, int    b, int    c, int    d) {return max(max(a, b), max(c, d));}
int2   max4(int2   a, int2   b, int2   c, int2   d) {return max(max(a, b), max(c, d));}
int3   max4(int3   a, int3   b, int3   c, int3   d) {return max(max(a, b), max(c, d));}
int4   max4(int4   a, int4   b, int4   c, int4   d) {return max(max(a, b), max(c, d));}
uint   max4(uint   a, uint   b, uint   c, uint   d) {return max(max(a, b), max(c, d));}
uint2  max4(uint2  a, uint2  b, uint2  c, uint2  d) {return max(max(a, b), max(c, d));}
uint3  max4(uint3  a, uint3  b, uint3  c, uint3  d) {return max(max(a, b), max(c, d));}
uint4  max4(uint4  a, uint4  b, uint4  c, uint4  d) {return max(max(a, b), max(c, d));}

float  min4(float  a, float  b, float  c, float  d) {return min(min(a, b), min(c, d));}
float2 min4(float2 a, float2 b, float2 c, float2 d) {return min(min(a, b), min(c, d));}
float3 min4(float3 a, float3 b, float3 c, float3 d) {return min(min(a, b), min(c, d));}
float4 min4(float4 a, float4 b, float4 c, float4 d) {return min(min(a, b), min(c, d));}
int    min4(int    a, int    b, int    c, int    d) {return min(min(a, b), min(c, d));}
int2   min4(int2   a, int2   b, int2   c, int2   d) {return min(min(a, b), min(c, d));}
int3   min4(int3   a, int3   b, int3   c, int3   d) {return min(min(a, b), min(c, d));}
int4   min4(int4   a, int4   b, int4   c, int4   d) {return min(min(a, b), min(c, d));}
uint   min4(uint   a, uint   b, uint   c, uint   d) {return min(min(a, b), min(c, d));}
uint2  min4(uint2  a, uint2  b, uint2  c, uint2  d) {return min(min(a, b), min(c, d));}
uint3  min4(uint3  a, uint3  b, uint3  c, uint3  d) {return min(min(a, b), min(c, d));}
uint4  min4(uint4  a, uint4  b, uint4  c, uint4  d) {return min(min(a, b), min(c, d));}

float maxc(float  t) {return t;}
float maxc(float2 t) {return max(t.x, t.y);}
float maxc(float3 t) {return max3(t.x, t.y, t.z);}
float maxc(float4 t) {return max4(t.x, t.y, t.z, t.w);}
float minc(float  t) {return t;}
float minc(float2 t) {return min(t.x, t.y);}
float minc(float3 t) {return min3(t.x, t.y, t.z);}
float minc(float4 t) {return min4(t.x, t.y, t.z, t.w);}

//Trying to be clever with dot(v, 1) is unfortunately wasted ALU on today's scalar GPUs
float sum(float2 v) {return v.x + v.y;}
float sum(float3 v) {return v.x + v.y + v.z;}
float sum(float4 v) {return (v.x + v.y) + (v.z + v.w);}
int sum(int2 v)     {return v.x + v.y;}
int sum(int3 v)     {return v.x + v.y + v.z;}
int sum(int4 v)     {return (v.x + v.y) + (v.z + v.w);}
uint sum(uint2 v)   {return v.x + v.y;}
uint sum(uint3 v)   {return v.x + v.y + v.z;}
uint sum(uint4 v)   {return (v.x + v.y) + (v.z + v.w);} 

uint2 togrid(uint i, uint row_pitch)
{
    uint2 grid;
    grid.y = i / row_pitch;
    grid.x = i - grid.y * row_pitch;
    return grid;
}

uint3 togrid(uint i, uint row_pitch, uint slice_pitch)
{
    uint3 grid;
    grid.z = i / (row_pitch * slice_pitch);
    i -= grid.z * (row_pitch * slice_pitch);
    grid.y = i / row_pitch;
    grid.x = i - grid.y * row_pitch;
    return grid;
}

uint fromgrid(uint2 grid, uint row_pitch)
{
    return row_pitch * grid.y + grid.x;
}

uint fromgrid(uint3 grid, uint row_pitch, uint slice_pitch)
{    
    return (grid.z * slice_pitch + grid.y) * row_pitch + grid.x;
}
