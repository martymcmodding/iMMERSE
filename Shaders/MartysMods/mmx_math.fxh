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

#include "mmx_global.fxh"

static const float PI      = 3.1415926535;
static const float HALF_PI = 1.5707963268;
static const float TAU     = 6.2831853072;

static const float FLOAT32MAX = 3.402823466e+38f;
static const float FLOAT16MAX = 65504.0;

//Useful math functions

namespace Math 
{

/*=============================================================================
	Fast Math
=============================================================================*/

float fast_sign(float x){return x >= 0.0 ? 1.0 : -1.0;}
float2 fast_sign(float2 x){return x >= 0.0.xx ? 1.0.xx : -1.0.xx;}
float3 fast_sign(float3 x){return x >= 0.0.xxx ? 1.0.xxx : -1.0.xxx;}
float4 fast_sign(float4 x){return x >= 0.0.xxxx ? 1.0.xxxx : -1.0.xxxx;}

#if COMPUTE_SUPPORTED != 0
 #define fast_sqrt(_x) asfloat(0x1FBD1DF5 + (asint(_x) >> 1))
#else 
 #define fast_sqrt(_x) sqrt(_x) //not bitwise shenanigans :(
#endif

float fast_acos(float x)                      
{                                                   
    float o = -0.156583 * abs(x) + HALF_PI;
    o *= fast_sqrt(1.0 - abs(x));              
    return x >= 0.0 ? o : PI - o;                   
}

float2 fast_acos(float2 x)                      
{                                                   
    float2 o = -0.156583 * abs(x) + HALF_PI;
    o *= fast_sqrt(1.0 - abs(x));              
    return x >= 0.0.xx ? o : PI - o;                   
}

/*=============================================================================
	Geometry
=============================================================================*/

float4 get_rotator(float phi)
{
    float2 t;
    sincos(phi, t.x, t.y);
    return float4(t.yx, -t.x, t.y);
}

float4 merge_rotators(float4 ra, float4 rb)
{
    return ra.xyxy * rb.xxzz + ra.zwzw * rb.yyww;
}

float2 rotate_2D(float2 v, float4 r)
{
    return float2(dot(v, r.xy), dot(v, r.zw));
}

float3x3 get_rotation_matrix(float3 axis, float angle)
{
    //http://www.songho.ca/opengl/gl_rotate.html
    float s, c; sincos(angle, s, c);
    float3x3 m = float3x3((1 - c) * axis.xxx * axis.xyz + float3(c, -s * axis.z, s * axis.y),
                          (1 - c) * axis.xyy * axis.yyz + float3(s * axis.z, c, -s * axis.x),
                          (1 - c) * axis.xyz * axis.zzz + float3(-s * axis.y, s * axis.x, c));
    return m;
}

float3x3 base_from_vector(float3 n)
{
    //pixar's method, optimized for ALU
    float2 nz = -n.xy / (1.0 + abs(n.z));//add_abs, rcp, mul
    float3 t = float3(1.0 + n.x*nz.x, n.x*nz.y, -n.x);//mad, mul, mov              
	float3 b = float3(1.0 + n.y*nz.y, n.x*nz.y, -n.y);//mad, mul, mov  
    //moving the crossover boundary back such that it doesn't flipflop on flat surfaces                
    t.z  = n.z >= 0.5 ? t.z : -t.z;//cmov
    b.xy = n.z >= 0.5 ? b.yx : -b.yx;//cmov
    return float3x3(t, b, n); 
}

float3 aabb_clip(float3 p, float3 mincorner, float3 maxcorner)
{
    float3 center = 0.5 * (maxcorner + mincorner);
    float3 range  = 0.5 * (maxcorner - mincorner);
    float3 delta = p - center;

    float3 t = abs(range / (delta + 1e-7));
    float mint = saturate(min(min(t.x, t.y), t.z));

    return center + delta * mint;
}

float2 aabb_hit_01(float2 origin, float2 dir)
{
    float2 hit_t = abs((dir < 0.0.xx ? origin : 1.0.xx - origin) / dir);
    return origin + dir * min(hit_t.x, hit_t.y);
}

float3 aabb_hit_01(float3 origin, float3 dir)
{
    float3 hit_t = abs((dir < 0.0.xxx ? origin : 1.0.xxx - origin) / dir);
    return origin + dir * min(min(hit_t.x, hit_t.y), hit_t.z);
}

bool inside_screen(float2 uv)
{
    return all(saturate(uv - uv * uv));
}

//TODO move to a packing header

//normalized 3D in, [0, 1] 2D out
float2 octahedral_enc(in float3 v) 
{
    float2 result = v.xy * rcp(dot(abs(v), 1)); 
    float2 sgn = fast_sign(v.xy);
    result = v.z < 0 ? sgn - abs(result.yx) * sgn : result;
    return result * 0.5 + 0.5;
}

//[0, 1] 2D in, normalized 3D out
float3 octahedral_dec(float2 o) 
{
    o = o * 2.0 - 1.0;
    float3 v = float3(o.xy, 1.0 - abs(o.x) - abs(o.y));
    //v.xy = v.z < 0 ? (1.0 - abs(v.yx)) * fast_sign(v.xy) : v.xy;
    float t = saturate(-v.z);
    v.xy += v.xy >= 0.0.xx ? -t.xx : t.xx;
    return normalize(v);
}

float3x3 invert(float3x3 m)
{
    float3x3 adj;
    adj[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]); 
    adj[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]); 
    adj[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
    adj[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    adj[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]); 
    adj[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    adj[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]); 
    adj[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]); 
    adj[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]); 

    float det = dot(float3(adj[0][0], adj[0][1], adj[0][2]), float3(m[0][0], m[1][0], m[2][0]));
    return adj * rcp(det + (abs(det) < 1e-8));
}

float4x4 invert(float4x4 m)  
{
    float4x4 adj;
    adj[0][0] = m[2][1] * m[3][2] * m[1][3] - m[3][1] * m[2][2] * m[1][3] + m[3][1] * m[1][2] * m[2][3] - m[1][1] * m[3][2] * m[2][3] - m[2][1] * m[1][2] * m[3][3] + m[1][1] * m[2][2] * m[3][3];
    adj[0][1] = m[3][1] * m[2][2] * m[0][3] - m[2][1] * m[3][2] * m[0][3] - m[3][1] * m[0][2] * m[2][3] + m[0][1] * m[3][2] * m[2][3] + m[2][1] * m[0][2] * m[3][3] - m[0][1] * m[2][2] * m[3][3];
    adj[0][2] = m[1][1] * m[3][2] * m[0][3] - m[3][1] * m[1][2] * m[0][3] + m[3][1] * m[0][2] * m[1][3] - m[0][1] * m[3][2] * m[1][3] - m[1][1] * m[0][2] * m[3][3] + m[0][1] * m[1][2] * m[3][3];
    adj[0][3] = m[2][1] * m[1][2] * m[0][3] - m[1][1] * m[2][2] * m[0][3] - m[2][1] * m[0][2] * m[1][3] + m[0][1] * m[2][2] * m[1][3] + m[1][1] * m[0][2] * m[2][3] - m[0][1] * m[1][2] * m[2][3];

    adj[1][0] = m[3][0] * m[2][2] * m[1][3] - m[2][0] * m[3][2] * m[1][3] - m[3][0] * m[1][2] * m[2][3] + m[1][0] * m[3][2] * m[2][3] + m[2][0] * m[1][2] * m[3][3] - m[1][0] * m[2][2] * m[3][3];
    adj[1][1] = m[2][0] * m[3][2] * m[0][3] - m[3][0] * m[2][2] * m[0][3] + m[3][0] * m[0][2] * m[2][3] - m[0][0] * m[3][2] * m[2][3] - m[2][0] * m[0][2] * m[3][3] + m[0][0] * m[2][2] * m[3][3];
    adj[1][2] = m[3][0] * m[1][2] * m[0][3] - m[1][0] * m[3][2] * m[0][3] - m[3][0] * m[0][2] * m[1][3] + m[0][0] * m[3][2] * m[1][3] + m[1][0] * m[0][2] * m[3][3] - m[0][0] * m[1][2] * m[3][3];
    adj[1][3] = m[1][0] * m[2][2] * m[0][3] - m[2][0] * m[1][2] * m[0][3] + m[2][0] * m[0][2] * m[1][3] - m[0][0] * m[2][2] * m[1][3] - m[1][0] * m[0][2] * m[2][3] + m[0][0] * m[1][2] * m[2][3];

    adj[2][0] = m[2][0] * m[3][1] * m[1][3] - m[3][0] * m[2][1] * m[1][3] + m[3][0] * m[1][1] * m[2][3] - m[1][0] * m[3][1] * m[2][3] - m[2][0] * m[1][1] * m[3][3] + m[1][0] * m[2][1] * m[3][3];
    adj[2][1] = m[3][0] * m[2][1] * m[0][3] - m[2][0] * m[3][1] * m[0][3] - m[3][0] * m[0][1] * m[2][3] + m[0][0] * m[3][1] * m[2][3] + m[2][0] * m[0][1] * m[3][3] - m[0][0] * m[2][1] * m[3][3];
    adj[2][2] = m[1][0] * m[3][1] * m[0][3] - m[3][0] * m[1][1] * m[0][3] + m[3][0] * m[0][1] * m[1][3] - m[0][0] * m[3][1] * m[1][3] - m[1][0] * m[0][1] * m[3][3] + m[0][0] * m[1][1] * m[3][3];
    adj[2][3] = m[2][0] * m[1][1] * m[0][3] - m[1][0] * m[2][1] * m[0][3] - m[2][0] * m[0][1] * m[1][3] + m[0][0] * m[2][1] * m[1][3] + m[1][0] * m[0][1] * m[2][3] - m[0][0] * m[1][1] * m[2][3];

    adj[3][0] = m[3][0] * m[2][1] * m[1][2] - m[2][0] * m[3][1] * m[1][2] - m[3][0] * m[1][1] * m[2][2] + m[1][0] * m[3][1] * m[2][2] + m[2][0] * m[1][1] * m[3][2] - m[1][0] * m[2][1] * m[3][2];
    adj[3][1] = m[2][0] * m[3][1] * m[0][2] - m[3][0] * m[2][1] * m[0][2] + m[3][0] * m[0][1] * m[2][2] - m[0][0] * m[3][1] * m[2][2] - m[2][0] * m[0][1] * m[3][2] + m[0][0] * m[2][1] * m[3][2];
    adj[3][2] = m[3][0] * m[1][1] * m[0][2] - m[1][0] * m[3][1] * m[0][2] - m[3][0] * m[0][1] * m[1][2] + m[0][0] * m[3][1] * m[1][2] + m[1][0] * m[0][1] * m[3][2] - m[0][0] * m[1][1] * m[3][2];
    adj[3][3] = m[1][0] * m[2][1] * m[0][2] - m[2][0] * m[1][1] * m[0][2] + m[2][0] * m[0][1] * m[1][2] - m[0][0] * m[2][1] * m[1][2] - m[1][0] * m[0][1] * m[2][2] + m[0][0] * m[1][1] * m[2][2];

    float det = dot(float4(adj[0][0], adj[1][0], adj[2][0], adj[3][0]), float4(m[0][0], m[0][1],  m[0][2],  m[0][3]));
    return adj * rcp(det + (abs(det) < 1e-8));
}

float2 anisotropy_map(float2 kernel, float3 n, float limit)
{    
    n.xy *= limit;
    float2 distorted = kernel - n.xy * dot(n.xy, kernel);
    return distorted;
}

//with elongation
float2 anisotropy_map2(float2 kernel, float3 n, float limit)
{    
    n.xy *= limit;
    float cosine = rsqrt(1 - dot(n.xy, n.xy));
    float2 distorted = kernel - n.xy * dot(n.xy, kernel) * cosine;
    return distorted * cosine;
}

float chebyshev_weight(float mean, float variance, float xi)
{
    return saturate(variance * rcp(max(1e-7, variance + (xi - mean) * (xi - mean))));
}

//DX9 safe float emulated bitfields... needed this for something that didn't work out
//so I dumped it here in case I need it again. Works up to 24 (25?) digits and must be init with 0!
bool bitfield_get(float bitfield, int bit)
{
	float state = floor(bitfield / exp2(bit)); //"right shift"
	return frac(state * 0.5) > 0.25; //"& 1"
}

void bitfield_set(inout float bitfield, int bit, bool value)
{
	bool is_set = bitfield_get(bitfield, bit);
	//bitfield += exp2(bit) * (is_set != value) * (value ? 1 : -1);
	bitfield += exp2(bit) * (value - is_set);	
}

}