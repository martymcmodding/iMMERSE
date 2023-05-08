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

namespace Sampling 
{

//for LUTs, when the volumes are placed below each other
float4 sample_volume_trilinear(sampler s, float3 uvw, int3 size, int atlas_idx)
{
    uvw = saturate(uvw);
    uvw = uvw * size - uvw;
    float3 rcpsize = rcp(size);
    uvw.xy = (uvw.xy + 0.5) * rcpsize.xy;
    
    float zlerp = frac(uvw.z);
    uvw.x = (uvw.x + uvw.z - zlerp) * rcpsize.z;

    float2 uv_a = uvw.xy;
    float2 uv_b = uvw.xy + float2(1.0/size.z, 0);
    
    int atlas_size = tex2Dsize(s).y * rcpsize.y;
    uv_a.y = (uv_a.y + atlas_idx) / atlas_size;
    uv_b.y = (uv_b.y + atlas_idx) / atlas_size;

    return lerp(tex2Dlod(s, uv_a, 0), tex2Dlod(s, uv_b, 0), zlerp);
}

//tetrahedral volume interpolation
//also DX9 safe - emulated integers suck...
float4 sample_volume_tetrahedral(sampler s, float3 uvw, int3 size, int atlas_idx)
{    
    float3 p = saturate(uvw) * (size - 1);   //p += float3(1.0/4096.0, 0, 1.0/2048.0); 
    float3 c000 = floor(p); float3 c111 = ceil(p);
    float3 delta = p - c000;
    
    //work out the axes with most/least delta (min axis goes backwards from 111)
    float3 comp = delta.xyz > delta.yzx; 
    float3 minaxis = comp.zxy * (1.0 - comp);
    float3 maxaxis = comp * (1.0 - comp.zxy);   
    
    float maxv = dot(maxaxis, delta);
    float minv = dot(minaxis, delta);
    float medv = dot(1 - maxaxis - minaxis, delta);

    float4 w = float4(1, maxv, medv, minv);
    w.xyz -= w.yzw;

    //3D coords of the 2 dynamic interpolants in the lattice    
    int3 cmin = lerp(c111, c000, minaxis);
    int3 cmax = lerp(c000, c111, maxaxis);

    return  tex2Dfetch(s, int2(c000.x + c000.z * size.x, c000.y + size.y * atlas_idx)) * w.x      
          + tex2Dfetch(s, int2(cmax.x + cmax.z * size.x, cmax.y + size.y * atlas_idx)) * w.y
          + tex2Dfetch(s, int2(cmin.x + cmin.z * size.x, cmin.y + size.y * atlas_idx)) * w.z
          + tex2Dfetch(s, int2(c111.x + c111.z * size.x, c111.y + size.y * atlas_idx)) * w.w;
}

float4 tex3D(sampler s, float3 uvw, int3 size)
{
    return sample_volume_trilinear(s, uvw, size, 0);
}

float4 sample_bicubic(sampler s, float2 iuv, int2 size)
{
    float4 uv;
	uv.xy = iuv * size;

    float2 center = floor(uv.xy - 0.5) + 0.5;
	float4 d = float4(uv.xy - center, 1 + center - uv.xy);
	float4 d2 = d * d;
	float4 d3 = d2 * d;
	float4 sd = d2 * (3 - 2 * d);

    float4 o = lerp(d2, d3, 0.3594) * 0.2; //approx |err|*255 < 0.2 < bilinear precision
	uv.xy = center - o.zw;
	uv.zw = center + 1 + o.xy;
	uv /= size.xyxy;

    float4 w = (1.0/6.0) + d * 0.5 + sd * (1.0/6.0);
	w = w.wwyy * w.zxzx;

    return w.x * tex2Dlod(s, uv.xy, 0)
	     + w.y * tex2Dlod(s, uv.zy, 0)
		 + w.z * tex2Dlod(s, uv.xw, 0)
		 + w.w * tex2Dlod(s, uv.zw, 0);
}

float4 tex2Dbicub(sampler s, float2 iuv)
{
	return sample_bicubic(s, iuv, tex2Dsize(s));
}

float4 sample_biquadratic(sampler s, float2 iuv, int2 size)
{
	float2 q = frac(iuv * size);
	float2 c = (q * (q - 1.0) + 0.5) * rcp(size);
    float4 uv = iuv.xyxy + float4(-c, c);
	return (tex2Dlod(s, uv.xy, 0)
          + tex2Dlod(s, uv.xw, 0)
		  + tex2Dlod(s, uv.zw, 0)
		  + tex2Dlod(s, uv.zy, 0)) * 0.25;
}

float4 tex2Dbiquadratic(sampler, float2 iuv)
{
    return sample_biquadratic(s, iuv, tex2Dsize(s));
}

}