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

namespace Texture 
{ 

float4 sample2D_biquadratic(sampler s, float2 iuv, int2 size)
{
	float2 q = frac(iuv * size);
	float2 c = (q * (q - 1.0) + 0.5) * rcp(size);
    float4 uv = iuv.xyxy + float4(-c, c);
	return (tex2Dlod(s, uv.xy, 0)
          + tex2Dlod(s, uv.xw, 0)
		  + tex2Dlod(s, uv.zw, 0)
		  + tex2Dlod(s, uv.zy, 0)) * 0.25;
}

float4 sample2D_biquadratic_auto(sampler s, float2 iuv)
{
    return sample2D_biquadratic(s, iuv, tex2Dsize(s));
}   

//Optimized Bspline bicubic filtering
//FXC assembly: 37->25 ALU, 5->3 registers
//One texture coord known early, better for latency
float4 sample2D_bspline(sampler s, float2 iuv, int2 size)
{
    float4 uv;
	uv.xy = iuv * size;

    float2 center = floor(uv.xy - 0.5) + 0.5;
	float4 d = float4(uv.xy - center, 1 + center - uv.xy);
	float4 d2 = d * d;
	float4 d3 = d2 * d;

    float4 o = d2 * 0.12812 + d3 * 0.07188; //approx |err|*255 < 0.2 < bilinear precision
	uv.xy = center - o.zw;
	uv.zw = center + 1 + o.xy;
	uv /= size.xyxy;

    float4 w = 0.16666666 + d * 0.5 + 0.5 * d2 - d3 * 0.3333333;
	w = w.wwyy * w.zxzx;

    return w.x * tex2Dlod(s, uv.xy, 0)
	     + w.y * tex2Dlod(s, uv.zy, 0)
		 + w.z * tex2Dlod(s, uv.xw, 0)
		 + w.w * tex2Dlod(s, uv.zw, 0);
}

float4 sample2D_bspline_auto(sampler s, float2 iuv)
{
	return sample2D_bspline(s, iuv, tex2Dsize(s));
}

float4 sample2D_catmullrom(in sampler tex, in float2 uv, in float2 texsize)
{
    float2 UV =  uv * texsize;
    float2 tc = floor(UV - 0.5) + 0.5;
	float2 f = UV - tc;
	float2 f2 = f * f; 
	float2 f3 = f2 * f;
    
    float2 w0 = f2 - 0.5 * (f3 + f);
	float2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
	float2 w3 = 0.5 * (f3 - f2);
	float2 w12 = 1.0 - w0 - w3;

    float4 ws[3];    
    ws[0].xy = w0;
	ws[1].xy = w12;
	ws[2].xy = w3;

	ws[0].zw = tc - 1.0;
	ws[1].zw = tc + 1.0 - w1 / w12;
	ws[2].zw = tc + 2.0;

	ws[0].zw /= texsize;
	ws[1].zw /= texsize;
	ws[2].zw /= texsize;

    float4 ret;
    ret  = tex2Dlod(tex, float2(ws[1].z, ws[0].w), 0) * ws[1].x * ws[0].y;    
    ret += tex2Dlod(tex, float2(ws[0].z, ws[1].w), 0) * ws[0].x * ws[1].y;    
    ret += tex2Dlod(tex, float2(ws[1].z, ws[1].w), 0) * ws[1].x * ws[1].y;    
    ret += tex2Dlod(tex, float2(ws[2].z, ws[1].w), 0) * ws[2].x * ws[1].y;    
    ret += tex2Dlod(tex, float2(ws[1].z, ws[2].w), 0) * ws[1].x * ws[2].y;    
    float normfact = 1.0 / (1.0 - (f.x - f2.x)*(f.y - f2.y) * 0.25); //PG23: closed form for the weight sum
    return max(0, ret * normfact);   
}

float4 sample2D_catmullrom_auto(sampler s, float2 iuv)
{
	return sample2D_catmullrom(s, iuv, tex2Dsize(s));
}

//for LUTs, when the volumes are placed below each other
float4 sample3D_trilinear(sampler s, float3 uvw, int3 size, int atlas_idx)
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
float4 sample3D_tetrahedral(sampler s, float3 uvw, int3 size, int atlas_idx)
{
    float3 p = saturate(uvw) * (size - 1);
    float3 c000 = floor(p); float3 c111 = ceil(p);
    float3 f = p - c000;

    float maxv = max(max(f.x, f.y), f.z);
    float minv = min(min(f.x, f.y), f.z);
    float medv = dot(f, 1) - maxv - minv;

    float3 minaxis = minv == f.x ? float3(1,0,0) : (minv == f.y ? float3(0,1,0) : float3(0,0,1));
    float3 maxaxis = maxv == f.x ? float3(1,0,0) : (maxv == f.y ? float3(0,1,0) : float3(0,0,1));
             
    int3 cmin = lerp(c111, c000, minaxis);
    int3 cmax = lerp(c000, c111, maxaxis);

    //3D barycentric
    float4 w = float4(1, maxv, medv, minv);
    w.xyz -= w.yzw;

    return  tex2Dfetch(s, int2(c000.x + c000.z * size.x, c000.y + size.y * atlas_idx)) * w.x     //000       
          + tex2Dfetch(s, int2(cmax.x + cmax.z * size.x, cmax.y + size.y * atlas_idx)) * w.y     //max
          + tex2Dfetch(s, int2(cmin.x + cmin.z * size.x, cmin.y + size.y * atlas_idx)) * w.z     //min
          + tex2Dfetch(s, int2(c111.x + c111.z * size.x, c111.y + size.y * atlas_idx)) * w.w;    //111
}

}