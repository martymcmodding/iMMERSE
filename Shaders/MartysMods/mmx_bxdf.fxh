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
#include "mmx_math.fxh"

namespace BXDF
{

/*=============================================================================
    Basics
=============================================================================*/

float2 sample_disc(float2 u)
{
    float2 dir;
    sincos(u.x * TAU, dir.y, dir.x);        
    dir *= sqrt(u.y);
    return dir;
}

float3 sample_sphere(float2 u)
{
    float3 dir;
    sincos(u.x * TAU, dir.y, dir.x);        
    dir.z = u.y * 2.0 - 1.0; 
    dir.xy *= sqrt(1.0 - dir.z * dir.z);
    return dir;
}

float3 ray_cosine(float2 u, float3 n)
{
    return normalize(sample_sphere(u) + n);
}

float3 ray_uniform(float2 u, float3 n)
{
    float3 dir = sample_sphere(u);
    dir = dot(dir, n) < 0 ? -dir : dir;
    return normalize(dir + n * 0.01);
}

//phase functions
float3 sample_phase_henyey_greenstein(float3 wo, float g, float2 u)
{
    float3 wi; sincos(TAU * u.y, wi.x, wi.y);
    float sqr = (1 - g * g) / (1 - g + 2 * g * u.x);    
    wi.z = (1 + g * g - sqr * sqr) / (2 * g); //cos(theta)
    wi.xy *= sqrt(saturate(1 - wi.z * wi.z)); //sin(theta)
    return mul(wi, Math::base_from_vector(wo));
}

/*=============================================================================
	PBR
=============================================================================*/

float fresnel_schlick(float cos_theta, float F0)
{
    float f = saturate(1 - cos_theta);
    float f2 = f * f;   
    return mad(f2 * f2 * f, 1 - F0, F0);
}

/*=============================================================================
	GGX / Trowbridge-Reitz
=============================================================================*/

namespace GGX 
{

float smith_G1(float ndotx, float alpha)
{
	float ndotx2 = ndotx * ndotx;
    float tantheta2 = (1 - ndotx2) / ndotx2;
    return 2 / (sqrt(mad(alpha*alpha, tantheta2, 1)) + 1);
}

float smith_G2_heightcorrelated(float ndotl, float ndotv, float alpha)
{
    float a2 = alpha * alpha;
    float termv = ndotl * sqrt((-ndotv * a2 + ndotv) * ndotv + a2);
    float terml = ndotv * sqrt((-ndotl * a2 + ndotl) * ndotl + a2);
    return (2 * ndotv * ndotl) / (termv + terml);
}

float smith_G2_over_G1_heightcorrelated(float alpha, float ndotwi, float ndotwo)
{
    float G1wi = smith_G1(ndotwi, alpha);
    float G1wo = smith_G1(ndotwo, alpha);
    return G1wi / (G1wi + G1wo - G1wi * G1wo);
}

float spec_half_angle_from_alpha(float alpha)
{
    return PI * alpha / (1 + alpha);
}

//Dupuy et al. VNDF sampling with spherical caps
//Same PDF as Heitz' GGX, thus can be used with F * G2 / G1
//no reason to keep Heitz' VNDF around, this is just better
float3 sample_vndf(float3 wi, float2 alpha, float2 u, float coverage)
{
    //warp to the hemisphere configuration 
    float3 wi_std = normalize(float3(wi.xy * alpha, wi.z));
    //construct spherical cap
    float3 c;
    c.z = mad((1 - u.y * coverage), (1 + wi_std.z), -wi_std.z);
    sincos(u.x * TAU, c.x, c.y);
    c.xy *= sqrt(saturate(1 - c.z * c.z));
    //compute halfway direction as standard normal
    float3 wm_std = wi_std + c;
    //warp back to the ellipsoid configuration
    return normalize(float3(wm_std.xy * alpha, wm_std.z));
}

//"Bounded VNDF Sampling for the Smith–GGX BRDF" Yusuke Tokuyoshi and Kenta Eto 2024
//Modified by Pascal Gilcher to add sample coverage and calculate ratio of bounded and unbounded vndf
//Multiply G2/G1 * F with pdf_ratio and it behave like regular VNDF sampling
float3 sample_vndf_bounded(float3 wi, float2 alpha, float2 u, float coverage, out float pdf_ratio)
{
    //preliminary variables
    float z2 = wi.z * wi.z;
    float a = saturate(min(alpha.x, alpha.y)); // Eq. 6
    float a2 = a * a;
    //warp to the hemisphere configuration 
    float3 wi_std = float3(wi.xy * alpha, wi.z);
    float t = sqrt((1 - z2) * a2 + z2);
    wi_std /= t; 
    //compute lower bound for scaling 
    float s = 1 + sqrt(saturate(1 - z2)); // Omit sgn for a <=1
    float s2 = s * s;    
    float k = (1 - a2) * s2 / (s2 + a2 * z2);    
    //calculate ratio of bounded and unbounded vndf
    pdf_ratio = (k * wi.z + t) / (wi.z + t);  
    //construct spherical cap
    float b = wi_std.z;
    b = wi.z > 0 ? k * b : b;    
    float3 c;
    c.z = mad((1 - u.y * coverage), (1 + b), -b);
    sincos(u.x * TAU, c.x, c.y);
    c.xy *= sqrt(saturate(1 - c.z * c.z));
    //compute halfway direction as standard normal
    float3 wm_std = c + wi_std;
    //warp back to the ellipsoid configuration
    return normalize(float3(wm_std.xy * alpha, wm_std.z));
}

//Same as above but isotropic and combined with Dupuy's TBN-less method of sampling
float3 sample_vndf_bounded_iso(float3 wi, float3 n, float alpha, float2 u, float coverage, out float pdf_ratio)
{
    //decompose into tangential and orthogonal
    float wi_z = dot(wi, n);
    float3 wi_xy = wi - wi_z * n;    
    //preliminary variables
    float a = saturate(alpha);
    float a2 = a * a;
    float z2 = wi_z * wi_z;    
    //warp to the hemisphere configuration    
    float3 wiStd = lerp(wi, wi_z * n, 1 + alpha);
    float t = sqrt((1 - z2) * a2 + z2);
    wiStd /= t;    
    //compute lower bound for scaling
    float s = 1 + sqrt(1 - z2);
    float s2 = s * s;
    float k = (s2 - a2 * s2) / (s2 + a2 * z2); 
    //calculate ratio of bounded and unbounded vndf
    pdf_ratio = (k * wi_z + t) / (wi_z + t);    
    //construct spherical cap
    float3 c_std; 
    float b = dot(wiStd, n); //z axis
    b = wi_z > 0 ? k * b : b;    
    c_std.z = mad((1 - u.y * coverage), (1 + b), -b);   
    sincos(u.x * TAU, c_std.x, c_std.y);
    c_std.xy *= sqrt(saturate(1.0 - c_std.z * c_std.z));
    //reflect sample to align with normal
    float3 wr = float3(n.xy, n.z + 1);
    float3 c = (dot(wr, c_std) / wr.z) * wr - c_std;
    //compute halfway direction as standard normal
    float3 wm_std = c + wiStd;
    float3 wm_std_z = n * dot(n, wm_std);
    float3 wm_std_xy = wm_std_z - wm_std;
    //warp back to the ellipsoid configuration
    return normalize(wm_std_z + alpha * wm_std_xy);
}

//D term for GGX
float ndf(float ndoth, float alpha)
{
	float a2 = alpha * alpha;
	float d = ((ndoth * a2 - ndoth) * ndoth + 1);
	return a2 / (d * d * PI);
}

float pdf_vndf_bounded_iso(float3 wi, float3 wo, float3 n, float alpha) 
{    
    float3 m = normalize(wi + wo);
    float ndoth = saturate(dot(m, n));
    float ndf = ndf(ndoth, alpha);

    float wi_z = dot(n, wi);
    float z2 = wi_z * wi_z;
    float a = saturate(alpha);
    float a2 = a * a; 
    float len2 = (1 - z2) * a2;
    float t = sqrt(len2 + z2);
  
    if(wi_z > 0.0)
    {       
        float s = 1 + sqrt(saturate(1 - z2));  
        float s2 = s * s;
        float k = (1 - a2) * s2 / (s2 + a2 * z2); 
        return ndf / (2 * (k * wi_z + t)) ;
    }
    //Numerically stable form of the previous PDF for i.z < 0
    return ndf * (t - wi_z) / (2 * len2);
}

float3 dominant_direction(float3 n, float3 v, float alpha)
{
    float roughness = sqrt(alpha);
    float f = (1 - roughness) * (sqrt(1 - roughness) + roughness);
    float3 r = reflect(-v, n);
    return normalize(lerp(n, r, f));
}

} //namespace GGX

} //namespace