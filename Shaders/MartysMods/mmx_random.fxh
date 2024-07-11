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

#include "mmx_math.fxh"

namespace Random 
{

//PG: found using hash prospector, bias 0.10704308166917044
//if you copy it with those exact coefficients, I will know >:)
uint uint_hash(uint x)
{
    x ^= x >> 16;
    x *= 0x21f0aaad;
    x ^= x >> 15;
    x *= 0xd35a2d97;
    x ^= x >> 16;
    return x;
}

float uint_to_unorm(uint u)//32
{
    return asfloat((u >> 9u) | 0x3f800000u) - 1.0;
}

float2 uint_to_unorm2(uint u)//16|16
{
    return asfloat((uint2(u << 7u, u >> 9u) & 0x7fff80u) | 0x3f800000u) - 1.0;
}

float3 uint_to_unorm3(uint u)//11|11|10
{
    return asfloat((uint3(u >> 9u,  u << 2u, u << 13u ) & 0x7ff000u) | 0x3f800000u) - 1.0;
}

float4 uint_to_unorm4(uint u)//8|8|8|8
{
    return asfloat((uint4(u >> 9u,  u >> 1u, u << 7u, u << 15u) & 0x7f8000u) | 0x3f800000u) - 1.0;
}

float  next1D(inout uint rng_state){rng_state = uint_hash(rng_state);return uint_to_unorm(rng_state);}
float2 next2D(inout uint rng_state){rng_state = uint_hash(rng_state);return uint_to_unorm2(rng_state);}
float3 next3D(inout uint rng_state){rng_state = uint_hash(rng_state);return uint_to_unorm3(rng_state);}
float4 next4D(inout uint rng_state){rng_state = uint_hash(rng_state);return uint_to_unorm4(rng_state);}

float2 boxmuller(float2 u)
{
    float2 g; sincos(TAU * u.x, g.x, g.y);
    return g * sqrt(-2.0 * log(1 - u.y));
}

float3 boxmuller3D(float3 u)
{
    float3 g; sincos(TAU * u.x, g.x, g.y);
    g.z = u.y * 2.0 - 1.0;
    g.xy *= sqrt(1.0 - g.z * g.z);      
    return g * sqrt(-2.0 * log(u.z));
}

}