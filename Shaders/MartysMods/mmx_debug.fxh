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

namespace Debug
{

float3 viridis(float t)
{
    t = saturate(t);
    float3 res =          float3(-5.435455855934631,   4.645852612178535,   26.3124352495832);
    res = mad(res, t.xxx, float3( 4.776384997670288, -13.74514537774601,   -65.35303263337234));
    res = mad(res, t.xxx, float3( 6.228269936347081,  14.17993336680509,    56.69055260068105));
    res = mad(res, t.xxx, float3(-4.634230498983486,  -5.799100973351585,  -19.33244095627987));
    res = mad(res, t.xxx, float3(-0.3308618287255563,  0.214847559468213,    0.09509516302823659));
    res = mad(res, t.xxx, float3( 0.1050930431085774,  1.404613529898575,    1.384590162594685));
    res = mad(res, t.xxx, float3( 0.2777273272234177,  0.005407344544966578, 0.3340998053353061));
    return saturate(res);
}

float3 turbo(float t)
{
    t = saturate(t);
	float3 res =          float3(59.2864, 2.82957, 27.3482);
	res = mad(res, t.xxx, float3(-152.94239396, 4.2773, -89.9031));	
	res = mad(res, t.xxx, float3(132.13108234, -14.185, 110.36276771));
	res = mad(res, t.xxx, float3(-42.6603, 4.84297, -60.582));
	res = mad(res, t.xxx, float3(4.61539, 2.19419, 12.6419));
	res = mad(res, t.xxx, float3(0.135721, 0.0914026, 0.106673));
	return saturate(res);
}

float3 inferno(float t)
{
    t = saturate(t);
	float3 res =          float3( 4.069046086, -4.193858954,  4.324996022);
	res = mad(res, t.xxx, float3(-8.490712758, +8.389314011, -3.608884658));
	res = mad(res, t.xxx, float3(+3.892783760, -4.821108251, +2.798380308));
	res = mad(res, t.xxx, float3(+0.278906882, +1.605395918, -5.893222355));
	res = mad(res, t.xxx, float3(+1.228188385, +0.015360518, +3.122510347));
	res = mad(res, t.xxx, float3(-0.027780558, +0.014065206, -0.019628385));
	return saturate(res);
}

} //namespace