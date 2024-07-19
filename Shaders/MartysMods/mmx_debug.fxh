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
    const float3 c0 = float3( 0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const float3 c1 = float3( 0.1050930431085774, 1.404613529898575,    1.384590162594685);
    const float3 c2 = float3(-0.3308618287255563, 0.214847559468213,    0.09509516302823659);
    const float3 c3 = float3(-4.634230498983486, -5.799100973351585,  -19.33244095627987);
    const float3 c4 = float3( 6.228269936347081, 14.17993336680509,    56.69055260068105);
    const float3 c5 = float3( 4.776384997670288,-13.74514537774601,   -65.35303263337234);
    const float3 c6 = float3(-5.435455855934631,  4.645852612178535,   26.3124352495832);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}

} //namespace