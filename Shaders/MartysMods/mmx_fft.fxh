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

//#pragma once //allow including multiple times so we can create multiple instances of the same shader

//make sure inputs are set
#ifndef FFT_WORKING_SIZE
 #error "Define Size, bruv"
#endif
#ifndef FFT_RADIX
 #error "Define Radix, bruv"
#endif
#ifndef FFT_INSTANCE
 #error "Define instance, bruv"
#endif
#ifndef FFT_AXIS
 #error "Define axis, bruv"
#endif
#ifndef FFT_CHANNELS
 #error "Define channels, bruv"
#endif

namespace FFT_INSTANCE
{

float2 complex_conj(float2 z)
{
    return float2(z.x, -z.y);
}

float2 complex_mul(float2 c1, float2 c2)
{
#if 0 //normal
    return float2(c1.x * c2.x - c1.y * c2.y,                   
                  c1.y * c2.x + c1.x * c2.y);   
#else //gauss - maybe influences precision?
    float2 z = c1 * c2;
    return float2(z.x - z.y, dot(c1 * c2.yx, 1));
 #endif
}

float2 get_twiddle_factor(float n, float k)
{
    float2 tw; sincos((TAU * k) / n, tw.y, tw.x); return tw;
}

uint reverse_index_bits(uint index, uint size)
{
    return reversebits(index) >> (32u - size);
}

void fft_radix2(bool forward, inout float2 z0, inout float2 z1)
{
    z0 += z1;
    z1 = z0 - z1 - z1;
}

void fft_radix4(bool forward, inout float2 z[4])
{    
    fft_radix2(forward, z[0], z[2]);
    fft_radix2(forward, z[1], z[3]);

    float2 zt0 = forward ? complex_conj(z[3]).yx : complex_conj(z[3].yx);
    float2 zt1 = z[1];

    z[0] = z[0] + zt1;
	z[1] = z[2] + zt0;
	z[3] = z[2] - zt0;
	z[2] = z[0] - zt1 - zt1;
}

void fft_radix8(bool forward, inout float2 z[8])
{
    float2 A[4] = {z[0], z[2], z[4], z[6]};
    float2 B[4] = {z[1], z[3], z[5], z[7]};

    fft_radix4(forward, A);
    fft_radix4(forward, B);

    float2 tw = rsqrt(2.0);
    tw = forward ? tw : complex_conj(tw);

    float2 zt0 = complex_mul(tw, B[1]); //z[3]

    z[0] = A[0] + B[0];
	z[4] = A[0] - B[0];

	z[1] = A[1] + zt0;
	z[5] = A[1] - zt0;

    [flatten]
    if(forward)
	{		
		z[2] = float2(A[2].x - B[2].y, A[2].y + B[2].x);// V4 + i V5		
		z[6] = float2(A[2].x + B[2].y, A[2].y - B[2].x);// V4 - i V5
	}
	else
	{		
		z[2] = float2(A[2].x + B[2].y, A[2].y - B[2].x);// V4 - iV5		
		z[6] = float2(A[2].x - B[2].y, A[2].y + B[2].x);// V4 + iV5
	}

    tw.x = -tw.x;
    zt0 = complex_mul(tw, B[3]); //z[7]	

    z[3] = A[3] + zt0;
    z[7] = A[3] - zt0;
}

void fft_radix(bool forward, inout float2 z[FFT_RADIX])
{
#if FFT_RADIX == 2
    fft_radix2(forward, z[0], z[1]);
#elif FFT_RADIX == 4
    fft_radix4(forward, z);
#else 
    fft_radix8(forward, z);
#endif
}

groupshared float2 tgsm[FFT_WORKING_SIZE];

void FFTPass(uint2 dtid, uint threadid, sampler s_in, storage s_out, bool forward)
{
    static const uint group_size = FFT_WORKING_SIZE / FFT_RADIX; 
    float2 local[FFT_RADIX];
#if FFT_CHANNELS == 4
    float2 local2[FFT_RADIX];    
#endif
    [loop]
    for(uint j = 0; j < FFT_RADIX; j++)
    {
#if FFT_AXIS == 0
        uint2 p = uint2(threadid + j * group_size, dtid.y);     
#else 
        uint2 p = uint2(dtid.x, threadid + j * group_size);  
#endif
        float4 rcrc = tex2Dfetch(s_in, p);      
        local[j] = rcrc.xy;
#if FFT_CHANNELS == 4 
        local2[j] = rcrc.zw;
#endif
    }

    uint k = 0;
    [unroll]
    for(uint n = 1; n < group_size;)
    {
        //transpose with shared mem and fetch next batch
        uint curr_lane = k + (threadid - k) * FFT_RADIX;

        fft_radix(forward, local);
        [loop]for(uint j = 0; j < FFT_RADIX; j++) tgsm[curr_lane + j * n] = local[j];
        barrier();
        [loop]for(uint j = 0; j < FFT_RADIX; j++) local[j] = tgsm[threadid + j * group_size];
        barrier();
#if FFT_CHANNELS == 4 
        fft_radix(forward, local2);
        [loop]for(uint j = 0; j < FFT_RADIX; j++) tgsm[curr_lane + j * n] = local2[j];
        barrier();
        [loop]for(uint j = 0; j < FFT_RADIX; j++) local2[j] = tgsm[threadid + j * group_size];
        barrier();  
#endif

        n *= FFT_RADIX;
        k = threadid % n;

        //twiddle it
        float2 tw = get_twiddle_factor(n * FFT_RADIX, k);
        tw = forward ? tw : complex_conj(tw); 
        float2 tw_curr = tw;       
        
        [unroll]for(uint j = 1; j < FFT_RADIX; j++)
        {
            local[j] = complex_mul(tw_curr, local[j]);
#if FFT_CHANNELS == 4 
            local2[j] = complex_mul(tw_curr, local2[j]);
#endif
            tw_curr = complex_mul(tw_curr, tw);
        }
    }

    //last fft pass split off the main loop
    fft_radix(forward, local);
#if FFT_CHANNELS == 4 
    fft_radix(forward, local2);
#endif

    [loop]for(uint j = 0; j < FFT_RADIX; j++)
    {
#if FFT_CHANNELS == 4 
        float4 result = float4(local[j], local2[j]);
#else 
        float4 result = local[j].xyyy;
#endif
#if FFT_AXIS == 0
        uint2 p = uint2(threadid + j * group_size, dtid.y);     
#else 
        uint2 p = uint2(dtid.x, threadid + j * group_size);  
#endif
        tex2Dstore(s_out, p, result * rsqrt(FFT_WORKING_SIZE));
    }         
}
}