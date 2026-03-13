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
#if 1 //normal
    return float2(c1.x * c2.x - c1.y * c2.y,                   
                  c1.y * c2.x + c1.x * c2.y);   
#else //gauss - maybe influences precision?
    float2 z = c1 * c2;
    return float2(z.x - z.y, dot(c1 * c2.yx, 1));
 #endif
}

float2 get_twiddle(float phi)
{
    float2 tw; sincos(phi, tw.y, tw.x); return tw;
}   

void fft_radix2(const bool forward, inout float2 z0, inout float2 z1)
{
    z0 += z1;
    z1 = z0 - z1 - z1;
}

void fft_radix4(const bool forward, inout float2 z0, 
                                    inout float2 z1, 
                                    inout float2 z2, 
                                    inout float2 z3)
{    
    fft_radix2(forward, z0, z2);
    fft_radix2(forward, z1, z3);

    z3 = forward ? complex_conj(z3).yx : complex_conj(z3.yx);
    float2 zt1 = z1;

    z0 = z0 + zt1;
	z1 = z2 + z3;
	z3 = z2 - z3;
	z2 = z0 - zt1 - zt1;
}

void fft_radix4(const bool forward, inout float2 z[4])
{    
    fft_radix2(forward, z[0], z[2]);
    fft_radix2(forward, z[1], z[3]);

    z[3] = forward ? complex_conj(z[3]).yx : complex_conj(z[3].yx);
    float2 zt1 = z[1];

    z[0] = z[0] + zt1;
	z[1] = z[2] + z[3];
	z[3] = z[2] - z[3];
	z[2] = z[0] - zt1 - zt1;    
}

void fft_radix8(const bool forward, inout float2 z[8])
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
		z[2] = float2(A[2].x - B[2].y, A[2].y + B[2].x);	
		z[6] = float2(A[2].x + B[2].y, A[2].y - B[2].x);
	}
	else
	{		
		z[2] = float2(A[2].x + B[2].y, A[2].y - B[2].x);	
		z[6] = float2(A[2].x - B[2].y, A[2].y + B[2].x);
	}

    tw.x = -tw.x;
    zt0 = complex_mul(tw, B[3]); //z[7]	

    z[3] = A[3] + zt0;
    z[7] = A[3] - zt0;
}

groupshared float tgsm[FFT_WORKING_SIZE];

void stockham_transpose(inout float2 local[8], uint id, uint n, uint k, uint fft_radix, uint group_size)
{
    uint curr_lane = k + (id - k) * fft_radix;
    [unroll]for(uint j = 0; j < fft_radix; j++) tgsm[curr_lane + j * n] = local[j].x;
    barrier(); 
    [unroll]for(uint j = 0; j < fft_radix; j++) local[j].x = tgsm[id + j * group_size];
    barrier(); 
    [unroll]for(uint j = 0; j < fft_radix; j++) tgsm[curr_lane + j * n] = local[j].y;
    barrier(); 
    [unroll]for(uint j = 0; j < fft_radix; j++) local[j].y = tgsm[id + j * group_size];
    barrier();        
}

void fft_pass_mixed_radix(uint2 dtid, uint id, sampler s_in, storage s_out, const bool forward)
{
    static const uint fft_radix = 8;
    static const uint group_size = FFT_WORKING_SIZE / fft_radix; 

    float2 localRG[8];
#if FFT_CHANNELS == 4
    float2 localBA[8];    
#endif

    [loop]
    for(uint j = 0; j < fft_radix; j++)
    {
#if FFT_AXIS == 0
        uint2 p = uint2(id + j * group_size, dtid.y);     
#else 
        uint2 p = uint2(dtid.x, id + j * group_size);  
#endif
        float4 rcrc = tex2Dfetch(s_in, p);      
        localRG[j] = rcrc.xy;
#if FFT_CHANNELS == 4 
        localBA[j] = rcrc.zw;
#endif
    }

    uint n = 1;
    uint k = 0;  

    fft_radix8(forward, localRG);
    stockham_transpose(localRG, id, n, k, fft_radix, group_size);  

#if FFT_CHANNELS == 4
    fft_radix8(forward, localBA);
    stockham_transpose(localBA, id, n, k, fft_radix, group_size);    
#endif   

    n *= fft_radix; k = id % n;

    [unroll]
    for(; n < group_size; n *= fft_radix, k = id % n)
    {
        float phi = (float(k) / float(n * fft_radix)) * TAU;
        phi = forward ? phi : -phi;
        
        [unroll]
        for(uint j = 1; j < fft_radix; j++) 
        {
            localRG[j] = complex_mul(localRG[j], get_twiddle(phi * j)); 
#if FFT_CHANNELS == 4
            localBA[j] = complex_mul(localBA[j], get_twiddle(phi * j)); 
#endif
        }     

        fft_radix8(forward, localRG);
        stockham_transpose(localRG, id, n, k, fft_radix, group_size);  

#if FFT_CHANNELS == 4
        fft_radix8(forward, localBA);
        stockham_transpose(localBA, id, n, k, fft_radix, group_size);    
#endif
    }

    uint remaining_radix = FFT_WORKING_SIZE / n;
    if(remaining_radix == 2) //4x radix2 in parallel
    {
        float4 phi = (float4(k + uint4(0, 1, 2, 3) * group_size) / FFT_WORKING_SIZE) * TAU;
        phi = forward ? phi : -phi;
        localRG[4] = complex_mul(localRG[4], get_twiddle(phi.x));
        localRG[5] = complex_mul(localRG[5], get_twiddle(phi.y));
        localRG[6] = complex_mul(localRG[6], get_twiddle(phi.z));
        localRG[7] = complex_mul(localRG[7], get_twiddle(phi.w));        
        fft_radix2(forward, localRG[0], localRG[4]);
        fft_radix2(forward, localRG[1], localRG[5]);
        fft_radix2(forward, localRG[2], localRG[6]);
        fft_radix2(forward, localRG[3], localRG[7]); 
#if FFT_CHANNELS == 4
        localBA[4] = complex_mul(localBA[4], get_twiddle(phi.x));
        localBA[5] = complex_mul(localBA[5], get_twiddle(phi.y));
        localBA[6] = complex_mul(localBA[6], get_twiddle(phi.z));
        localBA[7] = complex_mul(localBA[7], get_twiddle(phi.w));        
        fft_radix2(forward, localBA[0], localBA[4]);
        fft_radix2(forward, localBA[1], localBA[5]);
        fft_radix2(forward, localBA[2], localBA[6]);
        fft_radix2(forward, localBA[3], localBA[7]);
#endif
    }
    else if(remaining_radix == 4) //2x radix4 in parallel
    {
        float2 phi = (float2(k + uint2(0, 1) * group_size) / FFT_WORKING_SIZE) * TAU;
        phi = forward ? phi : -phi;
        localRG[2] = complex_mul(localRG[2], get_twiddle(phi.x * 1.0));
        localRG[4] = complex_mul(localRG[4], get_twiddle(phi.x * 2.0));
        localRG[6] = complex_mul(localRG[6], get_twiddle(phi.x * 3.0));
        localRG[3] = complex_mul(localRG[3], get_twiddle(phi.y * 1.0));
        localRG[5] = complex_mul(localRG[5], get_twiddle(phi.y * 2.0));
        localRG[7] = complex_mul(localRG[7], get_twiddle(phi.y * 3.0));
        fft_radix4(forward, localRG[0], localRG[2], localRG[4], localRG[6]);
        fft_radix4(forward, localRG[1], localRG[3], localRG[5], localRG[7]);
#if FFT_CHANNELS == 4
        localBA[2] = complex_mul(localBA[2], get_twiddle(phi.x * 1.0));
        localBA[4] = complex_mul(localBA[4], get_twiddle(phi.x * 2.0));
        localBA[6] = complex_mul(localBA[6], get_twiddle(phi.x * 3.0));
        localBA[3] = complex_mul(localBA[3], get_twiddle(phi.y * 1.0));
        localBA[5] = complex_mul(localBA[5], get_twiddle(phi.y * 2.0));
        localBA[7] = complex_mul(localBA[7], get_twiddle(phi.y * 3.0));
        fft_radix4(forward, localBA[0], localBA[2], localBA[4], localBA[6]);
        fft_radix4(forward, localBA[1], localBA[3], localBA[5], localBA[7]);
#endif
    } 
    else if(remaining_radix == 8) //1x radix8 duh
    { 
        float phi = (float(k) / FFT_WORKING_SIZE) * TAU;
        phi = forward ? phi : -phi;        
        [unroll]for(uint j = 1; j < fft_radix; j++) localRG[j] = complex_mul(localRG[j], get_twiddle(phi * j));
        fft_radix8(forward, localRG);
#if FFT_CHANNELS == 4
        [unroll]for(uint j = 1; j < fft_radix; j++) localBA[j] = complex_mul(localBA[j], get_twiddle(phi * j));
        fft_radix8(forward, localBA);
#endif
    } 

    [loop]
    for(uint j = 0; j < fft_radix; j++)
    {
#if FFT_AXIS == 0
        uint2 p = uint2(id + j * group_size, dtid.y);     
#else 
        uint2 p = uint2(dtid.x, id + j * group_size);  
#endif
#if FFT_CHANNELS == 4 
        tex2Dstore(s_out, p, float4(localRG[j], localBA[j]) * rsqrt(FFT_WORKING_SIZE));
#else 
        tex2Dstore(s_out, p, float4(localRG[j].xyyy) * rsqrt(FFT_WORKING_SIZE));
#endif
    }
}


} //namespace