/*=============================================================================
                                                           
 d8b 888b     d888 888b     d888 8888888888 8888888b.   .d8888b.  8888888888 
 Y8P 8888b   d8888 8888b   d8888 888        888   Y88b d88P  Y88b 888        
     88888b.d88888 88888b.d88888 888        888    888 Y88b.      888        
 888 888Y88888P888 888Y88888P888 8888888    888   d88P  "Y888b.   8888888    
 888 888 Y888P 888 888 Y888P 888 888        8888888P"      "Y88b. 888        
 888 888  Y8P  888 888  Y8P  888 888        888 T88b         "888 888        
 888 888   "   888 888   "   888 888        888  T88b  Y88b  d88P 888        
 888 888       888 888       888 8888888888 888   T88b  "Y8888P"  8888888888                                                                 
                                                                            
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

===============================================================================

    Motion Blur

    Author:         Pascal Gilcher

    More info:      https://martysmods.com
                    https://patreon.com/mcflypg
                    https://github.com/martymcmodding  	

=============================================================================*/

/*=============================================================================
	Preprocessor settings
=============================================================================*/

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

/*=============================================================================
	Textures, Samplers, Globals, Structs
=============================================================================*/

//do NOT change anything here. "hurr durr I changed this and now it works"
//you ARE breaking things down the line, if the shader does not work without changes
//here, it's by design.

texture ColorInputTex : COLOR;
texture DepthInputTex : DEPTH;
sampler ColorInput 	{ Texture = ColorInputTex; };
sampler DepthInput  { Texture = DepthInputTex; };

#include ".\MartysMods\mmx_global.fxh"
#include ".\MartysMods\mmx_depth.fxh"
#include ".\MartysMods\mmx_math.fxh"
#include ".\MartysMods\mmx_deferred.fxh"

texture TileDownsampleIntermediateTex               { Width = BUFFER_WIDTH/20;   Height = BUFFER_HEIGHT;   Format = RG16F;  };
sampler sTileDownsampleIntermediateTex              { Texture = TileDownsampleIntermediateTex; };
texture TileTex               						{ Width = BUFFER_WIDTH/20;   Height = BUFFER_HEIGHT/20;   Format = RG16F;  };
sampler sTileTex              						{ Texture = TileTex; };
texture TileTexDilated               				{ Width = BUFFER_WIDTH/20;   Height = BUFFER_HEIGHT/20;   Format = RG16F;  };
sampler sTileTexDilated              				{ Texture = TileTexDilated;MipFilter=POINT; MagFilter=POINT; MinFilter=POINT; };

struct VSOUT
{
    float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

struct CSIN 
{
    uint3 groupthreadid     : SV_GroupThreadID;         //XYZ idx of thread inside group
    uint3 groupid           : SV_GroupID;               //XYZ idx of group inside dispatch
    uint3 dispatchthreadid  : SV_DispatchThreadID;      //XYZ idx of thread inside dispatch
    uint threadid           : SV_GroupIndex;            //flattened idx of thread inside group
};

/*=============================================================================
	Functions
=============================================================================*/


/*=============================================================================
	Shader Entry Points
=============================================================================*/

VSOUT MainVS(in uint id : SV_VertexID)
{
    VSOUT o;
    FullscreenTriangleVS(id, o.vpos, o.uv); 
    return o;
}

//minimize by 20 horizontally
void TileDownsample0PS(in VSOUT i, out float2 o : SV_Target0)
{	
	float3 wavg = 0;
	float3 maxvec = 0;

	for(int x = 0; x < 20; x++)
	{
		float2 p = floor(i.vpos.xy) * float2(20.0, 1.0) + float2(x, 0);
		p *= BUFFER_PIXEL_SIZE;

		float2 vec = Deferred::get_motion(p);
		float len2 = dot(vec, vec);

		maxvec = len2 > maxvec.z ? float3(vec, len2) : maxvec;
		wavg += float3(vec * len2, len2); //weighted avg by squared length, just so a _single_ outlier does not fuck up our avg completely as our estimated vectors are noisy and inaccurate subpixel-wise
	}

	wavg.xy /= wavg.z + 1e-7;

	//cos angle between max and weighted avg -> 
	float cosv = dot(safenormalize(wavg.xy), safenormalize(maxvec.xy));
	float confidenceinmax = saturate(1.0 - cosv * 50.0);
	o = lerp(wavg.xy, maxvec.xy, confidenceinmax);
	 
}

//minimize by 20 vertically
void TileDownsample1PS(in VSOUT i, out float2 o : SV_Target0)
{	
	float3 wavg = 0;
	float3 maxvec = 0;

	for(int y = 0; y < 20; y++)
	{
		float2 p = floor(i.vpos.xy) * float2(1.0, 20.0) + float2(0, y);
		float2 vec = tex2Dfetch(sTileDownsampleIntermediateTex, p).xy;
		float len2 = dot(vec, vec);

		maxvec = len2 > maxvec.z ? float3(vec, len2) : maxvec;
		wavg += float3(vec * len2, len2);  //weighted avg by squared length, just so a _single_ outlier does not fuck up our avg completely as our estimated vectors are noisy and inaccurate subpixel-wise
	}

	wavg.xy /= wavg.z + 1e-7;

	//cos angle between max and weighted avg -> 
	float cosv = dot(safenormalize(wavg.xy), safenormalize(maxvec.xy));
	float confidenceinmax = saturate(1.0 - cosv * 50.0);
	o = lerp(wavg.xy, maxvec.xy, confidenceinmax);
}

void TileDilatePS(in VSOUT i, out float2 o : SV_Target0)
{
	float3 maxvec = 0;
	[unroll]for(int x = -2; x <= 2; x++)
	[unroll]for(int y = -2; y <= 2; y++)
	{
		float2 mv = tex2Doffset(sTileTex, i.uv, int2(x, y)).xy;
		float len2 = dot(mv, mv);
		maxvec = len2 > maxvec.z ? float3(mv, len2) : maxvec;
	}
	o = maxvec.xy;
}

float3 showmotion(float2 motion)
{
	float angle = atan2(motion.y, motion.x);
	float dist = length(motion);
	float3 rgb = saturate(3 * abs(2 * frac(angle / 6.283 + float3(0, -1.0/3.0, 1.0/3.0)) - 1) - 1);
	return lerp(0.5, rgb, saturate(dist * 100));
}

void PSOut(in VSOUT i, out float3 o : SV_Target0)
{	
	//o = 100 * length(tex2D(sMotionTexIntermediate0, i.uv).xy);
	o = showmotion(-tex2D(sTileTexDilated, i.uv).xy);
}

/*=============================================================================
	Techniques
=============================================================================*/

technique MartysMods_Motionblur
<
    ui_label = "iMMERSE Motion Blur";
    ui_tooltip =        
        "                          MartysMods - Motion Blur                            \n"
        "                   MartysMods Epic ReShade Effects (iMMERSE)                  \n"
        "______________________________________________________________________________\n"
        "\n"

       // "Launchpad is a catch-all setup shader that prepares various data for the other\n"
       // "effects. Enable this effect and move it to the top of the effect list.        \n"
        "\n"
        "\n"
        "Visit https://martysmods.com for more information.                            \n"
        "\n"       
        "______________________________________________________________________________";
>
{
    pass {VertexShader = MainVS; PixelShader  = TileDownsample0PS; RenderTarget = TileDownsampleIntermediateTex; }
	pass {VertexShader = MainVS; PixelShader  = TileDownsample1PS; RenderTarget = TileTex; }	
	pass {VertexShader = MainVS; PixelShader  = TileDilatePS;      RenderTarget = TileTexDilated; }	
	pass {VertexShader = MainVS;PixelShader  = PSOut;  }	
	

}