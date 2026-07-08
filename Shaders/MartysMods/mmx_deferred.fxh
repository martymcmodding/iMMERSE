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

namespace Deferred 
{

namespace IPC 
{

//this needs to be DX9 compatible, so I use 1 channel per type of request (multiple can be true).
//To avoid having 3 passes for the requests, I solve this with render target write masks instead.
#define MARTYSMODS_IPC_FEATURE_NORMALS 		1
#define MARTYSMODS_IPC_FEATURE_ALBEDO  		2
#define MARTYSMODS_IPC_FEATURE_OPTICALFLOW  4

texture2D PredicationBuffer  { Format = RGBA8; };	
sampler2D sPredicationBuffer { Texture = PredicationBuffer; };

float4 PredicationVS(in uint id : SV_VertexID) : SV_Position {return float4(0, 0, 0, 1);}
float4 SetPredicationPS(in float4 vpos : SV_Position) : SV_Target0 {return 1;}
float4 ClearPredicationPS(in float4 vpos : SV_Position) : SV_Target0 {return 0;}

#define IPC_REQUEST_FEATURE(_mask) pass IPCRequest{PrimitiveTopology=POINTLIST;VertexCount=1;VertexShader=Deferred::IPC::PredicationVS;PixelShader=Deferred::IPC::SetPredicationPS;  RenderTarget=Deferred::IPC::PredicationBuffer;RenderTargetWriteMask=_mask;}
#define IPC_CLEAR() 			   pass IPCRequest{PrimitiveTopology=POINTLIST;VertexCount=1;VertexShader=Deferred::IPC::PredicationVS;PixelShader=Deferred::IPC::ClearPredicationPS;RenderTarget=Deferred::IPC::PredicationBuffer;}

//not using the defines from global.fxh to avoid having to pull in that dependency early
#if __RENDERER__ >= 0xA000 //DX10
bool is_requested(uint mask)
{
	return dot(tex2Dlod(sPredicationBuffer, 0.0.xxxx), mask & uint4(1,2,4,8)) > 0.5;
}
#else 
bool is_requested(float mask)
{
	float4 sel = frac(floor(mask / float4(1,2,4,8)) * 0.5) > 0.25.xxxx;
	return dot(tex2Dlod(sPredicationBuffer, 0.0.xxxx), sel) > 0.5;
}
#endif
} //namespace IPC

/*=============================================================================*/

//octahedral RG16U
texture GeoNormalsTexV4  { Width = BUFFER_WIDTH_DLSS; Height = BUFFER_HEIGHT_DLSS; Format = RG16; };
sampler sGeoNormalsTexV4 { Texture = GeoNormalsTexV4; MinFilter = POINT; MipFilter = POINT; MagFilter = POINT; };

texture NormalsTexV4     { Width = BUFFER_WIDTH_DLSS; Height = BUFFER_HEIGHT_DLSS; Format = RG16; };
sampler sNormalsTexV4    { Texture = NormalsTexV4;    MinFilter = POINT; MipFilter = POINT; MagFilter = POINT; };

float3 get_normals(float2 uv)
{
    return -Math::octahedral_dec(tex2Dlod(sNormalsTexV4, uv, 0).xy); //fixes bugs in RTGI, positive z gives better precision
}

float3 get_geometry_normals(float2 uv)
{
    return -Math::octahedral_dec(tex2Dlod(sGeoNormalsTexV4, uv, 0).xy);
}

//delta UV
texture MotionVectorsTex { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler sMotionVectorsTex{ Texture = MotionVectorsTex;};

float2 get_motion(float2 uv)
{
    return tex2Dlod(sMotionVectorsTex, uv, 0).xy;
}

//RGB = albedo, A = linear Z (simplifies textured normals generation)
texture AlbedoTex        { Width = BUFFER_WIDTH;      Height = BUFFER_HEIGHT;      Format = RGBA16F; };
sampler sAlbedoTex       { Texture = AlbedoTex; };

float3 get_albedo(float2 uv)
{
    return tex2Dlod(sAlbedoTex, uv, 0).rgb;
}

float3 fetch_albedo(int2 p)
{
    return tex2Dfetch(sAlbedoTex, p).rgb;
}

}