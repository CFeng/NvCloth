/*
* Copyright (c) 2008-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Sample.h"
#include <sstream>

#include <Windows.h>
#include <iostream>
#include <io.h>
#include <fcntl.h>

#include <NvCloth/Fabric.h>
#include <NvCloth/Solver.h>
#include <NvCloth/Cloth.h>
#include <NvCloth/Factory.h>
#include <cuda.h>
#include <NvClothExt/ClothFabricCooker.h>
#include <foundation/PxMat44.h>
#include <foundation/PxTransform.h>
#include <foundation/PxQuat.h>
#include "CallbackImplementations.h"

using namespace std;

struct Triangle
{
    Triangle() {}
    Triangle(uint32_t _a, uint32_t _b, uint32_t _c) :
        a(_a), b(_b), c(_c) {}
    uint32_t a, b, c;

    Triangle operator+(uint32_t offset)const { return Triangle(a + offset, b + offset, c + offset); }
};

void GeneratePlaneCloth(float width, float height, int segmentsX, int segmentsY, physx::PxMat44 transform,
    std::vector<physx::PxVec3>& mVertices, std::vector<Triangle>& mTriangles, std::vector<physx::PxReal>& mInvMasses)
{
/*
GeneratePlaneCloth(x,y,2,2) generates:

    v0______v1_____v2     v0______v1_____v2
     |      |      |       |\     |\     |
     |  Q0  |  Q1  |       |  \t0 |  \t2 |
     |      |      |       | t1 \ | t3 \ |
    v3------v4-----v5     v3-----\v4----\v5
     |      |      |       | \    | \    |
     |  Q2  |  Q3  |       |   \t4|   \t6|
     |______|______|       |_t5_\_|_t7__\|
    v6      v7     v8     v6      v7     v8
*/


	mVertices.clear();
	mTriangles.clear();
	mInvMasses.clear();

	mVertices.resize((segmentsX + 1) * (segmentsY + 1));
	mInvMasses.resize((segmentsX + 1) * (segmentsY + 1));
	mTriangles.resize(segmentsX * segmentsY * 2);

	physx::PxVec3 topLeft(-width * 0.5f, 0.f, -height * 0.5f);

	// Vertices
	for (int y = 0; y < segmentsY + 1; y++)
	{
		for(int x = 0; x < segmentsX + 1; x++)
		{
            physx::PxVec3 pos = physx::PxVec3(((float)x / (float)segmentsX) * width, 0.f, ((float)y / (float)segmentsY) * height);
            mVertices[x + y * (segmentsX + 1)] = transform.transform(topLeft + pos);
            mInvMasses[x + y * (segmentsX + 1)] = 1.0f;
		}
	}

	// Triangles
	for (int y = 0; y < segmentsY; y++)
	{
		for(int x = 0; x < segmentsX; x++)
		{
			if((x^y)&1)
			{
				//Top right to bottom left
				mTriangles[(x + y * segmentsX) * 2 + 0] = Triangle( (uint32_t)(x + 0) + (y + 0) * (segmentsX + 1),
																	(uint32_t)(x + 1) + (y + 0) * (segmentsX + 1),
																	(uint32_t)(x + 0) + (y + 1) * (segmentsX + 1));
																		    	   	      
				mTriangles[(x + y * segmentsX) * 2 + 1] = Triangle( (uint32_t)(x + 1) + (y + 0) * (segmentsX + 1),
																	(uint32_t)(x + 1) + (y + 1) * (segmentsX + 1),
																	(uint32_t)(x + 0) + (y + 1) * (segmentsX + 1));
			}
			else
			{
				//Top left to bottom right
				mTriangles[(x + y * segmentsX) * 2 + 0] = Triangle( (uint32_t)(x + 0) + (y + 0) * (segmentsX + 1),
																	(uint32_t)(x + 1) + (y + 0) * (segmentsX + 1),
																	(uint32_t)(x + 1) + (y + 1) * (segmentsX + 1));
																		    	   	      
				mTriangles[(x + y * segmentsX) * 2 + 1] = Triangle( (uint32_t)(x + 0) + (y + 0) * (segmentsX + 1),
																	(uint32_t)(x + 1) + (y + 1) * (segmentsX + 1),
																	(uint32_t)(x + 0) + (y + 1) * (segmentsX + 1));
			}
		}																    		 
	}
}

template <typename T>
nv::cloth::BoundedData ToBoundedData(T& vector)
{
    nv::cloth::BoundedData d;
    d.data = &vector[0];
    d.stride = sizeof(vector[0]);
    d.count = (physx::PxU32)vector.size();

    return d;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	AllocConsole();
	FILE* fp;
	freopen_s(&fp, "CONOUT$", "w", stdout);
#endif

	nv::cloth::Fabric* mFabric[1];
	nv::cloth::Solver* mSolver;
	//ClothActor* mClothActor[1];

	physx::PxVec3 mOffset;
	//Renderable* sphere;

    //////////////////////////
    NvClothEnvironment::AllocateEnv();

    CUcontext					mCUDAContext;
    cuInit(0);
    int deviceCount = 0;
    if (cuDeviceGetCount(&deviceCount) != CUDA_SUCCESS)
    {
        return 1;
    }

    if (cuCtxCreate(&mCUDAContext, 0, 0) != CUDA_SUCCESS)
    {
        return 1;
    }

    nv::cloth::Factory* factory = NvClothCreateFactoryCUDA(mCUDAContext);

	//////////////////////////
    mSolver = factory->createSolver();
	//trackSolver(mSolver);

    //initializeCloth(0, physx::PxVec3(0.0f, 0.0f, 0.0f));
    physx::PxVec3 offset = physx::PxVec3(0.0f, 0.0f, 0.0f);
    mOffset = offset;

	///////////////////////////////////////////////////////////////////////
	//ClothMeshData clothMesh;
	std::vector<physx::PxVec3>	mVertices;
	std::vector<Triangle>		mTriangles;
	std::vector<physx::PxReal>	mInvMasses;

	int segmentsX = 69, segmentsY = 79;

	physx::PxMat44 transform = physx::PxTransform(physx::PxVec3(0.f, 13.f, 0.f)+ mOffset, physx::PxQuat(0, physx::PxVec3(1.f, 0.f, 0.f)));
	GeneratePlaneCloth(5.f, 6.f, segmentsX, segmentsY, transform, mVertices, mTriangles, mInvMasses);

	//clothMesh.AttachClothPlaneByAngles(69, 79);
	for (int y = 0; y < segmentsY + 1; y++)
	{
		for (int x = 0; x < segmentsX + 1; x++)
		{
			if (y == 0 && (x == 0 || x == segmentsX))
			{
                mInvMasses[x + y * (segmentsX + 1)] = 0.0f;
			}
		}
	}

    //clothMesh.SetInvMasses(0.5f);
    // Doesn't modify attached vertices
    for (int i = 0; i < (int)mInvMasses.size(); ++i)
    {
        if (mInvMasses[i] > 1e-6f)
            mInvMasses[i] = 0.5f;
    }


	//mClothActor[0] = new ClothActor;
	//nv::cloth::ClothMeshDesc meshDesc = clothMesh.GetClothMeshDesc();
    nv::cloth::ClothMeshDesc meshDesc;
    //meshDesc.setToDefault();
    meshDesc.points = ToBoundedData(mVertices);
    meshDesc.triangles = ToBoundedData(mTriangles);
    meshDesc.invMasses = ToBoundedData(mInvMasses);


	nv::cloth::Vector<int32_t>::Type phaseTypeInfo;
	mFabric[0] = NvClothCookFabricFromMesh(factory, meshDesc, physx::PxVec3(0.0f, 0.0f, 1.0f), &phaseTypeInfo, false);
	//trackFabric(mFabric[0]);

	// Initialize start positions and masses for the actual cloth instance
	// (note: the particle/vertex positions do not have to match the mesh description here. Set the positions to the initial shape of this cloth instance)
	std::vector<physx::PxVec4> particlesCopy;
	particlesCopy.resize(mVertices.size());

	physx::PxVec3 clothOffset = transform.getPosition();
	for(int i = 0; i < (int)mVertices.size(); i++)
	{
		// To put attachment point closer to each other
		if(mInvMasses[i] < 1e-6)
			mVertices[i] = (mVertices[i] - clothOffset)*0.95f + clothOffset;

		particlesCopy[i] = physx::PxVec4(mVertices[i], mInvMasses[i]); // w component is 1/mass, or 0.0f for anchored/fixed particles
	}

	// Create the cloth from the initial positions/masses and the fabric
    nv::cloth::Cloth* mCloth = factory->createCloth(nv::cloth::Range<physx::PxVec4>(&particlesCopy[0], &particlesCopy[0] + particlesCopy.size()), *mFabric[0]);
    particlesCopy.clear(); particlesCopy.shrink_to_fit();

	mCloth->setGravity(physx::PxVec3(0.0f, -9.8f, 0.0f));
	mCloth->setDamping(physx::PxVec3(0.1f, 0.1f, 0.1f));

	physx::PxVec4 spheres[1] = {physx::PxVec4(physx::PxVec3(0.f, 10.f, -1.f) + mOffset,1.5)};

	//mClothActor[0]->mCloth->setSpheres(nv::cloth::Range<physx::PxVec4>(spheres, spheres + 1), 0, mClothActor[0]->mCloth->getNumSpheres());
	mCloth->setSpheres(nv::cloth::Range<physx::PxVec4>(spheres, spheres + 1), nv::cloth::Range<physx::PxVec4>(spheres, spheres + 1));
	

	// Setup phase configs
	std::vector<nv::cloth::PhaseConfig> phases(mFabric[0]->getNumPhases());
	for(int i = 0; i < (int)phases.size(); i++)
	{
		phases[i].mPhaseIndex = i;
		phases[i].mStiffness = 1.0f;
		phases[i].mStiffnessMultiplier = 1.0f;
		phases[i].mCompressionLimit = 1.0f;
		phases[i].mStretchLimit = 1.0f;
	}
	mCloth->setPhaseConfig(nv::cloth::Range<nv::cloth::PhaseConfig>(&phases.front(), &phases.back()));

	//trackClothActor(mClothActor[0]);

	// Add the cloth to the solver for simulation
	//addClothToSolver(mClothActor[0], mSolver);
	mSolver->addCloth(mCloth);
	//assert(mClothSolverMap.find(clothActor) == mClothSolverMap.end());
	//mClothSolverMap[clothActor] = solver;


	return 0;
}
