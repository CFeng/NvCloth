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

using namespace std;

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
        //NvClothEnvironment::AllocateEnv();

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
    int 0 = 0;
    physx::PxVec3 offset = physx::PxVec3(0.0f, 0.0f, 0.0f);
    mOffset = offset;

	///////////////////////////////////////////////////////////////////////
	ClothMeshData clothMesh;

	physx::PxMat44 transform = PxTransform(PxVec3(0.f, 13.f, 0.f)+ mOffset, PxQuat(0, PxVec3(1.f, 0.f, 0.f)));
	clothMesh.GeneratePlaneCloth(5.f, 6.f, 69, 79, false, transform);
	clothMesh.AttachClothPlaneByAngles(69, 79);
	clothMesh.SetInvMasses(0.5f);

	mClothActor[0] = new ClothActor;
	nv::cloth::ClothMeshDesc meshDesc = clothMesh.GetClothMeshDesc();

	nv::cloth::Vector<int32_t>::Type phaseTypeInfo;
	mFabric[0] = NvClothCookFabricFromMesh(factory, meshDesc, physx::PxVec3(0.0f, 0.0f, 1.0f), &phaseTypeInfo, false);
	//trackFabric(mFabric[0]);

	// Initialize start positions and masses for the actual cloth instance
	// (note: the particle/vertex positions do not have to match the mesh description here. Set the positions to the initial shape of this cloth instance)
	std::vector<physx::PxVec4> particlesCopy;
	particlesCopy.resize(clothMesh.mVertices.size());

	physx::PxVec3 clothOffset = transform.getPosition();
	for(int i = 0; i < (int)clothMesh.mVertices.size(); i++)
	{
		// To put attachment point closer to each other
		if(clothMesh.mInvMasses[i] < 1e-6)
			clothMesh.mVertices[i] = (clothMesh.mVertices[i] - clothOffset)*0.95f + clothOffset;

		particlesCopy[i] = physx::PxVec4(clothMesh.mVertices[i], clothMesh.mInvMasses[i]); // w component is 1/mass, or 0.0f for anchored/fixed particles
	}

	// Create the cloth from the initial positions/masses and the fabric
	mClothActor[0]->mCloth = factory->createCloth(nv::cloth::Range<physx::PxVec4>(&particlesCopy[0], &particlesCopy[0] + particlesCopy.size()), *mFabric[0]);
	particlesCopy.clear(); particlesCopy.shrink_to_fit();

	mClothActor[0]->mCloth->setGravity(physx::PxVec3(0.0f, -9.8f, 0.0f));
	mClothActor[0]->mCloth->setDamping(physx::PxVec3(0.1f, 0.1f, 0.1f));

	physx::PxVec4 spheres[1] = {physx::PxVec4(physx::PxVec3(0.f, 10.f, -1.f) + mOffset,1.5)};

	//mClothActor[0]->mCloth->setSpheres(nv::cloth::Range<physx::PxVec4>(spheres, spheres + 1), 0, mClothActor[0]->mCloth->getNumSpheres());
	mClothActor[0]->mCloth->setSpheres(nv::cloth::Range<physx::PxVec4>(spheres, spheres + 1), nv::cloth::Range<physx::PxVec4>(spheres, spheres + 1));
	

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
	mClothActor[0]->mCloth->setPhaseConfig(nv::cloth::Range<nv::cloth::PhaseConfig>(&phases.front(), &phases.back()));

	//trackClothActor(mClothActor[0]);

	// Add the cloth to the solver for simulation
	//addClothToSolver(mClothActor[0], mSolver);
	mSolver->addCloth(mClothActor[0]->mCloth);
	//assert(mClothSolverMap.find(clothActor) == mClothSolverMap.end());
	//mClothSolverMap[clothActor] = solver;


	return 0;
}
