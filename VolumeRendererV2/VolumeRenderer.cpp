#include "VolumeRenderer.h"

VolumeRenderer::VolumeRenderer(RTCScene sceneGeometry, Scene* scene) : 
	sceneGeometry(sceneGeometry),
	scene(scene)
{
	//initialize per thread data
	for (int i = 0; i < RenderingSettings::THREAD_COUNT; i++) {
		mt[i] = std::mt19937(rd());
		distribution[i] = std::uniform_real_distribution<double>(0.0, 1.0);
		gaussStandardDist[i] = std::normal_distribution<double>(0.0, 1.0);

		//arrays:
		seedVertices[i] = new vec3[RenderingSettings::MAX_SEGMENT_COUNT];
		perturbedVertices[i] = new vec3[RenderingSettings::MAX_SEGMENT_COUNT];
		seedSegmentLengths[i] = new float[RenderingSettings::MAX_SEGMENT_COUNT];
		seedSegmentLengthSquares[i] = new double[RenderingSettings::MAX_SEGMENT_COUNT];
		estimatorPDFs[i] = new double[RenderingSettings::MAX_SEGMENT_COUNT];
		cumulatedPathTracingPDFs[i] = new double[RenderingSettings::MAX_SEGMENT_COUNT + 1];
		
		successfulPathCounter[i] = 0;
		occludedPathCounter[i] = 0;
		segmentsTooShortCounter[i] = 0;
		successfullyCreatedPaths[i] = 0;
		rejectedSuccessfulPaths[i] = 0;

		pathTracingPaths[i] = new Path();
	}
	frameBuffer = new vec3[RenderingSettings::HEIGHT * RenderingSettings::WIDTH];

	//given the mean cosine, calculate the appropriate sigmaForHg_scaledForMFP:
	float t = MediumParameters::hg_g_F * 10.0f;
	int lower = (int)floorf(t);
	int upper = (int)ceilf(t);
	double misWeight = t - floorf(t);
	//linear interpolation of sigmaPhi if between two values:
	if (upper <= 9) {
		sigmaForHG = mix(Constants::sigmaPhi[lower], Constants::sigmaPhi[upper], misWeight);
	}
	else {
		sigmaForHG = mix(Constants::sigmaPhi[lower], 0.0, misWeight);
	}
}

VolumeRenderer::~VolumeRenderer()
{
	delete[] frameBuffer;
	delete scene;

	//initialize per thread data
	for (int i = 0; i < RenderingSettings::THREAD_COUNT; i++) {
		//delete arrays:
		Path* pathPointer = pathTracingPaths[i];
		delete pathPointer;

		delete[] cumulatedPathTracingPDFs[i];
		delete[] seedVertices[i];
		delete[] perturbedVertices[i];
		delete[] seedSegmentLengths[i];
		delete[] seedSegmentLengthSquares[i];
		delete[] estimatorPDFs[i];
			
	}
}

void VolumeRenderer::renderScene()
{
	//if sequential execution wished, set omp thread count to 1
	if (RenderingSettings::RENDER_PARALLEL) {
		//assert(omp_get_max_threads() >= RenderingSettings::THREAD_COUNT);
		if (omp_get_max_threads() >= RenderingSettings::THREAD_COUNT) {
			omp_set_num_threads(RenderingSettings::THREAD_COUNT);
			cout << "max num omp threads = " << omp_get_max_threads() << endl;
		}
		else {
			cout << "specified THREAD_COUNT = " << RenderingSettings::THREAD_COUNT << " exceeds maximum value of the machine: " << omp_get_max_threads() << endl;
			cout << "please change THREAD_COUNT in Settings.h to a value in range [1, " << omp_get_max_threads() << "]" << endl;
			cout << "exit program by typing a letter..." << endl;
			string s;
			cin >> s;
			exit(EXIT_FAILURE);
		}

	}
	else {
		cout << "sequential execution" << endl;
		omp_set_num_threads(1);
	}


	//choose the integrator:
	switch (RenderingSettings::integrator) {
		case TEST_RENDERING: integrator = &VolumeRenderer::intersectionTestRendering; break;
		case PATH_TRACING_NO_SCATTERING: integrator = &VolumeRenderer::pathTracingNoScattering; break;
		case PATH_TRACING_NEE_MIS_NO_SCATTERING: integrator = &VolumeRenderer::pathTracing_NEE_MIS_NoScattering; break;
		case PATH_TRACING_RANDOM_WALK: integrator = &VolumeRenderer::pathTracingRandomWalk; break;
		case PATH_TRACING_NEE_MIS: integrator = &VolumeRenderer::pathTracing_NEE_MIS; break;
		case PATH_TRACING_MVNEE: integrator = &VolumeRenderer::pathTracing_MVNEE; break;
		case PATH_TRACING_MVNEE_FINAL: integrator = &VolumeRenderer::pathTracing_MVNEE_FINAL; break;
		/*case PATH_TRACING_MVNEE_OPTIMIZED: integrator = &VolumeRenderer::pathTracing_MVNEE_Optimized; break;
		case PATH_TRACING_MVNEE_OPTIMIZED2: integrator = &VolumeRenderer::pathTracing_MVNEE_Optimized2; break;
		case PATH_TRACING_MVNEE_OPTIMIZED3: integrator = &VolumeRenderer::pathTracing_MVNEE_Optimized3; break;*/
		case PATH_TRACING_MVNEE_GAUSS_PERTURB: integrator = &VolumeRenderer::pathTracing_MVNEE_GaussPerturb; break;
		case PATH_TRACING_MVNEE_Constants_ALPHA: integrator = &VolumeRenderer::pathTracing_MVNEE_ConstantsAlpha; break;
	}

	//Start time measurement
	measureStart = std::chrono::system_clock::now();


	float pixelWidth = scene->camera.imagePlaneWidth / (float)RenderingSettings::WIDTH;
	float pixelHeight = scene->camera.imagePlaneHeight / (float)RenderingSettings::HEIGHT;
	vec3 pixelStartPos = scene->camera.cameraOrigin - ((scene->camera.imagePlaneWidth + pixelWidth) / 2.0f) * scene->camera.camRight - ((scene->camera.imagePlaneHeight + pixelHeight) / 2.0f) * scene->camera.camUp + scene->camera.distanceToImagePlane * scene->camera.camLookAt;
	
	const float num_samples = (float)RenderingSettings::SAMPLE_COUNT;
	
	//shoot ray through every pixel
#if defined _WIN32
	#pragma omp parallel for schedule(dynamic)
#else
	#pragma omp parallel for schedule(dynamic) collapse(2)	
#endif
	for (int j = 0; j < RenderingSettings::HEIGHT; j++) {
		for (int i = 0; i < RenderingSettings::WIDTH; i++) {
			int threadID = omp_get_thread_num();
			int frameBufferIndex;

			frameBufferIndex = i + j * RenderingSettings::WIDTH;
			vec3 pixelContribution = vec3(0.0f);

			//pivot position for subpixel sampling
			vec3 currentPixelOrigin = pixelStartPos + (float)i *pixelWidth * scene->camera.camRight + (float)j * pixelHeight * scene->camera.camUp;

			//multiple random walks per pixel:
			for (int sample = 0; sample < RenderingSettings::SAMPLE_COUNT; sample++) {
				//sample position within pixel rectangle:
				float xDiff = (float)sample1D(threadID) * pixelWidth;
				float yDiff = (float)sample1D(threadID) * pixelHeight;

				vec3 rayOrigin = currentPixelOrigin + xDiff * scene->camera.camRight + yDiff * scene->camera.camUp;
				vec3 rayDir = normalize(rayOrigin - scene->camera.cameraOrigin);

				//call integrator function:
				pixelContribution += (*this.*integrator)(rayOrigin, rayDir);

				//if (threadID == 1 && sample % 10000 == 0) {
				//	cout << "thread 1 processing!" << endl;
				//}
			}

			pixelContribution /= num_samples;

			//contribute to pixel
			frameBuffer[frameBufferIndex] = pixelContribution;			
		}
		//cout << "row " << j << " finished" << endl;
	}

	//Stop time measurement
	measureStop = std::chrono::system_clock::now();

	auto durationV = std::chrono::duration_cast<std::chrono::microseconds>(measureStop - measureStart).count();
	double duration = (double)durationV / (60.0 * 1000000.0);

	printRenderingParameters(RenderingSettings::SAMPLE_COUNT, duration);
	printCounters();

	writeBufferToFloatFile(RenderingSettings::sessionName, RenderingSettings::WIDTH, RenderingSettings::HEIGHT, frameBuffer);

#if defined ENABLE_OPEN_EXR
	string openExrFileName = RenderingSettings::sessionName;
	openExrFileName.append(".exr");
	saveBufferToOpenEXR(openExrFileName.c_str(), frameBuffer, RenderingSettings::WIDTH, RenderingSettings::HEIGHT);
#endif

	string tgaFileName = RenderingSettings::sessionName;
	tgaFileName.append(".tga");
	saveBufferToTGA(tgaFileName.c_str(), frameBuffer, RenderingSettings::WIDTH, RenderingSettings::HEIGHT);	

	vec3 mib = calcMeanImageBrightnessBlockwise();
}

/**
* Renders the scene and stops once the maximum duration is exceeded.
*/
void VolumeRenderer::renderSceneWithMaximalDuration(double maxDurationMinutes)
{
	//if sequential execution wished, set omp thread count to 1
	if (RenderingSettings::RENDER_PARALLEL) {
		//assert(omp_get_max_threads() >= RenderingSettings::THREAD_COUNT);
		if (omp_get_max_threads() >= RenderingSettings::THREAD_COUNT) {
			omp_set_num_threads(RenderingSettings::THREAD_COUNT);
			cout << "max num omp threads = " << omp_get_max_threads() << endl;
		}
		else {
			cout << "specified THREAD_COUNT = " << RenderingSettings::THREAD_COUNT << " exceeds maximum value of the machine: " << omp_get_max_threads() << endl;
			cout << "please change THREAD_COUNT in Settings.h to a value in range [1, " << omp_get_max_threads() << "]" << endl;
			cout << "exit program by typing a letter..." << endl;
			string s;
			cin >> s;
			exit(EXIT_FAILURE);
		}

	}
	else {
		cout << "sequential execution" << endl;
		omp_set_num_threads(1);
	}


	//choose the integrator:
	switch (RenderingSettings::integrator) {
		case TEST_RENDERING: integrator = &VolumeRenderer::intersectionTestRendering; break;
		case PATH_TRACING_NO_SCATTERING: integrator = &VolumeRenderer::pathTracingNoScattering; break;
		case PATH_TRACING_NEE_MIS_NO_SCATTERING: integrator = &VolumeRenderer::pathTracing_NEE_MIS_NoScattering; break;
		case PATH_TRACING_RANDOM_WALK: integrator = &VolumeRenderer::pathTracingRandomWalk; break;
		case PATH_TRACING_NEE_MIS: integrator = &VolumeRenderer::pathTracing_NEE_MIS; break;
		case PATH_TRACING_MVNEE: integrator = &VolumeRenderer::pathTracing_MVNEE; break;
		case PATH_TRACING_MVNEE_FINAL: integrator = &VolumeRenderer::pathTracing_MVNEE_FINAL; break;
		/*case PATH_TRACING_MVNEE_OPTIMIZED: integrator = &VolumeRenderer::pathTracing_MVNEE_Optimized; break;
		case PATH_TRACING_MVNEE_OPTIMIZED2: integrator = &VolumeRenderer::pathTracing_MVNEE_Optimized2; break;
		case PATH_TRACING_MVNEE_OPTIMIZED3: integrator = &VolumeRenderer::pathTracing_MVNEE_Optimized3; break;*/
		case PATH_TRACING_MVNEE_GAUSS_PERTURB: integrator = &VolumeRenderer::pathTracing_MVNEE_GaussPerturb; break;
		case PATH_TRACING_MVNEE_Constants_ALPHA: integrator = &VolumeRenderer::pathTracing_MVNEE_ConstantsAlpha; break;
	}	

	float pixelWidth = scene->camera.imagePlaneWidth / (float)RenderingSettings::WIDTH;
	float pixelHeight = scene->camera.imagePlaneHeight / (float)RenderingSettings::HEIGHT;
	vec3 pixelStartPos = scene->camera.cameraOrigin - ((scene->camera.imagePlaneWidth + pixelWidth) / 2.0f) * scene->camera.camRight - ((scene->camera.imagePlaneHeight + pixelHeight) / 2.0f) * scene->camera.camUp + scene->camera.distanceToImagePlane * scene->camera.camLookAt;

	
	//shoot ray through every pixel
	int samples = 0;
	bool timeNotElapsed = true;
	double currentDuration = 0.0;

	while (timeNotElapsed) {
		samples++;

		//Start time measurement
		measureStart = std::chrono::system_clock::now();

		//#pragma omp parallel for schedule(dynamic) collapse(2)
		#pragma omp parallel for schedule(dynamic)
		for (int j = 0; j < RenderingSettings::HEIGHT; j++) {
			for (int i = 0; i < RenderingSettings::WIDTH; i++) {
				int threadID = omp_get_thread_num();
				int frameBufferIndex;

				frameBufferIndex = i + j * RenderingSettings::WIDTH;

				//pivot position for subpixel sampling
				vec3 currentPixelOrigin = pixelStartPos + (float)i *pixelWidth * scene->camera.camRight + (float)j * pixelHeight * scene->camera.camUp;
				
				//sample position within pixel rectangle:
				float xDiff = (float)sample1D(threadID) * pixelWidth;
				float yDiff = (float)sample1D(threadID) * pixelHeight;

				vec3 rayOrigin = currentPixelOrigin + xDiff * scene->camera.camRight + yDiff * scene->camera.camUp;
				vec3 rayDir = normalize(rayOrigin - scene->camera.cameraOrigin);

				//call integrator function:
				vec3 pixelContribution = (*this.*integrator)(rayOrigin, rayDir);

				//contribute to pixel
				frameBuffer[frameBufferIndex] += pixelContribution;
			}			
		}
		//Stop time measurement
		measureStop = std::chrono::system_clock::now();				

		//calc elapsed time
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(measureStop - measureStart).count();
		double frameDuration = (double)duration / (60.0 * 1000000.0);
		currentDuration += frameDuration;

		if (samples % 100 == 0) {
			cout << "frame " << samples << " finished." << endl;
			cout << "elapsed time = " << currentDuration << endl;
		}

		//if maxTime exceeded, stop rendering!
		if (currentDuration >= maxDurationMinutes) {
			timeNotElapsed = false;
		}
	}

	int frameBufferIndex;
	for (int j = 0; j < RenderingSettings::HEIGHT; j++) {
		for (int i = 0; i < RenderingSettings::WIDTH; i++) {
			frameBufferIndex = i + j * RenderingSettings::WIDTH;
			frameBuffer[frameBufferIndex] /= (float)samples;
		}
	}	

	printRenderingParameters(samples, currentDuration);	
	printCounters();

	writeBufferToFloatFile(RenderingSettings::sessionName, RenderingSettings::WIDTH, RenderingSettings::HEIGHT, frameBuffer);

#if defined ENABLE_OPEN_EXR
	string openExrFileName = RenderingSettings::sessionName;
	openExrFileName.append(".exr");
	saveBufferToOpenEXR(openExrFileName.c_str(), frameBuffer, RenderingSettings::WIDTH, RenderingSettings::HEIGHT);
#endif

	string tgaFileName = RenderingSettings::sessionName;
	tgaFileName.append(".tga");
	saveBufferToTGA(tgaFileName.c_str(), frameBuffer, RenderingSettings::WIDTH, RenderingSettings::HEIGHT);

	vec3 mib = calcMeanImageBrightnessBlockwise();
}

vec3 VolumeRenderer::calcMeanImageBrightnessBlockwise()
{
	vec3 meanImageBrightness = vec3(0.0f);

	vec3 meanBrightnessBlocks[RenderingSettings::TILES_SIDE][RenderingSettings::TILES_SIDE];
	int blockWidth = RenderingSettings::WIDTH / RenderingSettings::TILES_SIDE;
	int blockHeight = RenderingSettings::HEIGHT / RenderingSettings::TILES_SIDE;	

	int lastBlockWidth = blockWidth + RenderingSettings::WIDTH % RenderingSettings::TILES_SIDE;
	int lastBlockHeight = blockHeight + RenderingSettings::HEIGHT % RenderingSettings::TILES_SIDE;

	//calculate mean image brightness per block
	for (int y = 0; y < RenderingSettings::TILES_SIDE; y++) {
		for (int x = 0; x < RenderingSettings::TILES_SIDE; x++) {
			vec3 block_mib = vec3(0.0f);
			int startX = x * blockWidth;
			int startY = y * blockHeight;

			int currWidth = blockWidth;
			int currHeight = blockHeight;

			if (x == RenderingSettings::TILES_SIDE - 1) {
				currWidth = lastBlockWidth;
			}
			if (y == RenderingSettings::TILES_SIDE - 1) {
				currHeight = lastBlockHeight;
			}

			float numPixelsPerBlock = (float)(currWidth*currHeight);

			for (int j = 0; j < currHeight; j++) {
				for (int i = 0; i < currWidth; i++) {
					int widthIndex = startX + i;
					/*if (widthIndex >= RenderingSettings::WIDTH) {
						cout << "width index out of bounds!" << endl;
					}*/
					int heightIndex = startY + j;
					/*if (heightIndex >= RenderingSettings::HEIGHT) {
						cout << "height index out of bounds!" << endl;
					}*/
					int index = widthIndex + heightIndex * RenderingSettings::WIDTH;
					vec3 currentPixel = frameBuffer[index];
					block_mib += currentPixel;
					meanImageBrightness += currentPixel;
				}
			}
			meanBrightnessBlocks[x][y] = block_mib / numPixelsPerBlock;
		}
	}

	//entire image brightness
	meanImageBrightness = meanImageBrightness / ((float)(RenderingSettings::WIDTH) * (float)(RenderingSettings::HEIGHT));

	//save textual output
	string blockMIBFileRED = RenderingSettings::sessionName;
	blockMIBFileRED.append("_RED.mib");
	ofstream oR(blockMIBFileRED.c_str(), ios::out);
	for (int j = 0; j < RenderingSettings::TILES_SIDE; j++) {
		for (int i = 0; i < RenderingSettings::TILES_SIDE; i++) {
			oR << meanBrightnessBlocks[i][j].r << ", ";
		}
		oR << endl;
	}
	oR << endl;
	oR << "whole image mib: " << meanImageBrightness.r << endl;
	//close the file
	oR.close();

	string blockMIBFileGREEN = RenderingSettings::sessionName;
	blockMIBFileGREEN.append("_GREEN.mib");
	ofstream oG(blockMIBFileGREEN.c_str(), ios::out);
	for (int j = 0; j < RenderingSettings::TILES_SIDE; j++) {
		for (int i = 0; i < RenderingSettings::TILES_SIDE; i++) {
			oG << meanBrightnessBlocks[i][j].g << ", ";
		}
		oG << endl;
	}
	oG << endl;
	oG << "whole image mib: " << meanImageBrightness.g << endl;
	//close the file
	oG.close();

	string blockMIBFileBLUE = RenderingSettings::sessionName;
	blockMIBFileBLUE.append("_BLUE.mib");
	ofstream oB(blockMIBFileBLUE.c_str(), ios::out);
	for (int j = 0; j < RenderingSettings::TILES_SIDE; j++) {
		for (int i = 0; i < RenderingSettings::TILES_SIDE; i++) {
			oB << meanBrightnessBlocks[i][j].b << ", ";
		}
		oB << endl;
	}
	oB << endl;
	oB << "whole image mib: " << meanImageBrightness.b << endl;
	//close the file
	oB.close();


	return meanImageBrightness;
}

double VolumeRenderer::sample1D(const int threadID)
{
	double xi = distribution[threadID](mt[threadID]);
	//while(xi == 1.0) {
	//	xi = distribution[threadID](mt[threadID]);
	//}
	return xi;
}

double VolumeRenderer::sample1DOpenInterval(const int threadID)
{
	double xi = distribution[threadID](mt[threadID]);
	while (xi == 1.0) { //|| xi == 0.0) {
		xi = distribution[threadID](mt[threadID]);
	}
	return xi;
}

vec3 VolumeRenderer::intersectionTestRendering(const vec3& rayOrigin, const vec3& rayDir)
{
	vec3 finalPixel(0.0f);

	RTCRay ray;
	vec3 intersectionNormal;
	if (scene->intersectScene(rayOrigin, rayDir, ray, intersectionNormal)) {
		vec3 intersectionPos = rayOrigin + ray.tfar * rayDir;

		if (scene->lightSource->lightIntersected(intersectionPos)) {
			finalPixel = vec3(1.0f, 1.0f, 1.0f);
		}
		else {
			finalPixel = vec3(1.0f, 0.0f, 0.0f);
		}
	}

	return finalPixel;
}

vec3 VolumeRenderer::pathTracingNoScattering(const vec3& rayOrigin, const vec3& rayDir)
{
	int threadID = omp_get_thread_num();
	vec3 finalValue(0.0f);
	vec3 measurementContrib(1.0f);
	double pdf = 1.0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;
	int currSegmentLength = 0;

	while (currSegmentLength < RenderingSettings::MAX_SEGMENT_COUNT) {

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal)) {
			//avoid self intersection!!
			if (ray.tfar > Constants::epsilon) {

				vec3 intersectionPos = currPosition + ray.tfar * currDir;

				//check if light was hit:
				if (scene->lightSource->lightIntersected(intersectionPos)) {
					float cos_theta_L = dot(-currDir, scene->lightSource->normal);

					//Normal culling
					if (scene->lightSource->validHitDirection(currDir)) {
						finalValue += (measurementContrib / (float)pdf) * scene->lightSource->Le;
					}

					//stop tracing
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}

				ObjectData surfaceData;
				scene->getObjectData(ray.geomID, &surfaceData);

				//sample new direction of diffuse BRDF:
				vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));
				float cos_theta = dot(newDir, intersectionNormal);
				if (cos_theta <= 0.0f) {
					break;
				}
				assert(cos_theta > 0.0f);

				//update measurementContrib and pdf:				
				measurementContrib *= evalDiffuseBRDF(surfaceData.albedo) * cos_theta;
				pdf *= diffuseBRDFSamplingPDF(intersectionNormal, newDir);

				currPosition = intersectionPos + Constants::epsilon * intersectionNormal;
				currDir = newDir;
				currSegmentLength++;
			}
			else {
				//cout << "self intersection!" << endl;
				break;
			}
		}
		else {
			//no intersection found -> stop tracing
			break;
		}
	}
	vec3 finalPixelValue = vec3((float)finalValue.x, (float)finalValue.y, (float)finalValue.z);
	return finalPixelValue;	
}

vec3 VolumeRenderer::pathTracing_NEE_MIS_NoScattering(const vec3& rayOrigin, const vec3& rayDir)
{
	int threadID = omp_get_thread_num();
	vec3 finalPixel(0.0f);

	vec3 measurementContrib(1.0f);
	double pdf = 1.0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;
	vec3 currNormal = scene->camera.camLookAt;

	double lastDirSamplingPDF = 0.0;

	int currSegmentLength = 0;

	while (currSegmentLength < RenderingSettings::MAX_SEGMENT_COUNT) {

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal)) {
			vec3 intersectionPos = currPosition + ray.tfar * currDir;			

			//check if light was hit:
			if (scene->lightSource->lightIntersected(intersectionPos)) {
				float cos_theta_j = dot(-currDir, scene->lightSource->normal);
				//Normal culling
				if (scene->lightSource->validHitDirection(currDir)) {
					vec3 finalContribution = measurementContrib / (float)pdf;
					assert(isfinite(finalContribution.x));

					double misWeight = 1.0;
					if (currSegmentLength > 0) {
						float G_NEE = cos_theta_j / (ray.tfar * ray.tfar);
						if (isfinite(G_NEE)) {
							double lightSamplingPDF = scene->lightSource->getPositionSamplingPDF();
							misWeight = misPowerWeight(lastDirSamplingPDF * G_NEE, lightSamplingPDF);
						}
						else {
							misWeight = 0.0;
						}
					}
					finalPixel += (float)misWeight * finalContribution * scene->lightSource->Le;
				}
				//stop tracing
				break;
			}

			//surface normal culling:
			if (dot(intersectionNormal, -currDir) <= 0.0f) {
				break;
			}

			//diffuse surface was hit: get objectData
			ObjectData surfaceData;
			scene->getObjectData(ray.geomID, &surfaceData);

			/////////////////////////
			// Next Event Estimation
			/////////////////////////

			vec3 lightPosSample = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));		

			vec3 dirToLight = lightPosSample - intersectionPos;
			float distanceToLight = length(dirToLight);
			dirToLight /= distanceToLight;

			//check visibility:
			float cos_theta_i = dot(dirToLight, intersectionNormal);
			//Normal culling at surface
			if (cos_theta_i > 0.0f) {
				RTCRay shadowRay;
				//if (!scene->surfaceOccluded(intersectionPos, dirToLight, distanceToLight, shadowRay)) {
				if (!scene->surfaceOccluded(intersectionPos + Constants::epsilon * intersectionNormal, dirToLight, distanceToLight, shadowRay)) {
					float cos_theta_j = dot(-dirToLight, scene->lightSource->normal);
					//normal culling at light
					if (scene->lightSource->validHitDirection(dirToLight)) {
		
						double lightSamplingPDF = scene->lightSource->getPositionSamplingPDF();

						//calculate GTerm
						float G_NEE = cos_theta_j / (distanceToLight * distanceToLight);
						assert(G_NEE > 0.0f);
						if (isfinite(G_NEE)) {
							double brdfDirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, dirToLight);
							assert(brdfDirSamplingPDF > 0.0);
							double misWeight = misPowerWeight(lightSamplingPDF, brdfDirSamplingPDF * G_NEE);

							//update Measurement contrib and PDf for this path using NEE:
							vec3 neeMeasurementContrib = measurementContrib * evalDiffuseBRDF(surfaceData.albedo) * cos_theta_i * G_NEE;
							float pdfNEE = (float)(pdf * lightSamplingPDF);

							if (isfinite(pdfNEE)) {
								assert(isfinite(pdfNEE));

								vec3 neeThroughput = neeMeasurementContrib / pdfNEE;
								if (isfinite(neeThroughput.x)) {
									assert(isfinite(neeThroughput.x));
									finalPixel += (float)misWeight * neeThroughput * scene->lightSource->Le;
								}
								else {
									cout << "throughput infinite?" << endl;
								}
							}
							else {
								cout << "PDF overflow?" << endl;
							}
						}
					}
				}
			}
			else {
				//cout << "normal culling at surface" << endl;
			}
			/////////////////////////
			// Sample BRDF
			/////////////////////////

			//sample new direction:
			vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));
			float cos_theta = dot(intersectionNormal, newDir);

			if (cos_theta <= 0.0f) {
				break;
			}
			assert(cos_theta > 0.0f);

			//update PDF and measurement contrib:
			lastDirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, newDir);

			if (lastDirSamplingPDF == 0.0) {
				//invalid sampling!
				break;
			}

			pdf *= lastDirSamplingPDF;	

			measurementContrib *= evalDiffuseBRDF(surfaceData.albedo) * cos_theta;

			currNormal = intersectionNormal;
			currPosition = intersectionPos + Constants::epsilon * intersectionNormal;
			currDir = newDir;
			currSegmentLength++;
		}
		else {
			//no intersection found -> stop tracing
			break;
		}
	}

	return finalPixel;
}

vec3 VolumeRenderer::pathTracingRandomWalk(const vec3& rayOrigin, const vec3& rayDir)
{
	int threadID = omp_get_thread_num();
	vec3 finalValue(0.0f);
	vec3 colorThroughput(1.0f);
	double measurementContrib = 1.0;
	double pdf = 1.0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;
	int currSegmentLength = 0;

	while (currSegmentLength < RenderingSettings::MAX_SEGMENT_COUNT) {

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {				

				//surface interaction
				vec3 intersectionPos = currPosition + ray.tfar * currDir;

				//check if light was hit:
				if (scene->lightSource->lightIntersected(intersectionPos)) {
					float cosThetaLight = dot(-currDir, scene->lightSource->normal);

					//Normal culling at light source
					if (scene->lightSource->validHitDirection(currDir)) {
						double transmittance = exp(-MediumParameters::mu_t * ray.tfar);
						double GLight = (double)cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						assert(isfinite(GLight));

						//update mc and pdf:
						measurementContrib *= transmittance * GLight;
						pdf *= transmittance* GLight;
						assert(isfinite(measurementContrib));
						assert(isfinite(pdf));
						double finalContribution = measurementContrib / pdf;
						assert(isfinite(finalContribution));

						finalValue = (float)finalContribution * colorThroughput * scene->lightSource->Le;
					}
					break;
				}


				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}

				/////////////////////////
				// Sample BRDF
				/////////////////////////
				//diffuse surface was hit: get objectData
				ObjectData surfaceData;
				scene->getObjectData(ray.geomID, &surfaceData);

				//sample new direction:
				vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));
				float cos_theta = dot(intersectionNormal, newDir);
				if (cos_theta <= 0.0f) {
					break;
				}

				//update mc and pdf:
				double transmittance = exp(-MediumParameters::mu_t * ray.tfar);
				colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
				measurementContrib *= transmittance * (double)cos_theta;
				pdf *= transmittance * diffuseBRDFSamplingPDF(intersectionNormal, newDir);

				currPosition = intersectionPos + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//medium interaction
			vec3 nextPosition = currPosition + (float)freePathLength * currDir;

			//sample new direction
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			float cos_theta = dot(currDir, newDir);
			float phase = henyeyGreenstein(cos_theta, MediumParameters::hg_g_F);
			double transmittance = exp(-MediumParameters::mu_t * freePathLength);
			double G = 1.0 / (freePathLength * freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));

			//update mc and pdf:
			measurementContrib *= MediumParameters::mu_s * (double)phase * transmittance * G;
			pdf *= MediumParameters::mu_t * (double)phase * transmittance * G;

			currPosition = nextPosition;
			currDir = newDir;
		}

		currSegmentLength++;
	}
	return finalValue;
}

vec3 VolumeRenderer::pathTracing_NEE_MIS(const vec3& rayOrigin, const vec3& rayDir)
{
	vec3 finalValue(0.0f);
	vec3 colorThroughput(1.0f);
	double measurementContrib = 1.0;
	double pdf = 1.0;

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;
	double lastDirSamplingPDF = 1.0;
	int threadID = omp_get_thread_num();

	int currSegmentLength = 0;

	while (currSegmentLength < RenderingSettings::MAX_SEGMENT_COUNT) {

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {				

				//surface interaction
				vec3 intersectionPos = currPosition + ray.tfar * currDir;

				//check if light was hit:
				if (scene->lightSource->lightIntersected(intersectionPos)) {
					float cosThetaLight = dot(-currDir, scene->lightSource->normal);

					//Normal culling at light source
					if (scene->lightSource->validHitDirection(currDir)) {
						double transmittance = exp(-MediumParameters::mu_t * ray.tfar);
						double GLight = (double)cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						assert(isfinite((float)GLight));

						//update mc and pdf:
						measurementContrib *= transmittance * GLight;
						pdf *= transmittance* GLight;
						assert(isfinite(measurementContrib));
						assert(isfinite(pdf));
						double finalContribution = measurementContrib / pdf;
						assert(isfinite(finalContribution));

						//double pathTracingPDF = lastDirSamplingPDF * GLight;
						//contribute distance sampling!
						double pathTracingPDF = lastDirSamplingPDF * GLight * transmittance;
						double misWeight = 1.0;
						if (currSegmentLength > 0.0) {
							misWeight = misPowerWeight(pathTracingPDF, scene->lightSource->getPositionSamplingPDF());
						}

						finalValue += (float)(misWeight * finalContribution) * colorThroughput * scene->lightSource->Le;
						successfulPathCounter[threadID]++;
					}
					occludedPathCounter[threadID]++;
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					occludedPathCounter[threadID]++;
					break;
				}

				currSegmentLength++;
				if (currSegmentLength >= RenderingSettings::MAX_SEGMENT_COUNT) {
					break;
				}

				//diffuse surface was hit: get objectData
				ObjectData surfaceData;
				scene->getObjectData(ray.geomID, &surfaceData);

				//update transmittance up to current position
				double transmittance = exp(-MediumParameters::mu_t * ray.tfar);
				measurementContrib *= transmittance;
				pdf *= transmittance;

				/////////////////////////
				// NEE 
				/////////////////////////
				vec3 lightSample = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
				vec3 isecToLight = lightSample - intersectionPos;
				float distToLight = length(isecToLight);
				vec3 dirToLight = isecToLight / distToLight;

				float cosTheta = dot(intersectionNormal, dirToLight);
				//normal culling at surface
				if (cosTheta > 0.0f && distToLight > Constants::epsilon) {
					RTCRay shadowRay;
					//if (!scene->surfaceOccluded(intersectionPos, dirToLight, distanceToLight, shadowRay)) {
					if (!scene->surfaceOccluded(intersectionPos + Constants::epsilon * intersectionNormal, dirToLight, distToLight, shadowRay)) {
						double pathTracingPDF = diffuseBRDFSamplingPDF(intersectionNormal, dirToLight);
						float cosThetaLight = dot(-dirToLight, scene->lightSource->normal);
						//normal culling at light
						if (scene->lightSource->validHitDirection(dirToLight)) {
							double GLight = (double)cosThetaLight / ((double)distToLight * (double)distToLight);
							double neeTransmittance = exp(-MediumParameters::mu_t * (double)distToLight);
							//pathTracingPDF *= GLight;

							//contribute distance sampling!
							pathTracingPDF *= GLight * neeTransmittance;

							double misWeight = misPowerWeight(scene->lightSource->getPositionSamplingPDF(), pathTracingPDF);

							vec3 neeColorThroughput = colorThroughput * evalDiffuseBRDF(surfaceData.albedo);
							double neeMeasurementContrib = measurementContrib * (double)cosTheta * neeTransmittance * GLight;
							double neePDF = pdf * scene->lightSource->getPositionSamplingPDF();

							double finalContribution = misWeight * (neeMeasurementContrib / neePDF);
							assert(isfinite(finalContribution));

							finalValue += (float)finalContribution * neeColorThroughput * scene->lightSource->Le;
							successfulPathCounter[threadID]++;
						}
						else {
							occludedPathCounter[threadID]++;
						}
					}
				}
				else {
					occludedPathCounter[threadID]++;
				}
				/////////////////////////
				// Sample BRDF
				/////////////////////////				

				//sample new direction:
				vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));
				float cos_theta = dot(intersectionNormal, newDir);
				if (cos_theta <= 0.0f) {
					occludedPathCounter[threadID]++;
					break;
				}

				//update mc and pdf:
				colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
				measurementContrib *= (double)cos_theta;
				lastDirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, newDir);
				pdf *= lastDirSamplingPDF;

				currPosition = intersectionPos + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				segmentsTooShortCounter[threadID]++;
				break;
			}
		}
		else {
			//medium interaction
			vec3 nextPosition = currPosition + (float)freePathLength * currDir;

			currSegmentLength++;
			if (currSegmentLength >= RenderingSettings::MAX_SEGMENT_COUNT) {
				break;
			}

			//update transmittance up to current position
			double G = 1.0 / (freePathLength * freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			double transmittance = exp(-MediumParameters::mu_t * freePathLength);
			measurementContrib *= transmittance * G;
			pdf *= MediumParameters::mu_t * transmittance * G;

			/////////////////////////
			// NEE in Medium
			/////////////////////////
			vec3 lightSample = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
			vec3 posToLight = lightSample - nextPosition;
			float distToLight = length(posToLight);
			vec3 dirToLight = posToLight / distToLight;

			if (distToLight > Constants::epsilon) {
				RTCRay shadowRay;
				//check visibility:
				if (!scene->surfaceOccluded(nextPosition, dirToLight, distToLight, shadowRay)) {

					float cosTheta = dot(currDir, dirToLight);
					double neePhase = (double)henyeyGreenstein(cosTheta, MediumParameters::hg_g_F);
					double pathTracingPDF = neePhase;
					float cosThetaLight = dot(-dirToLight, scene->lightSource->normal);
					//normal culling at light
					if (scene->lightSource->validHitDirection(dirToLight)) {
						double GLight = cosThetaLight / ((double)distToLight * (double)distToLight);
						double neeTransmittance = exp(-MediumParameters::mu_t * (double)distToLight);
						//pathTracingPDF *= GLight;

						//contribute transmittance:
						pathTracingPDF *= GLight * neeTransmittance;

						double misWeight = misPowerWeight(scene->lightSource->getPositionSamplingPDF(), pathTracingPDF);

						double neeMeasurementContrib = measurementContrib * MediumParameters::mu_s * neePhase * neeTransmittance * GLight;
						double neePDF = pdf * scene->lightSource->getPositionSamplingPDF();
						assert(isfinite(neePDF));
						assert(neePDF > 0.0f);

						double finalContribution = misWeight * (neeMeasurementContrib / neePDF);
						assert(isfinite(finalContribution));

						finalValue += (float)finalContribution * colorThroughput * scene->lightSource->Le;
						successfulPathCounter[threadID]++;
					}
					else {
						occludedPathCounter[threadID]++;
					}
				}
				else {
					occludedPathCounter[threadID]++;
				}
			}
			else {
				segmentsTooShortCounter[threadID]++;
			}
			/////////////////////////
			// sample new direction
			/////////////////////////
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			float cos_theta = dot(currDir, newDir);
			float phase = henyeyGreenstein(cos_theta, MediumParameters::hg_g_F);
			lastDirSamplingPDF = (double)phase;
			
			//update mc and pdf:
			measurementContrib *= MediumParameters::mu_s * (double)phase;
			pdf *= (double)phase;

			currPosition = nextPosition;
			currDir = newDir;
		}
	}
	return finalValue;
}


/**
* Calculates the measurement contribution as well as the path tracing PDF of the given path.
* On top of that, the PDFs of all MVNEE estimators are calculated.
*
* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
* This contribution is weighted by the MIS weight using the estimator PDFs.
*
* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
* @param estimatorIndex index of the estimator that created the path:
*			estimatorIndex = 0: path tracing
*			estimatorIndex > 0: MVNEE starting after i path tracing segments
* @return: the MIS weighted contribution of the path
*/
vec3 VolumeRenderer::calcFinalWeightedContribution(Path* path, int estimatorIndex)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= RenderingSettings::MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];

	double measurementContrib = 1.0;
	double pathTracingPDF = 1.0;

	vec3 colorThroughput = vec3(1.0f, 1.0f, 1.0f);

	tl_cumulatedPathTracingPDFs[0] = 1.0;

	/////////////////////////////////////////////////////////////
	// Find the suffix of the path, that could have been created with MVNEE
	/////////////////////////////////////////////////////////////
	int firstPossibleMVNEEEstimatorIndex = 1;
	//start at vertex before the light vertex and find last surface vertex:
	for (int i = segmentCount - 1; i > 0; i--) {
		if (path->getTypeAt(i) == TYPE_SURFACE) {
			firstPossibleMVNEEEstimatorIndex = i;
			break;
		}
	}

	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//first segment needs special handling, as no phase function is calculated:
	PathVertex firstVertex, secondVertex;
	path->getVertex(0, firstVertex);
	assert(firstVertex.vertexType == TYPE_ORIGIN);
	path->getVertex(1, secondVertex);
	assert(secondVertex.vertexType != TYPE_MVNEE);

	vec3 startDir = secondVertex.vertex - firstVertex.vertex;
	float firstDistance = length(startDir);
	if (firstDistance <= 0.0f) {
		return errorValue;
	}
	startDir = startDir / firstDistance; //normalize startDir

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {		
		//Normal culling
		if (scene->lightSource->validHitDirection(startDir)) {
			return scene->lightSource->Le;
		} 
		return errorValue;		
	}
	
	double firstTransmittance = exp(-mu_t * (double)firstDistance);
	double firstG = 1.0 / ((double)(firstDistance)* (double)(firstDistance));
	assert(isfinite(firstG));
	if (secondVertex.vertexType == TYPE_SURFACE) {
		float cosTheta = dot(secondVertex.surfaceNormal, -startDir);
		if (cosTheta <= 0.0f) {
			cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta << endl;
			return errorValue;
		}
		assert(cosTheta > 0.0f);
		firstG *= (double)cosTheta;

		measurementContrib *= firstTransmittance * firstG;
		pathTracingPDF *= firstTransmittance * firstG;
	}
	else {
		measurementContrib *= firstTransmittance * firstG;
		pathTracingPDF *= mu_t * firstTransmittance * firstG;
	}

	//first path tracing pdf calculated:
	tl_cumulatedPathTracingPDFs[1] = pathTracingPDF;

	//calc measurement contrib and pdf for all remaining vertices:
	vec3 previousDir = startDir;
	for (int i = 2; i < path->getVertexCount(); i++) {
		PathVertex currVert, prevVert;
		path->getVertex(i, currVert);
		path->getVertex(i - 1, prevVert);
		vec3 currentDir = currVert.vertex - prevVert.vertex;
		float currentDistance = length(currentDir);
		//if (currentDistance < Constants::epsilon) {
		//	//very small distance -> maybe this will become a problem later!
		//	cout << "very small distance!" << endl;
		//	return errorValue;
		//}
		if (currentDistance <= 0.0f) {			
			return errorValue;
		}
		currentDir = currentDir / currentDistance; //normalize direction

		double currTransmittance = exp(-mu_t * (double)currentDistance);
		double currG = 1.0 / ((double)(currentDistance) * (double)(currentDistance));		
		assert(isfinite(currG));

		//contribute phase function * mu_s / BRDF depending on vertex
		if (prevVert.vertexType == TYPE_SURFACE) {
			//BRDF direction
			ObjectData surfaceData;
			scene->getObjectData(prevVert.geometryID, &surfaceData);

			float cosThetaBRDF = dot(prevVert.surfaceNormal, currentDir);
			if (cosThetaBRDF <= 0.0f) {
				cout << "wrong outgoing direction at surface!: cosTheta = " << cosThetaBRDF << endl;
				return errorValue;
			}
			assert(cosThetaBRDF > 0.0f);

			colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
			measurementContrib *= (double)cosThetaBRDF;
			pathTracingPDF *= diffuseBRDFSamplingPDF(prevVert.surfaceNormal, currentDir);
		}
		else {
			//phase direction
			float cosThetaPhase = dot(previousDir, currentDir);
			double phase = (double)henyeyGreenstein(cosThetaPhase, MediumParameters::hg_g_F);

			measurementContrib *= mu_s * phase;
			pathTracingPDF *= phase;
		}
				
		if (i == segmentCount) {
			//Light connection segment:
			if (scene->lightSource->validHitDirection(currentDir)) {
				//special treatment when light source was hit:
				float cosLightF = dot(-currentDir, scene->lightSource->normal);
				assert(cosLightF >= 0.0f);
				currG *= (double)cosLightF;

				measurementContrib *= currTransmittance * currG;
				pathTracingPDF *= currTransmittance * currG;
			}
			else {
				cout << "path hits light from backside!" << endl;
				return errorValue;
			}
		}
		//contribute G-Terms, transmittance: take care of medium coefficients!
		else {
			if (currVert.vertexType == TYPE_SURFACE) {
				float cosTheta = dot(currVert.surfaceNormal, -currentDir);
				if (cosTheta <= 0.0f) {
					//cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta<< endl;
					return errorValue;
				}
				assert(cosTheta > 0.0f);
				currG *= (double)cosTheta;

				measurementContrib *= currTransmittance * currG;
				pathTracingPDF *= currTransmittance * currG;
			}
			else {
				measurementContrib *= currTransmittance * currG;
				pathTracingPDF *= mu_t * currTransmittance * currG;
			}			
		}
		
		tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
		previousDir = currentDir;
	}
	assert(isfinite(pathTracingPDF));

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;	

	//set estimator pdfs to 0 for all cases where estimator can not have created this path!
	for (int i = 1; i < firstPossibleMVNEEEstimatorIndex; i++) {
		tl_estimatorPDFs[i] = 0.0;
	}

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path
		vec3 lightVertex = path->getVertexPosition(segmentCount);

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = distanceToLight - lastSeedDistance;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;
			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					vec3 forkToPerturbed = perturbedVertex - forkVertex;

					// calculate sigma for perturbation: first convolution formula:
					double leftSideFactor = 0.0;
					for (int x = 0; x <= s; x++) {
						leftSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
					}
					assert(leftSideFactor > 0.0);
					leftSideFactor = sqrt(leftSideFactor);

					double rightSideFactor = 0.0;
					for (int x = s + 1; x < mvneeSegmentCount; x++) {
						rightSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
					}
					assert(rightSideFactor > 0.0);
					rightSideFactor = sqrt(rightSideFactor);

					double sigma = calcGaussProduct(leftSideFactor * sigmaForHG, rightSideFactor * sigmaForHG);
					double finalGGXAlpha = sigma * 1.637618734; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					float distanceToPrevVertex = tl_seedSegmentLengths[s];
					double segmentCountPDF = mu_t * exp(-mu_t * (double)distanceToPrevVertex); //pdf for seed distance sampling
					double combinedPDF = (segmentCountPDF * uvPerturbPDF);

					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						//contribute sampling for last segment: pdf for sampling distance >= distance to Light!
						perturbationPDF *= exp(-mu_t * (double)lastSeedSegmentLength);

						double finalEstimatorPDF = pathTracingStartPDF * scene->lightSource->getPositionSamplingPDF() * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			const int estimatorCount = segmentCount;
			float misWeight = misPowerWeight(finalPDF, tl_estimatorPDFs, estimatorCount);
			vec3 finalPixel = misWeight * finalContribution * colorThroughput * scene->lightSource->Le;
			return finalPixel;
		}
	}

	return errorValue;
}


/**
* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
*
* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
*/
vec3 VolumeRenderer::pathTracing_MVNEE(const vec3& rayOrigin, const vec3& rayDir)
{
	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	while (pathTracingPath->getSegmentLength() < RenderingSettings::MAX_SEGMENT_COUNT) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {				

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				if (scene->lightSource->lightIntersected(surfaceVertex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////

					//normal culling
					if (scene->lightSource->validHitDirection(currDir)) {
						//estimator Index for Path tracing is 0!
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution(pathTracingPath, 0);
						}
						else {
							finalContribution = scene->lightSource->Le;
						}
						
						finalPixel += finalContribution;
					}
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}

				//sample direction based on BRDF:
				vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addMediumVertex(nextScatteringVertex, TYPE_MEDIUM);

			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= RenderingSettings::MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// MVNEE
		//////////////////////////////		

		//first sample a light vertex
		vec3 lightVertex = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
		vec3 forkToLight = lightVertex - forkVertex.vertex;
		float distanceToLight = length(forkToLight);
		if (distanceToLight > Constants::epsilon) {

			//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
			float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;
			if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			
				vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

				vec3 u, v;
				coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

				//sample the segment lengths for the seed path:
				int mvneeSegmentCount;

				bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, &mvneeSegmentCount, threadID);

				//special case: surface normal culling for only one segment:
				if (validPath && mvneeSegmentCount == 1) {
					if (forkVertex.vertexType == TYPE_SURFACE) {
						float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
						if (cosThetaSurface <= 0.0f) {
							validPath = false;
						}
					}
					//light normal culling!
					if (!scene->lightSource->validHitDirection(omega)) {
						validPath = false;
					}
				}

				if (validPath && mvneeSegmentCount > 0) {
					//consider max segment length:
					if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= RenderingSettings::MAX_SEGMENT_COUNT) {

						//determine seed vertices based on sampled segment lengths:
						vec3 prevVertex = forkVertex.vertex;
						for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
							vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
							tl_seedVertices[s] = newSeedVertex;
							prevVertex = newSeedVertex;
						}
						//add light vertex
						tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

						////////////////////////////
						//  perturb all vertices but the lightDiskVertex:
						////////////////////////////
						vec3 previousPerturbedVertex = forkVertex.vertex;
						
						vec3 seedVertex;
						vec3 perturbedVertex;
						bool validMVNEEPath = true;
						for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
							seedVertex = tl_seedVertices[p];

							//sanity check: distances to start and end may not be <= 0
							float distanceToLineEnd = length(lightVertex - seedVertex);
							float distanceToLineStart = length(seedVertex - forkVertex.vertex);
							if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
								validMVNEEPath = false;
								break;
							}

							// calculate sigma for perturbation: first convolution formula:
							double leftSideFactor = 0.0;
							for (int x = 0; x <= p; x++) {
								leftSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
							}
							leftSideFactor = sqrt(leftSideFactor);
							assert(leftSideFactor > 0.0f);
							double rightSideFactor = 0.0;
							for (int x = p + 1; x < mvneeSegmentCount; x++) {
								rightSideFactor += ((double)tl_seedSegmentLengths[x] * (double)tl_seedSegmentLengths[x]);
							}
							rightSideFactor = sqrt(rightSideFactor);
							assert(rightSideFactor > 0.0f);

							double sigma = calcGaussProduct(leftSideFactor * sigmaForHG, rightSideFactor * sigmaForHG);
							double finalGGXAlpha = sigma * 1.637618734; //conversion with Constants factor

							//perform perturbation using ggx radius and uniform angle in u,v plane 
							perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);


							//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
							if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
								vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
								float firstDistToPerturbed = length(forkToFirstPerturbed);
								//if (firstDistToPerturbed >= Constants::epsilon) {
								if (firstDistToPerturbed > 0.0f) {
									forkToFirstPerturbed /= firstDistToPerturbed;
									float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
									if (cosThetaSurface <= 0.0f) {
										validMVNEEPath = false;
										break;
									}
								}
								else {
									validMVNEEPath = false;
									break;
								}

								previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
							}

							//visibility check:
							float maxT = tl_seedSegmentLengths[p];
							vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

							RTCRay shadowRay;
							if (scene->surfaceOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
								validMVNEEPath = false;
								break;
							}

							//add vertex to path
							pathTracingPath->addMediumVertex(perturbedVertex, TYPE_MVNEE);
							previousPerturbedVertex = perturbedVertex;
						}

						if (validMVNEEPath) {
							vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
							float lastSegmentLength = length(lastSegmentDir);
							if (lastSegmentLength > 0.0f) {
								assert(lastSegmentLength > 0.0f);
								lastSegmentDir /= lastSegmentLength;

								//visibility check: first potential normal culling:
								if (scene->lightSource->validHitDirection(lastSegmentDir)) {

									if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
									}

									//occlusion check:
									RTCRay lastShadowRay;
									if (!scene->surfaceOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
										PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, scene->lightSource->normal);
										pathTracingPath->addVertex(lightVertexStruct);

										//calculate measurement constribution, pdf and mis weight:
										//estimator Index for MVNEE is index of last path tracing vertex
										//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
										vec3 finalContribution = calcFinalWeightedContribution(pathTracingPath, lastPathTracingVertexIndex);
										finalPixel += finalContribution;
									}
								}
							}
							else {
								cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
							}
						}
					}
					//delete all MVNEE vertices from the path construct, so path racing stays correct
					pathTracingPath->cutMVNEEVertices();
				}

			}
		}
	}

	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;
}


/**
* Calculates the measurement contribution as well as the path tracing PDF of the given path.
* On top of that, the PDFs of all MVNEE estimators are calculated.
*
* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
* This contribution is weighted by the MIS weight using the estimator PDFs.
*
* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
* @param estimatorIndex index of the estimator that created the path:
*			estimatorIndex = 0: path tracing
*			estimatorIndex > 0: MVNEE starting after i path tracing segments
* @return: the MIS weighted contribution of the path
*/
vec3 VolumeRenderer::calcFinalWeightedContribution_FINAL(Path* path, int estimatorIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= RenderingSettings::MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		return scene->lightSource->Le;
	}


	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();	

	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;


	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			rejectedSuccessfulPaths[threadID]++;
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				rejectedSuccessfulPaths[threadID]++;
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				ObjectData surfaceData;
				scene->getObjectData(prevVert.geometryID, &surfaceData);

				float cosThetaBRDF = dot(prevVert.surfaceNormal, currentDir);
				if (cosThetaBRDF <= 0.0f) {
					cout << "wrong outgoing direction at surface!: cosTheta = " << cosThetaBRDF << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
				assert(cosThetaBRDF > 0.0f);

				colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
				measurementContrib *= (double)cosThetaBRDF;
				pathTracingPDF *= diffuseBRDFSamplingPDF(prevVert.surfaceNormal, currentDir);
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, MediumParameters::hg_g_F);

				measurementContrib *= mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (scene->lightSource->validHitDirection(currentDir)) {
					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, scene->lightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						//cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta<< endl;
						rejectedSuccessfulPaths[threadID]++;
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	double sigmaForHgSquare = sigmaForHG * sigmaForHG;

	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path
		vec3 lightVertex = path->getVertexPosition(segmentCount);

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;
				tl_seedSegmentLengthSquares[s] = distanceToPrevVertex * distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			float lastSegmentLength = distanceToLight - lastSeedDistance;
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = lastSegmentLength;
			tl_seedSegmentLengthSquares[mvneeSegmentCount - 1] = (double)lastSegmentLength * (double)lastSegmentLength;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;

			double leftSideFactor = 0.0;
			double rightSideFactor = 0.0;
			for (int x = 0; x < mvneeSegmentCount; x++) {
				rightSideFactor += tl_seedSegmentLengthSquares[x];
			}

			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					vec3 forkToPerturbed = perturbedVertex - forkVertex;

					double currSegLengthSqu = tl_seedSegmentLengthSquares[s];
					leftSideFactor += currSegLengthSqu;
					rightSideFactor -= currSegLengthSqu;
					if (rightSideFactor <= 0.0f) {
						validMVNEEPath = false;
						break;
					}

					double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
					double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constants; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					float distanceToPrevVertex = tl_seedSegmentLengths[s];

					double combinedPDF = (mu_t * uvPerturbPDF);
					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						perturbationPDF *= exp(-mu_t * (double)distanceToLight);

						double finalEstimatorPDF = pathTracingStartPDF * scene->lightSource->getPositionSamplingPDF() * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			//const int estimatorCount = segmentCount;
			//float misWeight = misPowerWeight(finalPDF, tl_estimatorPDFs, estimatorCount);

			int mvneeEstimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], mvneeEstimatorCount);

			vec3 finalPixel = misWeight * finalContribution * colorThroughput * scene->lightSource->Le;
			return finalPixel;
		}
	}

	return errorValue;
}


/**
* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
* as well as on surfaces. Creates one path tracing path and multiple MVNEE pathsstarting at the given rayOrigin with the given direction.
* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
*
* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
* MVNEE perturbation is performed using GGX2D sampling in the u-v-plane of the seed vertices.
*/
vec3 VolumeRenderer::pathTracing_MVNEE_FINAL(const vec3& rayOrigin, const vec3& rayDir)
{
	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];

	double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	tl_cumulativePathTracingPDFs[0] = 1.0;

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	double pathTracingPDF = 1.0;
	double measurementContrib = 1.0;
	double newMeasurementContrib = 1.0;
	vec3 colorThroughput(1.0f);
	vec3 newColorThroughput(1.0f);

	//index of first vertex which could serve as a MVNEE estimator starting point
	int firstPossibleMVNEEEstimatorIndex = 1;


	while (pathTracingPath->getSegmentLength() < RenderingSettings::MAX_SEGMENT_COUNT) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				if (scene->lightSource->lightIntersected(surfaceVertex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////

					//normal culling
					if (scene->lightSource->validHitDirection(currDir)) {

						//update contribution
						double transmittance = exp(-mu_t * ray.tfar);
						float cosThetaLight = dot(-currDir, scene->lightSource->normal);
						double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						measurementContrib *= transmittance * G;
						pathTracingPDF *= transmittance * G;
						tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						//estimator Index for Path tracing is 0!
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution_FINAL(pathTracingPath, 0, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						}
						else {
							finalContribution = scene->lightSource->Le;
						}

						finalPixel += finalContribution;
					}
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}


				//update pdf and mc:
				double transmittance = exp(-mu_t * ray.tfar);
				float cosThetaSurface = dot(-currDir, intersectionNormal);
				double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				measurementContrib *= transmittance * G;
				pathTracingPDF *= transmittance * G;
				tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

				//update index for MVNEE start vertex:
				firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();


				//sample direction based on BRDF:
				vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));

				//update pdf, measurement contrib and colorThroughput for BRDF-interaction:
				double dirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, newDir);
				float cos_theta_surface = dot(intersectionNormal, newDir);
				newMeasurementContrib = measurementContrib * (double)cos_theta_surface;
				pathTracingPDF *= dirSamplingPDF;
				ObjectData surfaceData;
				scene->getObjectData(ray.geomID, &surfaceData);
				newColorThroughput = colorThroughput * evalDiffuseBRDF(surfaceData.albedo);

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addMediumVertex(nextScatteringVertex, TYPE_MEDIUM);


			//update pdf and mc:
			double transmittance = exp(-mu_t * freePathLength);
			double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			measurementContrib *= transmittance * G;
			pathTracingPDF *= mu_t * transmittance * G;
			tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			//update pdf and mc for phase function
			float cos_theta_Phase = dot(currDir, newDir);
			double phase = (double)henyeyGreenstein(cos_theta_Phase, MediumParameters::hg_g_F);
			newMeasurementContrib = measurementContrib * mu_s * phase;
			pathTracingPDF *= phase;

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= RenderingSettings::MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// MVNEE
		//////////////////////////////		

		//first sample a light vertex
		vec3 lightVertex = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
		vec3 forkToLight = lightVertex - forkVertex.vertex;
		float distanceToLight = length(forkToLight);
		if (distanceToLight > Constants::epsilon) {

			//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
			float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;
			if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {

				vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

				vec3 u, v;
				coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

				//sample the segment lengths for the seed path:
				int mvneeSegmentCount;

				bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);

				//special case: surface normal culling for only one segment:
				if (validPath && mvneeSegmentCount == 1) {
					if (forkVertex.vertexType == TYPE_SURFACE) {
						float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
						if (cosThetaSurface <= 0.0f) {
							validPath = false;
						}
					}
					//light normal culling!
					if (!scene->lightSource->validHitDirection(omega)) {
						validPath = false;
					}
				}

				if (validPath && mvneeSegmentCount > 0) {
					//consider max segment length:
					if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= RenderingSettings::MAX_SEGMENT_COUNT) {

						//determine seed vertices based on sampled segment lengths:
						vec3 prevVertex = forkVertex.vertex;
						for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
							vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
							tl_seedVertices[s] = newSeedVertex;
							prevVertex = newSeedVertex;
						}
						//add light vertex
						tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

						////////////////////////////
						//  perturb all vertices but the lightDiskVertex:
						////////////////////////////
						vec3 previousPerturbedVertex = forkVertex.vertex;
						double sigmaForHgSquare = sigmaForHG * sigmaForHG;

						vec3 seedVertex;
						vec3 perturbedVertex;
						bool validMVNEEPath = true;

						double leftSideFactor = 0.0;
						double rightSideFactor = 0.0;
						for (int x = 0; x < mvneeSegmentCount; x++) {
							rightSideFactor += tl_seedSegmentLengthSquares[x];
						}

						for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
							seedVertex = tl_seedVertices[p];

							//sanity check: distances to start and end may not be <= 0
							float distanceToLineEnd = length(lightVertex - seedVertex);
							float distanceToLineStart = length(seedVertex - forkVertex.vertex);
							if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
								validMVNEEPath = false;
								break;
							}

							double currSegLengthSqr = tl_seedSegmentLengthSquares[p];
							leftSideFactor += currSegLengthSqr;
							rightSideFactor -= currSegLengthSqr;
							if (rightSideFactor <= 0.0f) {
								validMVNEEPath = false;
								break;
							}
							
							double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
							double finalGGXAlpha = sigma * Constants::GGX_CONVERSION_Constants; //conversion with Constants factor

							//perform perturbation using ggx radius and uniform angle in u,v plane 
							perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);

							//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
							if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
								vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
								float firstDistToPerturbed = length(forkToFirstPerturbed);
								if (firstDistToPerturbed > 0.0f) {
									forkToFirstPerturbed /= firstDistToPerturbed;
									float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
									if (cosThetaSurface <= 0.0f) {
										validMVNEEPath = false;
										break;
									}
								}
								else {
									validMVNEEPath = false;
									break;
								}

								previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
							}

							//visibility check:
							float maxT = tl_seedSegmentLengths[p];
							vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

							RTCRay shadowRay;
							if (scene->surfaceOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
								validMVNEEPath = false;
								break;
							}

							//add vertex to path
							pathTracingPath->addMediumVertex(perturbedVertex, TYPE_MVNEE);
							previousPerturbedVertex = perturbedVertex;
						}

						if (validMVNEEPath) {
							vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
							float lastSegmentLength = length(lastSegmentDir);
							if (lastSegmentLength > 0.0f) {
								assert(lastSegmentLength > 0.0f);
								lastSegmentDir /= lastSegmentLength;

								//visibility check: first potential normal culling:
								if (scene->lightSource->validHitDirection(lastSegmentDir)) {

									if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
									}

									//occlusion check:
									RTCRay lastShadowRay;
									if (!scene->surfaceOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
										PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, scene->lightSource->normal);
										pathTracingPath->addVertex(lightVertexStruct);

										//calculate measurement constribution, pdf and mis weight:
										//estimator Index for MVNEE is index of last path tracing vertex
										//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
										vec3 finalContribution = calcFinalWeightedContribution_FINAL(pathTracingPath, lastPathTracingVertexIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
										finalPixel += finalContribution;
									}
								}
							}
							else {
								//cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
							}
						}
					}
					//delete all MVNEE vertices from the path construct, so path racing stays correct
					pathTracingPath->cutMVNEEVertices();
				}

			}
		}

		measurementContrib = newMeasurementContrib;
		colorThroughput = newColorThroughput;
	}

	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;
}


/**
* Calculates the measurement contribution as well as the path tracing PDF of the given path.
* On top of that, the PDFs of all MVNEE estimators are calculated. THIS IS AN OPTIMIZED VERSION (see "pathTracing_MVNEE_GaussPerturb"):
* many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution. A lot of ppdf and measurement contrib
* calcualtion steps can be avoided, as those are calcualted whilst sampling the path.
*
* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
* This contribution is weighted by the MIS weight using the estimator PDFs.
*
* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
* @param estimatorIndex index of the estimator that created the path:
*			estimatorIndex = 0: path tracing
*			estimatorIndex > 0: MVNEE starting after i path tracing segments
* @param firstPossibleMVNEEEstimatorIndex index of the first vertex that can be used as the starting point for MVNEE
* @param currentMeasurementContrib: measurement contrib of the path from start to the vertex at "estimatorIndex"
* @param currentColorThroughput:  color threoughput of the path from start to the vertex at "estimatorIndex"
* @return: the MIS weighted contribution of the path
*/
vec3 VolumeRenderer::calcFinalWeightedContribution_GaussPerturb(Path* path, const int estimatorIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= RenderingSettings::MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		return scene->lightSource->Le;
	}


	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();

	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;


	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			rejectedSuccessfulPaths[threadID]++;
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				rejectedSuccessfulPaths[threadID]++;
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				ObjectData surfaceData;
				scene->getObjectData(prevVert.geometryID, &surfaceData);

				float cosThetaBRDF = dot(prevVert.surfaceNormal, currentDir);
				if (cosThetaBRDF <= 0.0f) {
					cout << "wrong outgoing direction at surface!: cosTheta = " << cosThetaBRDF << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
				assert(cosThetaBRDF > 0.0f);

				colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
				measurementContrib *= (double)cosThetaBRDF;
				pathTracingPDF *= diffuseBRDFSamplingPDF(prevVert.surfaceNormal, currentDir);
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, MediumParameters::hg_g_F);

				measurementContrib *= mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (scene->lightSource->validHitDirection(currentDir)) {
					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, scene->lightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						//cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta<< endl;
						rejectedSuccessfulPaths[threadID]++;
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	double sigmaForHgSquare = sigmaForHG * sigmaForHG;

	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path
		vec3 lightVertex = path->getVertexPosition(segmentCount);

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;
				tl_seedSegmentLengthSquares[s] = distanceToPrevVertex * distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			float lastSegmentLength = distanceToLight - lastSeedDistance;
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = lastSegmentLength;
			tl_seedSegmentLengthSquares[mvneeSegmentCount - 1] = (double)lastSegmentLength * (double)lastSegmentLength;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;

			double leftSideFactor = 0.0;
			double rightSideFactor = 0.0;
			for (int x = 0; x < mvneeSegmentCount; x++) {
				rightSideFactor += tl_seedSegmentLengthSquares[x];
			}

			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					vec3 forkToPerturbed = perturbedVertex - forkVertex;

					double currSegLengthSqu = tl_seedSegmentLengthSquares[s];
					leftSideFactor += currSegLengthSqu;
					rightSideFactor -= currSegLengthSqu;
					if (rightSideFactor <= 0.0f) {
						validMVNEEPath = false;
						break;
					}

					double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					double uvPerturbPDF = gaussPDF(uDist, sigma) * gaussPDF(vDist, sigma);

					float distanceToPrevVertex = tl_seedSegmentLengths[s];

					double combinedPDF = (mu_t * uvPerturbPDF);
					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						perturbationPDF *= exp(-mu_t * (double)distanceToLight);

						double finalEstimatorPDF = pathTracingStartPDF * scene->lightSource->getPositionSamplingPDF() * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			//const int estimatorCount = segmentCount;
			//float misWeight = misPowerWeight(finalPDF, tl_estimatorPDFs, estimatorCount);

			int mvneeEstimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], mvneeEstimatorCount);

			vec3 finalPixel = misWeight * finalContribution * colorThroughput * scene->lightSource->Le;
			return finalPixel;
		}
	}

	return errorValue;

	/*const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= RenderingSettings::MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	int threadID = omp_get_thread_num();

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		//double contrib = currentMeasurementContrib / cumulatedPathTracingPDFs[threadID][1];
		//return (float)contrib * scene->lightSource->Le;
		return scene->lightSource->Le;
	}


	//////////////////////////
	//get thread local pre-reserved data for this thread:
	//////////////////////////	
	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			rejectedSuccessfulPaths[threadID]++;
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				rejectedSuccessfulPaths[threadID]++;
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				ObjectData surfaceData;
				scene->getObjectData(prevVert.geometryID, &surfaceData);

				float cosThetaBRDF = dot(prevVert.surfaceNormal, currentDir);
				if (cosThetaBRDF <= 0.0f) {
					cout << "wrong outgoing direction at surface!: cosTheta = " << cosThetaBRDF << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
				assert(cosThetaBRDF > 0.0f);

				colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
				measurementContrib *= (double)cosThetaBRDF;
				pathTracingPDF *= diffuseBRDFSamplingPDF(prevVert.surfaceNormal, currentDir);
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, MediumParameters::hg_g_F);

				measurementContrib *= mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (scene->lightSource->validHitDirection(currentDir)) {
					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, scene->lightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						//cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta<< endl;
						rejectedSuccessfulPaths[threadID]++;
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path
		vec3 lightVertex = path->getVertexPosition(segmentCount);

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!	
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengthSquares[s] = (double)(distanceToPrevVertex)* (double)(distanceToPrevVertex);

				lastSeedDistance = seedDistance;
			}
			if (validMVNEEPath) {
				//last segment:
				float lastSegmentLengthL = distanceToLight - lastSeedDistance;
				tl_seedSegmentLengthSquares[mvneeSegmentCount - 1] = (double)(lastSegmentLengthL)* (double)(lastSegmentLengthL);

				/////////////////////////////////
				// Calculate Perturbation PDF
				/////////////////////////////////
				double perturbationPDF = 1.0;

				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);

					// calculate sigma for perturbation: first convolution formula:
					double leftSideFactor = 0.0;
					for (int x = 0; x <= s; x++) {
						leftSideFactor += tl_seedSegmentLengthSquares[x];
					}
					assert(leftSideFactor > 0.0);

					double rightSideFactor = 0.0;
					for (int x = s + 1; x < mvneeSegmentCount; x++) {
						rightSideFactor += tl_seedSegmentLengthSquares[x];
					}
					assert(rightSideFactor > 0.0);

					double sigmaForHgSquare = sigmaForHG * sigmaForHG;
					double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);
					
					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					double uvPerturbPDF = gaussPDF(uDist, sigma) * gaussPDF(vDist, sigma);

					double combinedPDF = (mu_t * uvPerturbPDF);

					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					if (lastSegmentLengthL >= 0.0f) {
						perturbationPDF *= exp(-mu_t * (double)distanceToLight); //exp for entire length at once!

						double finalEstimatorPDF = pathTracingStartPDF * scene->lightSource->getPositionSamplingPDF() * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			const int estimatorCount = segmentCount;

			int mvneeEstimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], mvneeEstimatorCount);
			//float misWeight = misBalanceWeight(finalPDF, tl_estimatorPDFs, estimatorCount);
			//float misWeight = misPowerWeight(finalPDF, tl_estimatorPDFs, estimatorCount);
			vec3 finalPixel = misWeight * finalContribution * colorThroughput * scene->lightSource->Le;

			successfulPathCounter[threadID]++;

			return finalPixel;
		}
		else {
			cout << "finalContrib infinite!" << endl;
		}
	}
	rejectedSuccessfulPaths[threadID]++;
	return errorValue;*/
}

/**
* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
* as well as on surfaces. Creates one path tracing path and multiple MVNEE paths starting at the given rayOrigin with the given direction.
* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
*
* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
* MVNEE perturbation is performed using Gauss 2D sampling in the u-v-plane of the seed vertices.
*
* THIS IS ANOTHER OPTIMIZED VERSION: many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution.
* Further a lot of path tracing pdf and measurement contrib calcualtions are avoided by acumulating pdf and measurement contrib whilst
* sampling the path.
*/
vec3 VolumeRenderer::pathTracing_MVNEE_GaussPerturb(const vec3& rayOrigin, const vec3& rayDir)
{
	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];

	double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	tl_cumulativePathTracingPDFs[0] = 1.0;

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	double pathTracingPDF = 1.0;
	double measurementContrib = 1.0;
	double newMeasurementContrib = 1.0;
	vec3 colorThroughput(1.0f);
	vec3 newColorThroughput(1.0f);

	//index of first vertex which could serve as a MVNEE estimator starting point
	int firstPossibleMVNEEEstimatorIndex = 1;


	while (pathTracingPath->getSegmentLength() < RenderingSettings::MAX_SEGMENT_COUNT) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				if (scene->lightSource->lightIntersected(surfaceVertex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////

					//normal culling
					if (scene->lightSource->validHitDirection(currDir)) {

						//update contribution
						double transmittance = exp(-mu_t * ray.tfar);
						float cosThetaLight = dot(-currDir, scene->lightSource->normal);
						double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						measurementContrib *= transmittance * G;
						pathTracingPDF *= transmittance * G;
						tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						//estimator Index for Path tracing is 0!
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution_GaussPerturb(pathTracingPath, 0, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						}
						else {
							finalContribution = scene->lightSource->Le;
						}

						finalPixel += finalContribution;
					}
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					break;
				}


				//update pdf and mc:
				double transmittance = exp(-mu_t * ray.tfar);
				float cosThetaSurface = dot(-currDir, intersectionNormal);
				double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				measurementContrib *= transmittance * G;
				pathTracingPDF *= transmittance * G;
				tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

				//update index for MVNEE start vertex:
				firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();


				//sample direction based on BRDF:
				vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));

				//update pdf, measurement contrib and colorThroughput for BRDF-interaction:
				double dirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, newDir);
				float cos_theta_surface = dot(intersectionNormal, newDir);
				newMeasurementContrib = measurementContrib * (double)cos_theta_surface;
				pathTracingPDF *= dirSamplingPDF;
				ObjectData surfaceData;
				scene->getObjectData(ray.geomID, &surfaceData);
				newColorThroughput = colorThroughput * evalDiffuseBRDF(surfaceData.albedo);

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addMediumVertex(nextScatteringVertex, TYPE_MEDIUM);


			//update pdf and mc:
			double transmittance = exp(-mu_t * freePathLength);
			double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			measurementContrib *= transmittance * G;
			pathTracingPDF *= mu_t * transmittance * G;
			tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			//update pdf and mc for phase function
			float cos_theta_Phase = dot(currDir, newDir);
			double phase = (double)henyeyGreenstein(cos_theta_Phase, MediumParameters::hg_g_F);
			newMeasurementContrib = measurementContrib * mu_s * phase;
			pathTracingPDF *= phase;

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= RenderingSettings::MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// MVNEE
		//////////////////////////////		

		//first sample a light vertex
		vec3 lightVertex = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
		vec3 forkToLight = lightVertex - forkVertex.vertex;
		float distanceToLight = length(forkToLight);
		if (distanceToLight > Constants::epsilon) {

			//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
			float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;
			if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {

				vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

				vec3 u, v;
				coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

				//sample the segment lengths for the seed path:
				int mvneeSegmentCount;

				bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);

				//special case: surface normal culling for only one segment:
				if (validPath && mvneeSegmentCount == 1) {
					if (forkVertex.vertexType == TYPE_SURFACE) {
						float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
						if (cosThetaSurface <= 0.0f) {
							validPath = false;
						}
					}
					//light normal culling!
					if (!scene->lightSource->validHitDirection(omega)) {
						validPath = false;
					}
				}

				if (validPath && mvneeSegmentCount > 0) {
					//consider max segment length:
					if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= RenderingSettings::MAX_SEGMENT_COUNT) {

						//determine seed vertices based on sampled segment lengths:
						vec3 prevVertex = forkVertex.vertex;
						for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
							vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
							tl_seedVertices[s] = newSeedVertex;
							prevVertex = newSeedVertex;
						}
						//add light vertex
						tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

						////////////////////////////
						//  perturb all vertices but the lightDiskVertex:
						////////////////////////////
						vec3 previousPerturbedVertex = forkVertex.vertex;
						double sigmaForHgSquare = sigmaForHG * sigmaForHG;

						vec3 seedVertex;
						vec3 perturbedVertex;
						bool validMVNEEPath = true;

						double leftSideFactor = 0.0;
						double rightSideFactor = 0.0;
						for (int x = 0; x < mvneeSegmentCount; x++) {
							rightSideFactor += tl_seedSegmentLengthSquares[x];
						}

						for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
							seedVertex = tl_seedVertices[p];

							//sanity check: distances to start and end may not be <= 0
							float distanceToLineEnd = length(lightVertex - seedVertex);
							float distanceToLineStart = length(seedVertex - forkVertex.vertex);
							if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
								validMVNEEPath = false;
								break;
							}

							double currSegLengthSqr = tl_seedSegmentLengthSquares[p];
							leftSideFactor += currSegLengthSqr;
							rightSideFactor -= currSegLengthSqr;
							if (rightSideFactor <= 0.0f) {
								validMVNEEPath = false;
								break;
							}

							double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);

							//perform perturbation using gauss 2D
							perturbVertexGaussian2D(sigma, u, v, seedVertex, &perturbedVertex, threadID);

							//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
							if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
								vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
								float firstDistToPerturbed = length(forkToFirstPerturbed);
								if (firstDistToPerturbed > 0.0f) {
									forkToFirstPerturbed /= firstDistToPerturbed;
									float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
									if (cosThetaSurface <= 0.0f) {
										validMVNEEPath = false;
										break;
									}
								}
								else {
									validMVNEEPath = false;
									break;
								}

								previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
							}

							//visibility check:
							float maxT = tl_seedSegmentLengths[p];
							vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

							RTCRay shadowRay;
							if (scene->surfaceOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
								validMVNEEPath = false;
								break;
							}

							//add vertex to path
							pathTracingPath->addMediumVertex(perturbedVertex, TYPE_MVNEE);
							previousPerturbedVertex = perturbedVertex;
						}

						if (validMVNEEPath) {
							vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
							float lastSegmentLength = length(lastSegmentDir);
							if (lastSegmentLength > 0.0f) {
								assert(lastSegmentLength > 0.0f);
								lastSegmentDir /= lastSegmentLength;

								//visibility check: first potential normal culling:
								if (scene->lightSource->validHitDirection(lastSegmentDir)) {

									if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
									}

									//occlusion check:
									RTCRay lastShadowRay;
									if (!scene->surfaceOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
										PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, scene->lightSource->normal);
										pathTracingPath->addVertex(lightVertexStruct);

										//calculate measurement constribution, pdf and mis weight:
										//estimator Index for MVNEE is index of last path tracing vertex
										//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
										vec3 finalContribution = calcFinalWeightedContribution_GaussPerturb(pathTracingPath, lastPathTracingVertexIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
										finalPixel += finalContribution;
									}
								}
							}
							else {
								//cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
							}
						}
					}
					//delete all MVNEE vertices from the path construct, so path racing stays correct
					pathTracingPath->cutMVNEEVertices();
				}

			}
		}

		measurementContrib = newMeasurementContrib;
		colorThroughput = newColorThroughput;
	}

	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;


	/*
	//get thread local pre-reserverd data for this thread:
	int threadID = omp_get_thread_num();
	Path* pathTracingPath = pathTracingPaths[threadID];
	vec3* tl_seedVertices = seedVertices[threadID];
	vec3* tl_perturbedVertices = perturbedVertices[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];
	double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	tl_cumulativePathTracingPDFs[0] = 1.0;

	//add first Vertex
	PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir);
	pathTracingPath->addVertex(origin);

	vec3 finalPixel(0.0f);

	vec3 currPosition = rayOrigin;
	vec3 currDir = rayDir;

	double pathTracingPDF = 1.0;
	double measurementContrib = 1.0;
	double newMeasurementContrib = 1.0;
	vec3 colorThroughput(1.0f);
	vec3 newColorThroughput(1.0f);

	//index of first vertex which could serve as a MVNEE estimator starting point
	int firstPossibleMVNEEEstimatorIndex = 1;

	while (pathTracingPath->getSegmentLength() < RenderingSettings::MAX_SEGMENT_COUNT) {
		//sanity check assertions:
		assert(pathTracingPath->getVertexCount() > 0);
		PathVertex lastPathVertex;
		pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);
		pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		assert(lastPathVertex.vertexType != TYPE_MVNEE);

		//sample free path Lenth
		double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		//this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		PathVertex forkVertex;

		//intersect scene
		RTCRay ray;
		vec3 intersectionNormal;
		if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			if (ray.tfar > Constants::epsilon) {

				//////////////////////////////
				// Surface interaction
				//////////////////////////////
				vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				forkVertex.vertex = surfaceVertex;
				forkVertex.vertexType = TYPE_SURFACE;
				forkVertex.geometryID = ray.geomID;
				forkVertex.surfaceNormal = intersectionNormal;
				pathTracingPath->addVertex(forkVertex);

				if (scene->lightSource->lightIntersected(surfaceVertex)) {
					//////////////////////////////
					// Light hit
					//////////////////////////////

					//normal culling
					if (scene->lightSource->validHitDirection(currDir)) {
						double transmittance = exp(-mu_t * ray.tfar);
						float cosThetaLight = dot(-currDir, scene->lightSource->normal);
						double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						measurementContrib *= transmittance * G;
						pathTracingPDF *= transmittance * G;
						tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						//estimator Index for Path tracing is 0!
						successfullyCreatedPaths[threadID]++;
						vec3 finalContribution;
						if (pathTracingPath->getSegmentLength() > 1) {
							finalContribution = calcFinalWeightedContribution_GaussPerturb(pathTracingPath, 0, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						}
						else {
							finalContribution = scene->lightSource->Le;
						}

						finalPixel += finalContribution;
					}
					occludedPathCounter[threadID]++;
					break;
				}

				//surface normal culling:
				if (dot(intersectionNormal, -currDir) <= 0.0f) {
					occludedPathCounter[threadID]++;
					break;
				}

				//update pdf and mc:
				double transmittance = exp(-mu_t * ray.tfar);
				float cosThetaSurface = dot(-currDir, intersectionNormal);
				double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				measurementContrib *= transmittance * G;
				pathTracingPDF *= transmittance * G;
				tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

				//update index for MVNEE start vertex:
				firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();

				//sample direction based on BRDF:
				vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));

				//update pdf, measurement contrib and colorThroughput for BRDF-interaction:
				double dirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, newDir);
				float cos_theta_surface = dot(intersectionNormal, newDir);
				newMeasurementContrib = measurementContrib * (double)cos_theta_surface;
				pathTracingPDF *= dirSamplingPDF;
				ObjectData surfaceData;
				scene->getObjectData(ray.geomID, &surfaceData);
				newColorThroughput *= colorThroughput * evalDiffuseBRDF(surfaceData.albedo);

				//update variables:
				currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				currDir = newDir;
			}
			else {
				segmentsTooShortCounter[threadID]++;
				break;
			}
		}
		else {
			//////////////////////////////
			// Medium interaction
			//////////////////////////////
			vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			pathTracingPath->addMediumVertex(nextScatteringVertex, TYPE_MEDIUM);

			//update pdf and mc:
			double transmittance = exp(-mu_t * freePathLength);
			double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			if (!isfinite(G)) {
				break;
			}
			assert(isfinite(G));
			measurementContrib *= transmittance * G;
			pathTracingPDF *= mu_t * transmittance * G;
			tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			//sample direction based on Henyey-Greenstein:
			vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			//update pdf and mc for phase function
			float cos_theta_Phase = dot(currDir, newDir);
			double phase = (double)henyeyGreenstein(cos_theta_Phase, MediumParameters::hg_g_F);
			newMeasurementContrib = measurementContrib * mu_s * phase;
			pathTracingPDF *= phase;

			//update variables:
			forkVertex.vertex = nextScatteringVertex;
			forkVertex.vertexType = TYPE_MEDIUM;
			forkVertex.geometryID = -1;
			forkVertex.surfaceNormal = vec3(0.0f);

			currPosition = nextScatteringVertex;
			currDir = newDir;
		}

		int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		if (lastPathTracingVertexIndex >= RenderingSettings::MAX_SEGMENT_COUNT) {
			break;
		}

		//////////////////////////////
		// MVNEE
		//////////////////////////////		

		//first sample a light vertex
		vec3 lightVertex = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
		vec3 forkToLight = lightVertex - forkVertex.vertex;
		float distanceToLight = length(forkToLight);
		if (distanceToLight > Constants::epsilon) {

			//only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
			float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;
			if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			
				vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

				vec3 u, v;
				coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

				//sample the segment lengths for the seed path:
				int mvneeSegmentCount;
				int maxSegmentCount = RenderingSettings::MAX_SEGMENT_COUNT - pathTracingPath->getSegmentLength();

				bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);
				//sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID, maxSegmentCount);
				//bool validPath = true;

				//special case: normal culling for only one segment:
				if (validPath && mvneeSegmentCount == 1) {
					if (forkVertex.vertexType == TYPE_SURFACE) {
						float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
						if (cosThetaSurface <= 0.0f) {
							validPath = false;
							occludedPathCounter[threadID]++;
						}
					}

					//light normal culling!
					if (!scene->lightSource->validHitDirection(omega)) {
						validPath = false;
						occludedPathCounter[threadID]++;
					}
				}

				if (validPath && mvneeSegmentCount > 0) {
					//consider max segment length:
					if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= RenderingSettings::MAX_SEGMENT_COUNT) {

						//determine seed vertices based on sampled segment lengths:
						vec3 prevVertex = forkVertex.vertex;
						for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
							vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
							tl_seedVertices[s] = newSeedVertex;
							prevVertex = newSeedVertex;
						}
						//add light vertex
						tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

						////////////////////////////
						//  perturb all vertices but the lightDiskVertex:
						////////////////////////////
						vec3 previousPerturbedVertex = forkVertex.vertex;

						vec3 seedVertex;
						vec3 perturbedVertex;
						bool validMVNEEPath = true;
						for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
							seedVertex = tl_seedVertices[p];

							//sanity check: distances to start and end may not be <= 0
							float distanceToLineEnd = length(lightVertex - seedVertex);
							float distanceToLineStart = length(seedVertex - forkVertex.vertex);
							if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
								segmentsTooShortCounter[threadID]++;
								validMVNEEPath = false;
								break;
							}

							// calculate sigma for perturbation: first convolution formula:
							double leftSideFactor = 0.0;
							for (int x = 0; x <= p; x++) {
								leftSideFactor += tl_seedSegmentLengthSquares[x];
							}
							assert(leftSideFactor > 0.0f);
							double rightSideFactor = 0.0;
							for (int x = p + 1; x < mvneeSegmentCount; x++) {
								rightSideFactor += tl_seedSegmentLengthSquares[x];
							}
							assert(rightSideFactor > 0.0f);

							double sigmaForHgSquare = sigmaForHG * sigmaForHG;
							double sigma = calcGaussProductSquaredSigmas(leftSideFactor * sigmaForHgSquare, rightSideFactor * sigmaForHgSquare);

							//perform perturbation using gauss 2D
							perturbVertexGaussian2D(sigma, u, v, seedVertex, &perturbedVertex, threadID);


							//for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
							if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
								vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
								float firstDistToPerturbed = length(forkToFirstPerturbed);
								if (firstDistToPerturbed >= 0.0f) {
									forkToFirstPerturbed /= firstDistToPerturbed;
									float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
									if (cosThetaSurface <= 0.0f) {
										validMVNEEPath = false;
										occludedPathCounter[threadID]++;
										break;
									}
								}
								else {
									validMVNEEPath = false;
									break;
								}

								previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
							}

							//visibility check:
							float maxT = tl_seedSegmentLengths[p];
							vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);
							RTCRay shadowRay;
							if (scene->surfaceOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
								validMVNEEPath = false;
								occludedPathCounter[threadID]++;
								break;
							}

							//add vertex to path
							pathTracingPath->addMediumVertex(perturbedVertex, TYPE_MVNEE);
							previousPerturbedVertex = perturbedVertex;
						}

						if (validMVNEEPath) {
							vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
							float lastSegmentLength = length(lastSegmentDir);
							if (lastSegmentLength > 0.0f) {
								assert(lastSegmentLength > 0.0f);
								lastSegmentDir /= lastSegmentLength;

								//visibility check: first potential normal culling:
								if (scene->lightSource->validHitDirection(lastSegmentDir)) {
									if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
										previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
									}

									//occlusion check:
									RTCRay lastShadowRay;
									if (!scene->surfaceOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
										PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, scene->lightSource->normal);
										pathTracingPath->addVertex(lightVertexStruct);

										//calculate measurement constribution, pdf and mis weight:
										//estimator Index for MVNEE is index of last path tracing vertex
										//make sure the last path tracing vertex index is set correctly in order to use the correct PDF
										successfullyCreatedPaths[threadID]++;
										vec3 finalContribution = calcFinalWeightedContribution_GaussPerturb(pathTracingPath, lastPathTracingVertexIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
										finalPixel += finalContribution;
									}
									else {
										occludedPathCounter[threadID]++;
									}
								}
								else {
									occludedPathCounter[threadID]++;
								}
							}
							else {
								segmentsTooShortCounter[threadID]++;
								cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
							}
						}
					}
					//delete all MVNEE vertices from the path construct, so path tracing stays correct
					pathTracingPath->cutMVNEEVertices();
				}

			}
			else {
				segmentsTooShortCounter[threadID]++;
			}
		}
		measurementContrib = newMeasurementContrib;
		colorThroughput = newColorThroughput;
	}

	//Path sampling finished: now reset Path, so it can be filled again
	pathTracingPath->reset();

	return finalPixel;*/
}

/**
* Calculates the measurement contribution as well as the path tracing PDF of the given path.
* On top of that, the PDFs of all MVNEE estimators are calculated. THIS IS AN OPTIMIZED VERSION (see "pathTracing_MVNEE_ConstantsAlpha"):
* many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution. A lot of pdf and measurement contrib
* calcualtion steps can be avoided, as those are calcualted whilst sampling the path.
*
* As a result, the contribution of the path is calculated using the pdf of the estimator that created the path originally.
* This contribution is weighted by the MIS weight using the estimator PDFs.
*
* For this method, the light is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* @param path containing all vertices of the path from camera to light source, as well as surface information (normal objectID,etc.)
* @param estimatorIndex index of the estimator that created the path:
*			estimatorIndex = 0: path tracing
*			estimatorIndex > 0: MVNEE starting after i path tracing segments
* @param firstPossibleMVNEEEstimatorIndex index of the first vertex that can be used as the starting point for MVNEE
* @param currentMeasurementContrib: measurement contrib of the path from start to the vertex at "estimatorIndex"
* @param currentColorThroughput:  color threoughput of the path from start to the vertex at "estimatorIndex"
* @return: the MIS weighted contribution of the path
*/
vec3 VolumeRenderer::calcFinalWeightedContribution_ConstantsAlpha(Path* path, const int estimatorIndex, const int firstPossibleMVNEEEstimatorIndex, const double& currentMeasurementContrib, const vec3& currentColorThroughput)
{
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= RenderingSettings::MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		return scene->lightSource->Le;
	}


	//get thread local pre-reservers data for this thread:
	int threadID = omp_get_thread_num();

	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];
	double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;


	/////////////////////////////////////////////////////////////
	// calculate measurementContribution as well as path tracing pdfs
	/////////////////////////////////////////////////////////////

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			rejectedSuccessfulPaths[threadID]++;
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				rejectedSuccessfulPaths[threadID]++;
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				ObjectData surfaceData;
				scene->getObjectData(prevVert.geometryID, &surfaceData);

				float cosThetaBRDF = dot(prevVert.surfaceNormal, currentDir);
				if (cosThetaBRDF <= 0.0f) {
					cout << "wrong outgoing direction at surface!: cosTheta = " << cosThetaBRDF << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
				assert(cosThetaBRDF > 0.0f);

				colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
				measurementContrib *= (double)cosThetaBRDF;
				pathTracingPDF *= diffuseBRDFSamplingPDF(prevVert.surfaceNormal, currentDir);
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, MediumParameters::hg_g_F);

				measurementContrib *= mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (scene->lightSource->validHitDirection(currentDir)) {
					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, scene->lightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						//cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta<< endl;
						rejectedSuccessfulPaths[threadID]++;
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////

	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path
		vec3 lightVertex = path->getVertexPosition(segmentCount);

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!			
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);
				tl_seedSegmentLengths[s] = distanceToPrevVertex;
				tl_seedSegmentLengthSquares[s] = distanceToPrevVertex * distanceToPrevVertex;

				lastSeedDistance = seedDistance;
			}
			//last segment:
			float lastSegmentLength = distanceToLight - lastSeedDistance;
			tl_seedSegmentLengths[mvneeSegmentCount - 1] = lastSegmentLength;
			tl_seedSegmentLengthSquares[mvneeSegmentCount - 1] = (double)lastSegmentLength * (double)lastSegmentLength;

			/////////////////////////////////
			// Calculate Perturbation PDF
			/////////////////////////////////
			double perturbationPDF = 1.0;

			if (validMVNEEPath) {
				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);

					double finalGGXAlpha = sigmaForHG * Constants::GGX_CONVERSION_Constants; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					float distanceToPrevVertex = tl_seedSegmentLengths[s];

					double combinedPDF = (mu_t * uvPerturbPDF);
					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					float lastSeedSegmentLength = tl_seedSegmentLengths[mvneeSegmentCount - 1];
					if (lastSeedSegmentLength >= 0.0f) {
						perturbationPDF *= exp(-mu_t * (double)distanceToLight);

						double finalEstimatorPDF = pathTracingStartPDF * scene->lightSource->getPositionSamplingPDF() * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			//const int estimatorCount = segmentCount;
			//float misWeight = misPowerWeight(finalPDF, tl_estimatorPDFs, estimatorCount);

			int mvneeEstimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], mvneeEstimatorCount);

			vec3 finalPixel = misWeight * finalContribution * colorThroughput * scene->lightSource->Le;
			return finalPixel;
		}
	}

	return errorValue;

	/*
	const int segmentCount = path->getSegmentLength();
	assert(path->getVertexCount() >= 2);
	assert(segmentCount <= RenderingSettings::MAX_SEGMENT_COUNT);
	assert(estimatorIndex >= 0);
	assert(estimatorIndex < segmentCount);
	vec3 errorValue(0.0f); //returned in case of errors

	int threadID = omp_get_thread_num();

	//special case: light was immediately hit after 1 segment:
	if (path->getVertexCount() == 2) {
		//double contrib = currentMeasurementContrib / cumulatedPathTracingPDFs[threadID][1];
		//return (float)contrib * scene->lightSource->Le;
		return scene->lightSource->Le;
	}


	//////////////////////////
	//get thread local pre-reserved data for this thread:
	//////////////////////////	
	vec3* tl_seedVertices = seedVertices[threadID];
	double* tl_cumulatedPathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	double* tl_estimatorPDFs = estimatorPDFs[threadID];

	//get measurement contrib and pdf and colorThroughput for the path using path tracing up to the last path tracing vertex:
	double measurementContrib = currentMeasurementContrib;
	double pathTracingPDF = tl_cumulatedPathTracingPDFs[segmentCount];
	vec3 colorThroughput = currentColorThroughput;

	//in case the path wasn't created with path tracing, measurement contrib and pdf have to be calculated for the end of the path
	if (estimatorIndex > 0) {
		pathTracingPDF = tl_cumulatedPathTracingPDFs[estimatorIndex];

		// first direction can potentially be sampled by BRDF!!
		PathVertex vertex1, vertex2;
		path->getVertex(estimatorIndex - 1, vertex1);
		path->getVertex(estimatorIndex, vertex2);

		vec3 previousDir = vertex2.vertex - vertex1.vertex;
		float previousDistance = length(previousDir);
		if (previousDistance <= 0.0f) {
			rejectedSuccessfulPaths[threadID]++;
			return errorValue;
		}
		previousDir /= previousDistance;

		//calculate path tracing PDF and measurement contrib for every remaining vertex:
		PathVertex currVert, prevVert;
		for (int i = estimatorIndex + 1; i < path->getVertexCount(); i++) {
			path->getVertex(i, currVert);
			path->getVertex(i - 1, prevVert);
			vec3 currentDir = currVert.vertex - prevVert.vertex;
			float currentDistance = length(currentDir);
			if (currentDistance <= 0.0f) {
				rejectedSuccessfulPaths[threadID]++;
				return errorValue;
			}
			currentDir = currentDir / currentDistance; //normalize direction

			double currTransmittance = exp(-mu_t * (double)currentDistance);
			double currG = 1.0 / ((double)(currentDistance)* (double)(currentDistance));
			assert(isfinite(currG));

			//contribute phase function * mu_s / BRDF depending on vertex
			if (prevVert.vertexType == TYPE_SURFACE) {
				//BRDF direction
				ObjectData surfaceData;
				scene->getObjectData(prevVert.geometryID, &surfaceData);

				float cosThetaBRDF = dot(prevVert.surfaceNormal, currentDir);
				if (cosThetaBRDF <= 0.0f) {
					cout << "wrong outgoing direction at surface!: cosTheta = " << cosThetaBRDF << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
				assert(cosThetaBRDF > 0.0f);

				colorThroughput *= evalDiffuseBRDF(surfaceData.albedo);
				measurementContrib *= (double)cosThetaBRDF;
				pathTracingPDF *= diffuseBRDFSamplingPDF(prevVert.surfaceNormal, currentDir);
			}
			else {
				//phase direction
				float cosThetaPhase = dot(previousDir, currentDir);
				double phase = (double)henyeyGreenstein(cosThetaPhase, MediumParameters::hg_g_F);

				measurementContrib *= mu_s * phase;
				pathTracingPDF *= phase;
			}

			if (i == segmentCount) {
				//Light connection segment:
				if (scene->lightSource->validHitDirection(currentDir)) {
					//special treatment when light source was hit:
					float cosLightF = dot(-currentDir, scene->lightSource->normal);
					assert(cosLightF >= 0.0f);
					currG *= (double)cosLightF;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					cout << "path hits light from backside!" << endl;
					rejectedSuccessfulPaths[threadID]++;
					return errorValue;
				}
			}
			//contribute G-Terms, transmittance: take care of medium coefficients!
			else {
				if (currVert.vertexType == TYPE_SURFACE) {
					float cosTheta = dot(currVert.surfaceNormal, -currentDir);
					if (cosTheta <= 0.0f) {
						//cout << "Surface cosTheta <= 0.0f: " << setprecision(12) << cosTheta<< endl;
						rejectedSuccessfulPaths[threadID]++;
						return errorValue;
					}
					assert(cosTheta > 0.0f);
					currG *= (double)cosTheta;

					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= currTransmittance * currG;
				}
				else {
					measurementContrib *= currTransmittance * currG;
					pathTracingPDF *= mu_t * currTransmittance * currG;
				}
			}

			tl_cumulatedPathTracingPDFs[i] = pathTracingPDF;
			previousDir = currentDir;
		}
	}

	//set path tracing pdf at index 0
	tl_estimatorPDFs[0] = pathTracingPDF;

	//////////////////////////////////////
	//calculate estimator PDFs:
	//////////////////////////////////////
	for (int e = firstPossibleMVNEEEstimatorIndex; e < segmentCount; e++) {
		double pathTracingStartPDF = tl_cumulatedPathTracingPDFs[e]; //pdf for sampling the first vertices with path tracing
		vec3 forkVertex = path->getVertexPosition(e); //last path tracing vertex and anchor point for the mvnee path
		vec3 lightVertex = path->getVertexPosition(segmentCount);

		vec3 forkToLight = lightVertex - forkVertex;
		float distanceToLight = length(forkToLight);

		float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;

		if (distanceToLight <= 0.0f) {
			tl_estimatorPDFs[e] = 0.0;
		}
		else if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {
			//MVNEE is only valid if expected segment count is smaller than MAX_MVNEE_SEGMENTS!	
			vec3 omega = forkToLight / distanceToLight; //normalize direction

			//segment count of mvnee path, which is also the mvnee vertex count
			int mvneeSegmentCount = segmentCount - e;

			//build tangent frame: (same for every seed vertex)
			vec3 u, v;
			coordinateSystem(omega, u, v);

			/////////////////////////////////
			// recover seed distances and seed vertices:
			/////////////////////////////////
			bool validMVNEEPath = true;
			float lastSeedDistance = 0.0f;
			for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
				vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
				vec3 forkToPerturbed = perturbedVertex - forkVertex;

				//use orthogonal projection to recover the seed distance:
				float seedDistance = dot(forkToPerturbed, omega);

				//now some things have to be checked in order to make sure the path can be created with mvnee:
				//the new Distance from forkVertex has to be larger than the summed distance,
				//also the summedDistance may never be larger than distanceToLight!!
				if (seedDistance <= 0.0f || seedDistance <= lastSeedDistance || seedDistance >= distanceToLight) {
					//path is invalid for mvnee, set estimatorPDF to 0
					tl_estimatorPDFs[e] = 0.0;
					validMVNEEPath = false;
					break;
				}

				assert(seedDistance > 0.0f);
				vec3 seedVertex = forkVertex + seedDistance * omega;
				tl_seedVertices[s] = seedVertex;
				float distanceToPrevVertex = seedDistance - lastSeedDistance;
				assert(distanceToPrevVertex > 0.0f);

				lastSeedDistance = seedDistance;
			}
			if (validMVNEEPath) {
				//last segment:
				float lastSegmentLengthL = distanceToLight - lastSeedDistance;

				/////////////////////////////////
				// Calculate Perturbation PDF
				/////////////////////////////////
				double perturbationPDF = 1.0;

				//uv-perturbation for all MVNEE-vertices but the light vertex!
				for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
					vec3 perturbedVertex = path->getVertexPosition(e + 1 + s);
					
					double finalGGXAlpha = sigmaForHG * 1.637618734; //multiply with Constants for conversion from gauss to ggx

					//update pdfs:
					vec3 seedToPerturbed = perturbedVertex - tl_seedVertices[s];
					float uDist = dot(seedToPerturbed, u);
					float vDist = dot(seedToPerturbed, v);
					float r2 = (uDist*uDist) + (vDist*vDist);
					double uvPerturbPDF = GGX_2D_PDF(r2, finalGGXAlpha); //pdf for u-v-plane perturbation

					double combinedPDF = (mu_t * uvPerturbPDF);

					perturbationPDF *= combinedPDF;
				}

				//set final estimator pdf if everything is valid
				if (validMVNEEPath) {
					if (lastSegmentLengthL >= 0.0f) {
						perturbationPDF *= exp(-mu_t * (double)distanceToLight); //exp for entire length at once!

						double finalEstimatorPDF = pathTracingStartPDF * scene->lightSource->getPositionSamplingPDF() * perturbationPDF;
						if (isfinite(finalEstimatorPDF)) {
							tl_estimatorPDFs[e] = finalEstimatorPDF;
						}
						else {
							cout << "finalEstimatorPDF infinite!" << endl;
							tl_estimatorPDFs[e] = 0.0;
						}
					}
					else {
						tl_estimatorPDFs[e] = 0.0;
					}
				}
				else {
					tl_estimatorPDFs[e] = 0.0;
				}
			}
			else {
				tl_estimatorPDFs[e] = 0.0;
			}
		}
		else {
			tl_estimatorPDFs[e] = 0.0;
		}
	}

	///////////////////////////////
	// calculate MIS weight
	///////////////////////////////
	double finalPDF = tl_estimatorPDFs[estimatorIndex]; //pdf of the estimator that created the path
	if (finalPDF > 0.0) {
		float finalContribution = (float)(measurementContrib / finalPDF);
		if (isfinite(finalContribution)) {
			assert(finalContribution >= 0.0);

			const int estimatorCount = segmentCount;

			int mvneeEstimatorCount = segmentCount - firstPossibleMVNEEEstimatorIndex;
			float misWeight = misBalanceWeight(finalPDF, pathTracingPDF, &tl_estimatorPDFs[firstPossibleMVNEEEstimatorIndex], mvneeEstimatorCount);
			//float misWeight = misBalanceWeight(finalPDF, tl_estimatorPDFs, estimatorCount);
			//float misWeight = misPowerWeight(finalPDF, tl_estimatorPDFs, estimatorCount);
			vec3 finalPixel = misWeight * finalContribution * colorThroughput * scene->lightSource->Le;

			successfulPathCounter[threadID]++;

			return finalPixel;
		}
		else {
			cout << "finalContrib infinite!" << endl;
		}
	}
	rejectedSuccessfulPaths[threadID]++;
	return errorValue;*/
}

/**
* Combination of path tracing with Multiple Vertex Next Event Estimation (MVNEE) for direct lighting calculation at vertices in the medium,
* as well as on surfaces. Creates one path tracing path and multiple MVNEE paths starting at the given rayOrigin with the given direction.
* This function returns the MIS weighted summed contributions of all created paths from the given origin to the light source.
*
* The light in this integrator is expected to be an area light, also all objects in the scene are expected to have a diffuse lambertian BRDF.
*
* As MVNEE seed paths, a line connection is used. Seed distances are sampled using transmittance distance sampling.
* MVNEE perturbation is performed using 2D GGX PDF sampling with a cosntant alpha-> so no calculation of position dependent perturbation is performed!
*
* THIS IS ANOTHER OPTIMIZED VERSION: many unnecessary loop iterations are avoided, further small tweaks are used in order to speed up execution.
* Further a lot of path tracing pdf and measurement contrib calcualtions are avoided by acumulating pdf and measurement contrib whilst
* sampling the path.
*/
 vec3 VolumeRenderer::pathTracing_MVNEE_ConstantsAlpha(const vec3& rayOrigin, const vec3& rayDir)
 {
	 //get thread local pre-reservers data for this thread:
	 int threadID = omp_get_thread_num();
	 Path* pathTracingPath = pathTracingPaths[threadID];
	 vec3* tl_seedVertices = seedVertices[threadID];
	 vec3* tl_perturbedVertices = perturbedVertices[threadID];
	 float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	 double* tl_seedSegmentLengthSquares = seedSegmentLengthSquares[threadID];

	 double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	 tl_cumulativePathTracingPDFs[0] = 1.0;

	 //add first Vertex
	 PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir);
	 pathTracingPath->addVertex(origin);

	 vec3 finalPixel(0.0f);

	 vec3 currPosition = rayOrigin;
	 vec3 currDir = rayDir;

	 double pathTracingPDF = 1.0;
	 double measurementContrib = 1.0;
	 double newMeasurementContrib = 1.0;
	 vec3 colorThroughput(1.0f);
	 vec3 newColorThroughput(1.0f);

	 //index of first vertex which could serve as a MVNEE estimator starting point
	 int firstPossibleMVNEEEstimatorIndex = 1;


	 while (pathTracingPath->getSegmentLength() < RenderingSettings::MAX_SEGMENT_COUNT) {
		 //sanity check assertions:
		 assert(pathTracingPath->getVertexCount() > 0);
		 PathVertex lastPathVertex;
		 pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		 assert(lastPathVertex.vertexType != TYPE_MVNEE);
		 pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		 assert(lastPathVertex.vertexType != TYPE_MVNEE);

		 //sample free path Lenth
		 double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		 //this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		 PathVertex forkVertex;

		 //intersect scene
		 RTCRay ray;
		 vec3 intersectionNormal;
		 if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			 if (ray.tfar > Constants::epsilon) {

				 //////////////////////////////
				 // Surface interaction
				 //////////////////////////////
				 vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				 forkVertex.vertex = surfaceVertex;
				 forkVertex.vertexType = TYPE_SURFACE;
				 forkVertex.geometryID = ray.geomID;
				 forkVertex.surfaceNormal = intersectionNormal;
				 pathTracingPath->addVertex(forkVertex);

				 if (scene->lightSource->lightIntersected(surfaceVertex)) {
					 //////////////////////////////
					 // Light hit
					 //////////////////////////////

					 //normal culling
					 if (scene->lightSource->validHitDirection(currDir)) {

						 //update contribution
						 double transmittance = exp(-mu_t * ray.tfar);
						 float cosThetaLight = dot(-currDir, scene->lightSource->normal);
						 double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						 measurementContrib *= transmittance * G;
						 pathTracingPDF *= transmittance * G;
						 tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						 //estimator Index for Path tracing is 0!
						 vec3 finalContribution;
						 if (pathTracingPath->getSegmentLength() > 1) {
							 finalContribution = calcFinalWeightedContribution_ConstantsAlpha(pathTracingPath, 0, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						 }
						 else {
							 finalContribution = scene->lightSource->Le;
						 }

						 finalPixel += finalContribution;
					 }
					 break;
				 }

				 //surface normal culling:
				 if (dot(intersectionNormal, -currDir) <= 0.0f) {
					 break;
				 }


				 //update pdf and mc:
				 double transmittance = exp(-mu_t * ray.tfar);
				 float cosThetaSurface = dot(-currDir, intersectionNormal);
				 double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				 measurementContrib *= transmittance * G;
				 pathTracingPDF *= transmittance * G;
				 tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

				 //update index for MVNEE start vertex:
				 firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();


				 //sample direction based on BRDF:
				 vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));

				 //update pdf, measurement contrib and colorThroughput for BRDF-interaction:
				 double dirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, newDir);
				 float cos_theta_surface = dot(intersectionNormal, newDir);
				 newMeasurementContrib = measurementContrib * (double)cos_theta_surface;
				 pathTracingPDF *= dirSamplingPDF;
				 ObjectData surfaceData;
				 scene->getObjectData(ray.geomID, &surfaceData);
				 newColorThroughput = colorThroughput * evalDiffuseBRDF(surfaceData.albedo);

				 //update variables:
				 currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				 currDir = newDir;
			 }
			 else {
				 break;
			 }
		 }
		 else {
			 //////////////////////////////
			 // Medium interaction
			 //////////////////////////////
			 vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			 pathTracingPath->addMediumVertex(nextScatteringVertex, TYPE_MEDIUM);


			 //update pdf and mc:
			 double transmittance = exp(-mu_t * freePathLength);
			 double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			 if (!isfinite(G)) {
				 break;
			 }
			 assert(isfinite(G));
			 measurementContrib *= transmittance * G;
			 pathTracingPDF *= mu_t * transmittance * G;
			 tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			 //sample direction based on Henyey-Greenstein:
			 vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			 //update pdf and mc for phase function
			 float cos_theta_Phase = dot(currDir, newDir);
			 double phase = (double)henyeyGreenstein(cos_theta_Phase, MediumParameters::hg_g_F);
			 newMeasurementContrib = measurementContrib * mu_s * phase;
			 pathTracingPDF *= phase;

			 //update variables:
			 forkVertex.vertex = nextScatteringVertex;
			 forkVertex.vertexType = TYPE_MEDIUM;
			 forkVertex.geometryID = -1;
			 forkVertex.surfaceNormal = vec3(0.0f);

			 currPosition = nextScatteringVertex;
			 currDir = newDir;
		 }

		 int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		 if (lastPathTracingVertexIndex >= RenderingSettings::MAX_SEGMENT_COUNT) {
			 break;
		 }

		 //////////////////////////////
		 // MVNEE
		 //////////////////////////////		

		 //first sample a light vertex
		 vec3 lightVertex = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
		 vec3 forkToLight = lightVertex - forkVertex.vertex;
		 float distanceToLight = length(forkToLight);
		 if (distanceToLight > Constants::epsilon) {

			 //only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
			 float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;
			 if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {

				 vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

				 vec3 u, v;
				 coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

				 //sample the segment lengths for the seed path:
				 int mvneeSegmentCount;

				 bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, tl_seedSegmentLengthSquares, &mvneeSegmentCount, threadID);

				 //special case: surface normal culling for only one segment:
				 if (validPath && mvneeSegmentCount == 1) {
					 if (forkVertex.vertexType == TYPE_SURFACE) {
						 float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
						 if (cosThetaSurface <= 0.0f) {
							 validPath = false;
						 }
					 }
					 //light normal culling!
					 if (!scene->lightSource->validHitDirection(omega)) {
						 validPath = false;
					 }
				 }

				 if (validPath && mvneeSegmentCount > 0) {
					 //consider max segment length:
					 if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= RenderingSettings::MAX_SEGMENT_COUNT) {

						 //determine seed vertices based on sampled segment lengths:
						 vec3 prevVertex = forkVertex.vertex;
						 for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
							 vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
							 tl_seedVertices[s] = newSeedVertex;
							 prevVertex = newSeedVertex;
						 }
						 //add light vertex
						 tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

						 ////////////////////////////
						 //  perturb all vertices but the lightDiskVertex:
						 ////////////////////////////
						 vec3 previousPerturbedVertex = forkVertex.vertex;

						 vec3 seedVertex;
						 vec3 perturbedVertex;
						 bool validMVNEEPath = true;

						 for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
							 seedVertex = tl_seedVertices[p];

							 //sanity check: distances to start and end may not be <= 0
							 float distanceToLineEnd = length(lightVertex - seedVertex);
							 float distanceToLineStart = length(seedVertex - forkVertex.vertex);
							 if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
								 validMVNEEPath = false;
								 break;
							 }

							 double finalGGXAlpha = sigmaForHG * Constants::GGX_CONVERSION_Constants; //conversion with Constants factor

							 //perform perturbation using ggx radius and uniform angle in u,v plane 
							 perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);

							 //for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
							 if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
								 vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
								 float firstDistToPerturbed = length(forkToFirstPerturbed);
								 if (firstDistToPerturbed > 0.0f) {
									 forkToFirstPerturbed /= firstDistToPerturbed;
									 float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
									 if (cosThetaSurface <= 0.0f) {
										 validMVNEEPath = false;
										 break;
									 }
								 }
								 else {
									 validMVNEEPath = false;
									 break;
								 }

								 previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
							 }

							 //visibility check:
							 float maxT = tl_seedSegmentLengths[p];
							 vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);

							 RTCRay shadowRay;
							 if (scene->surfaceOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
								 validMVNEEPath = false;
								 break;
							 }

							 //add vertex to path
							 pathTracingPath->addMediumVertex(perturbedVertex, TYPE_MVNEE);
							 previousPerturbedVertex = perturbedVertex;
						 }

						 if (validMVNEEPath) {
							 vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
							 float lastSegmentLength = length(lastSegmentDir);
							 if (lastSegmentLength > 0.0f) {
								 assert(lastSegmentLength > 0.0f);
								 lastSegmentDir /= lastSegmentLength;

								 //visibility check: first potential normal culling:
								 if (scene->lightSource->validHitDirection(lastSegmentDir)) {

									 if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
										 previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
									 }

									 //occlusion check:
									 RTCRay lastShadowRay;
									 if (!scene->surfaceOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
										 PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, scene->lightSource->normal);
										 pathTracingPath->addVertex(lightVertexStruct);

										 //calculate measurement constribution, pdf and mis weight:
										 //estimator Index for MVNEE is index of last path tracing vertex
										 //make sure the last path tracing vertex index is set correctly in order to use the correct PDF
										 vec3 finalContribution = calcFinalWeightedContribution_ConstantsAlpha(pathTracingPath, lastPathTracingVertexIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
										 finalPixel += finalContribution;
									 }
								 }
							 }
							 else {
								 //cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
							 }
						 }
					 }
					 //delete all MVNEE vertices from the path construct, so path racing stays correct
					 pathTracingPath->cutMVNEEVertices();
				 }

			 }
		 }

		 measurementContrib = newMeasurementContrib;
		 colorThroughput = newColorThroughput;
	 }

	 //Path sampling finished: now reset Path, so it can be filled again
	 pathTracingPath->reset();

	 return finalPixel;

	 /*
	 //get thread local pre-reserverd data for this thread:
	 int threadID = omp_get_thread_num();
	 Path* pathTracingPath = pathTracingPaths[threadID];
	 vec3* tl_seedVertices = seedVertices[threadID];
	 vec3* tl_perturbedVertices = perturbedVertices[threadID];
	 float* tl_seedSegmentLengths = seedSegmentLengths[threadID];
	 double* tl_cumulativePathTracingPDFs = cumulatedPathTracingPDFs[threadID];
	 tl_cumulativePathTracingPDFs[0] = 1.0;

	 //add first Vertex
	 PathVertex origin(rayOrigin, TYPE_ORIGIN, -1, rayDir);
	 pathTracingPath->addVertex(origin);

	 vec3 finalPixel(0.0f);

	 vec3 currPosition = rayOrigin;
	 vec3 currDir = rayDir;

	 double pathTracingPDF = 1.0;
	 double measurementContrib = 1.0;
	 double newMeasurementContrib = 1.0;
	 vec3 colorThroughput(1.0f);
	 vec3 newColorThroughput(1.0f);

	 //index of first vertex which could serve as a MVNEE estimator starting point
	 int firstPossibleMVNEEEstimatorIndex = 1;

	 while (pathTracingPath->getSegmentLength() < RenderingSettings::MAX_SEGMENT_COUNT) {
		 //sanity check assertions:
		 assert(pathTracingPath->getVertexCount() > 0);
		 PathVertex lastPathVertex;
		 pathTracingPath->getVertex(pathTracingPath->getSegmentLength(), lastPathVertex);
		 assert(lastPathVertex.vertexType != TYPE_MVNEE);
		 pathTracingPath->getVertex(pathTracingPath->getVertexCount() - 1, lastPathVertex);
		 assert(lastPathVertex.vertexType != TYPE_MVNEE);

		 //sample free path Lenth
		 double freePathLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);

		 //this is the starting vertex for MVNEE, depending on the intersection, it can either be a surface vertex or medium vertex
		 PathVertex forkVertex;

		 //intersect scene
		 RTCRay ray;
		 vec3 intersectionNormal;
		 if (scene->intersectScene(currPosition, currDir, ray, intersectionNormal) && ray.tfar <= freePathLength) {
			 if (ray.tfar > Constants::epsilon) {

				 //////////////////////////////
				 // Surface interaction
				 //////////////////////////////
				 vec3 surfaceVertex = currPosition + ray.tfar * currDir;
				 forkVertex.vertex = surfaceVertex;
				 forkVertex.vertexType = TYPE_SURFACE;
				 forkVertex.geometryID = ray.geomID;
				 forkVertex.surfaceNormal = intersectionNormal;
				 pathTracingPath->addVertex(forkVertex);

				 if (scene->lightSource->lightIntersected(surfaceVertex)) {
					 //////////////////////////////
					 // Light hit
					 //////////////////////////////

					 //normal culling
					 if (scene->lightSource->validHitDirection(currDir)) {
						 double transmittance = exp(-mu_t * ray.tfar);
						 float cosThetaLight = dot(-currDir, scene->lightSource->normal);
						 double G = cosThetaLight / ((double)ray.tfar * (double)ray.tfar);
						 measurementContrib *= transmittance * G;
						 pathTracingPDF *= transmittance * G;
						 tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

						 //estimator Index for Path tracing is 0!
						 successfullyCreatedPaths[threadID]++;
						 vec3 finalContribution;
						 if (pathTracingPath->getSegmentLength() > 1) {
							 finalContribution = calcFinalWeightedContribution_ConstantsAlpha(pathTracingPath, 0, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
						 }
						 else {
							 finalContribution = scene->lightSource->Le;
						 }

						 finalPixel += finalContribution;
					 }
					 occludedPathCounter[threadID]++;
					 break;
				 }

				 //surface normal culling:
				 if (dot(intersectionNormal, -currDir) <= 0.0f) {
					 occludedPathCounter[threadID]++;
					 break;
				 }

				 //update pdf and mc:
				 double transmittance = exp(-mu_t * ray.tfar);
				 float cosThetaSurface = dot(-currDir, intersectionNormal);
				 double G = cosThetaSurface / ((double)ray.tfar * (double)ray.tfar);
				 measurementContrib *= transmittance * G;
				 pathTracingPDF *= transmittance * G;
				 tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;

				 //update index for MVNEE start vertex:
				 firstPossibleMVNEEEstimatorIndex = pathTracingPath->getSegmentLength();

				 //sample direction based on BRDF:
				 vec3 newDir = sampleDiffuseBRDFDir(intersectionNormal, sample1D(threadID), sample1D(threadID));

				 //update pdf, measurement contrib and colorThroughput for BRDF-interaction:
				 double dirSamplingPDF = diffuseBRDFSamplingPDF(intersectionNormal, newDir);
				 float cos_theta_surface = dot(intersectionNormal, newDir);
				 newMeasurementContrib = measurementContrib * (double)cos_theta_surface;
				 pathTracingPDF *= dirSamplingPDF;
				 ObjectData surfaceData;
				 scene->getObjectData(ray.geomID, &surfaceData);
				 newColorThroughput *= colorThroughput * evalDiffuseBRDF(surfaceData.albedo);

				 //update variables:
				 currPosition = surfaceVertex + Constants::epsilon * intersectionNormal;
				 currDir = newDir;
			 }
			 else {
				 segmentsTooShortCounter[threadID]++;
				 break;
			 }
		 }
		 else {
			 //////////////////////////////
			 // Medium interaction
			 //////////////////////////////
			 vec3 nextScatteringVertex = currPosition + (float)freePathLength * currDir;
			 pathTracingPath->addMediumVertex(nextScatteringVertex, TYPE_MEDIUM);

			 //update pdf and mc:
			 double transmittance = exp(-mu_t * freePathLength);
			 double G = 1.0 / ((double)freePathLength * (double)freePathLength);
			 if (!isfinite(G)) {
				 break;
			 }
			 assert(isfinite(G));
			 measurementContrib *= transmittance * G;
			 pathTracingPDF *= mu_t * transmittance * G;
			 tl_cumulativePathTracingPDFs[pathTracingPath->getSegmentLength()] = pathTracingPDF;


			 //sample direction based on Henyey-Greenstein:
			 vec3 newDir = sampleHenyeyGreensteinDirection(currDir, sample1D(threadID), sample1D(threadID), MediumParameters::hg_g_F);

			 //update pdf and mc for phase function
			 float cos_theta_Phase = dot(currDir, newDir);
			 double phase = (double)henyeyGreenstein(cos_theta_Phase, MediumParameters::hg_g_F);
			 newMeasurementContrib = measurementContrib * mu_s * phase;
			 pathTracingPDF *= phase;

			 //update variables:
			 forkVertex.vertex = nextScatteringVertex;
			 forkVertex.vertexType = TYPE_MEDIUM;
			 forkVertex.geometryID = -1;
			 forkVertex.surfaceNormal = vec3(0.0f);

			 currPosition = nextScatteringVertex;
			 currDir = newDir;
		 }

		 int lastPathTracingVertexIndex = pathTracingPath->getSegmentLength();
		 if (lastPathTracingVertexIndex >= RenderingSettings::MAX_SEGMENT_COUNT) {
			 break;
		 }

		 //////////////////////////////
		 // MVNEE
		 //////////////////////////////		

		 //first sample a light vertex
		 vec3 lightVertex = scene->lightSource->sampleLightPosition(sample1D(threadID), sample1D(threadID));
		 vec3 forkToLight = lightVertex - forkVertex.vertex;
		 float distanceToLight = length(forkToLight);
		 if (distanceToLight > Constants::epsilon) {
			 //only perform MVNEE if the expected segment count is smaller than MAX_MVNEE_SEGMENTS
			 float expectedSegments = distanceToLight / MediumParameters::meanFreePathF;
			 if (isfinite(expectedSegments) && expectedSegments > 0.0f && expectedSegments <= RenderingSettings::MAX_MVNEE_SEGMENTS_F) {

				 vec3 omega = forkToLight / distanceToLight; //direction of the line to the light source

				 vec3 u, v;
				 coordinateSystem(omega, u, v); //build tangent frame: (same for every seed vertex)

				 //sample the segment lengths for the seed path:
				 int mvneeSegmentCount;
				 int maxSegmentCount = RenderingSettings::MAX_SEGMENT_COUNT - pathTracingPath->getSegmentLength();

				 bool validPath = sampleSeedSegmentLengths(distanceToLight, tl_seedSegmentLengths, &mvneeSegmentCount, threadID);

				 //special case: normal culling for only one segment:
				 if (validPath && mvneeSegmentCount == 1) {
					 if (forkVertex.vertexType == TYPE_SURFACE) {
						 float cosThetaSurface = dot(omega, forkVertex.surfaceNormal);
						 if (cosThetaSurface <= 0.0f) {
							 validPath = false;
							 occludedPathCounter[threadID]++;
						 }
					 }
					 //light normal culling!
					 if (!scene->lightSource->validHitDirection(omega)) {
						 validPath = false;
						 occludedPathCounter[threadID]++;
					 }
				 }

				 if (validPath && mvneeSegmentCount > 0) {
					 //consider max segment length:
					 if (pathTracingPath->getSegmentLength() + mvneeSegmentCount <= RenderingSettings::MAX_SEGMENT_COUNT) {

						 //determine seed vertices based on sampled segment lengths:
						 vec3 prevVertex = forkVertex.vertex;
						 for (int s = 0; s < (mvneeSegmentCount - 1); s++) {
							 vec3 newSeedVertex = prevVertex + tl_seedSegmentLengths[s] * omega;
							 tl_seedVertices[s] = newSeedVertex;
							 prevVertex = newSeedVertex;
						 }
						 //add light vertex
						 tl_seedVertices[mvneeSegmentCount - 1] = lightVertex;

						 ////////////////////////////
						 //  perturb all vertices but the lightDiskVertex:
						 ////////////////////////////
						 vec3 previousPerturbedVertex = forkVertex.vertex;
						 
						 vec3 seedVertex;
						 vec3 perturbedVertex;
						 bool validMVNEEPath = true;
						 for (int p = 0; p < (mvneeSegmentCount - 1); p++) {
							 seedVertex = tl_seedVertices[p];

							 //sanity check: distances to start and end may not be <= 0
							 float distanceToLineEnd = length(lightVertex - seedVertex);
							 float distanceToLineStart = length(seedVertex - forkVertex.vertex);
							 if (distanceToLineEnd <= 0.0f || distanceToLineStart <= 0.0f) {
								 segmentsTooShortCounter[threadID]++;
								 validMVNEEPath = false;
								 break;
							 }

							 double finalGGXAlpha = sigmaForHG * 1.637618734; //Constants value: conversion with Constants factor

							 //perform perturbation using ggx radius and uniform angle in u,v plane 
							 perturbVertexGGX2D(finalGGXAlpha, u, v, seedVertex, &perturbedVertex, threadID);


							 //for first perturbed vertex, check if perturbation violates the normal culling condition on the surface!
							 if (p == 0 && forkVertex.vertexType == TYPE_SURFACE) {
								 vec3 forkToFirstPerturbed = perturbedVertex - forkVertex.vertex;
								 float firstDistToPerturbed = length(forkToFirstPerturbed);
								 if (firstDistToPerturbed > 0.0) {
									 forkToFirstPerturbed /= firstDistToPerturbed;
									 float cosThetaSurface = dot(forkToFirstPerturbed, forkVertex.surfaceNormal);
									 if (cosThetaSurface <= 0.0f) {
										 validMVNEEPath = false;
										 occludedPathCounter[threadID]++;
										 break;
									 }
								 }
								 else {
									 validMVNEEPath = false;
									 break;
								 }
								 previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
							 }

							 //visibility check:
							 float maxT = tl_seedSegmentLengths[p];
							 vec3 visibilityDirection = normalize(perturbedVertex - previousPerturbedVertex);
							 RTCRay shadowRay;
							 if (scene->surfaceOccluded(previousPerturbedVertex, visibilityDirection, maxT, shadowRay)) {
								 validMVNEEPath = false;
								 occludedPathCounter[threadID]++;
								 break;
							 }

							 //add vertex to path
							 pathTracingPath->addMediumVertex(perturbedVertex, TYPE_MVNEE);
							 previousPerturbedVertex = perturbedVertex;
						 }

						 if (validMVNEEPath) {
							 vec3 lastSegmentDir = lightVertex - previousPerturbedVertex;
							 float lastSegmentLength = length(lastSegmentDir);
							 if (lastSegmentLength > 0.0f) {
								 assert(lastSegmentLength > 0.0f);
								 lastSegmentDir /= lastSegmentLength;

								 //visibility check: first potential normal culling:
								 if (scene->lightSource->validHitDirection(lastSegmentDir)) {

									 if (mvneeSegmentCount == 1 && forkVertex.vertexType == TYPE_SURFACE) {
										 previousPerturbedVertex = forkVertex.vertex + Constants::epsilon * forkVertex.surfaceNormal;
									 }

									 //occlusion check:
									 RTCRay lastShadowRay;
									 if (!scene->surfaceOccluded(previousPerturbedVertex, lastSegmentDir, lastSegmentLength, lastShadowRay)) {
										 PathVertex lightVertexStruct(lightVertex, TYPE_MVNEE, -1, scene->lightSource->normal);
										 pathTracingPath->addVertex(lightVertexStruct);

										 //calculate measurement constribution, pdf and mis weight:
										 //estimator Index for MVNEE is index of last path tracing vertex
										 //make sure the last path tracing vertex index is set correctly in order to use the correct PDF
										 successfullyCreatedPaths[threadID]++;
										 vec3 finalContribution = calcFinalWeightedContribution_ConstantsAlpha(pathTracingPath, lastPathTracingVertexIndex, firstPossibleMVNEEEstimatorIndex, measurementContrib, colorThroughput);
										 finalPixel += finalContribution;
									 }
									 else {
										 occludedPathCounter[threadID]++;
									 }
								 }
								 else {
									 occludedPathCounter[threadID]++;
								 }
							 }
							 else {
								 segmentsTooShortCounter[threadID]++;
								 cout << "last MVNEE segment length <= 0: " << lastSegmentLength << endl;
							 }
						 }
					 }
					 //delete all MVNEE vertices from the path construct, so path tracing stays correct
					 pathTracingPath->cutMVNEEVertices();
				 }

			 }
			 else {
				 segmentsTooShortCounter[threadID]++;
			 }
		 }
		 measurementContrib = newMeasurementContrib;
		 colorThroughput = newColorThroughput;
	 }

	 //Path sampling finished: now reset Path, so it can be filled again
	 pathTracingPath->reset();

	 return finalPixel;*/
 }

/** Perturb a vertex on the tangent plane using a ggx sampled radius, and a uniformly sampled angle
* @param input: vector that has to be perturbed
* @param alpha width of ggx bell curve
* @param u: first tangent vector as perturbation direction 1
* @param v: second tangent vector as perturbation direction 2
* @return output: perturbed input vector
*/
inline void VolumeRenderer::perturbVertexGGX2D(const double& alpha, const vec3& u, const vec3& v, const vec3& input, vec3* output, const int threadID)
{
	glm::vec2 uv = sampleGGX2D(alpha, sample1D(threadID), sample1D(threadID));
	vec3 out = input + uv.x * u + uv.y * v;

	*output = out;
}


/**
* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH).
* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
*/
bool VolumeRenderer::sampleSeedSegmentLengths(const double& curve_length, float* segments, int* segmentCount, const int threadID)
{
	assert(curve_length > 0.0);
	double lengthSum = 0.0;
	int vertexCount = 0;
	bool valid = true;
	while (lengthSum < curve_length) {
		if (vertexCount >= RenderingSettings::MAX_SEGMENT_COUNT) {
			valid = false;
			break;
		}

		double currSegmentLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);
		if (lengthSum + currSegmentLength <= curve_length)
		{
			segments[vertexCount] = (float)currSegmentLength;
		}
		else {
			segments[vertexCount] = (float)(curve_length - lengthSum);
		}
		lengthSum += currSegmentLength;
		vertexCount++;
	}

	*segmentCount = vertexCount;
	return valid;
}

/**
* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
* until the curve_length is exceeded. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
* Return value is false if MAX_SEGMENT_LENGTH is exceeded, true otherwise.
*/
inline bool VolumeRenderer::sampleSeedSegmentLengths(const double& curve_length, float* segments, double* segmentSquares, int* segmentCount, const int threadID)
{
	assert(curve_length > 0.0);
	double lengthSum = 0.0;
	int vertexCount = 0;
	bool valid = true;
	while (lengthSum < curve_length) {
		if (vertexCount >= RenderingSettings::MAX_SEGMENT_COUNT) {
			valid = false;
			break;
		}

		double currSegmentLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);
		if (lengthSum + currSegmentLength <= curve_length)
		{
			segments[vertexCount] = (float)currSegmentLength;
			segmentSquares[vertexCount] = currSegmentLength * currSegmentLength;
		}
		else {
			double lastSegmentL = curve_length - lengthSum;
			segments[vertexCount] = (float)(lastSegmentL);
			segmentSquares[vertexCount] = lastSegmentL * lastSegmentL;
		}
		lengthSum += currSegmentLength;
		vertexCount++;
	}

	*segmentCount = vertexCount;
	return valid;
}

/**
* Samples the appropriate segment count. This is done by summing up sampled free path lengths (transmittance formula as in path tracing)
* until the curve_length is exceeded or the provided maxSegments are reached. Also returns the segments in an array (maximal MAX_PATH_LENGTH) as well as the squared segment lengths.
* Once maxSegments are reached, the sampling is stopped and the remaining distance is set as the last segment. This ensures that during segment sampling, no invalid paths are created.
*/
inline void VolumeRenderer::sampleSeedSegmentLengths(const double& curve_length, float* segments, double* segmentSquares, int* segmentCount, const int threadID, const int maxSegments)
{
	assert(curve_length > 0.0);
	assert(maxSegments > 0);
	assert(maxSegments < RenderingSettings::MAX_SEGMENT_COUNT);
	double lengthSum = 0.0;
	int vertexCount = 0;
	while (lengthSum < curve_length) {

		if (vertexCount == maxSegments - 1) {
			double lastSegmentL = curve_length - lengthSum;
			segments[vertexCount] = (float)(lastSegmentL);
			segmentSquares[vertexCount] = lastSegmentL * lastSegmentL;
			vertexCount++;
			break;
		}

		double currSegmentLength = sampleFreePathLength(sample1DOpenInterval(threadID), MediumParameters::mu_t);
		if (lengthSum + currSegmentLength <= curve_length)
		{
			segments[vertexCount] = (float)currSegmentLength;
			segmentSquares[vertexCount] = currSegmentLength * currSegmentLength;
		}
		else {
			double lastSegmentL = curve_length - lengthSum;
			segments[vertexCount] = (float)(lastSegmentL);
			segmentSquares[vertexCount] = lastSegmentL * lastSegmentL;
		}
		lengthSum += currSegmentLength;
		vertexCount++;
	}

	*segmentCount = vertexCount;
}


void VolumeRenderer::saveBufferToTGA(const char* filename, vec3* imageBuffer, int imageWidth, int imageHeight)
{
	ofstream o(filename, ios::out | ios::binary);
	short width = (short)imageWidth;
	short height = (short)imageHeight;

	//Write the header
	o.put(0);
	o.put(0);
	o.put(2);                         /* uncompressed RGB */
	o.put(0); 		o.put(0);
	o.put(0); 	o.put(0);
	o.put(0);
	o.put(0); 	o.put(0);           /* X origin */
	o.put(0); 	o.put(0);           /* y origin */
	o.put((width & 0x00FF));
	o.put((width & 0xFF00) / 256);
	o.put((height & 0x00FF));
	o.put((height & 0xFF00) / 256);
	o.put(32);                        /* 24 bit bitmap */
	o.put(0);

	//Write the pixel data
	const unsigned char a = 255;
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {
			vec3 currentPixel = imageBuffer[x + imageWidth * y];
			//vec3 currentPixel = frameBuffer[x][y];

			/*float gamma_corrected_r = currentPixel.r;
			float gamma_corrected_g = currentPixel.g;
			float gamma_corrected_b = currentPixel.b;*/

			//fake tonemap
			/*float gamma_corrected_r = currentPixel.r / (currentPixel.r + 1.0f);
			float gamma_corrected_g = currentPixel.g / (currentPixel.g + 1.0f);
			float gamma_corrected_b = currentPixel.b / (currentPixel.b + 1.0f);*/

			float gamma_corrected_r = powf(currentPixel.r, 1.0f/2.2f);
			float gamma_corrected_g = powf(currentPixel.g, 1.0f/2.2f);
			float gamma_corrected_b = powf(currentPixel.b, 1.0f/2.2f);

			//cap float value so that it fits into char
			//WARNING: tone mapping?
			const unsigned char r = (unsigned char)(fminf(gamma_corrected_r * 255.0f, 255.0f));
			const unsigned char g = (unsigned char)(fminf(gamma_corrected_g * 255.0f, 255.0f));
			const unsigned char b = (unsigned char)(fminf(gamma_corrected_b * 255.0f, 255.0f));

			o.put(b);
			o.put(g);
			o.put(r);
			o.put(a);
		}
	}

	//close the file
	o.close();
}

#if defined ENABLE_OPEN_EXR
void VolumeRenderer::saveBufferToOpenEXR(const char* filename, vec3* imageBuffer, int imageWidth, int imageHeight)
{
	Array2D<Rgba> rgbaData(imageHeight, imageWidth);

	for (int y = 0; y < imageHeight; y++) {
		//for (int j = 0; j < imageWidth; j++) {
		for (int x = 0; x < imageWidth; x++) {
			vec3 currentPixel = imageBuffer[y * imageWidth + x];

			assert(isfinite(currentPixel.x));

			float tonemappedValueR = currentPixel.x;
			float tonemappedValueG = currentPixel.y;
			float tonemappedValueB = currentPixel.z;
			//float tonemappedValue = currValue / (currValue + 1.0f);

			Rgba &p = rgbaData[imageHeight - 1 - y][x];
			p.r = tonemappedValueR;
			p.g = tonemappedValueG;
			p.b = tonemappedValueB;
			p.a = 1.0f;

		}
	}

	RgbaOutputFile file(filename, imageWidth, imageHeight, WRITE_RGBA);
	file.setFrameBuffer(&rgbaData[0][0], 1, imageWidth); //memory layout
	file.writePixels(imageHeight);
}
#endif

void VolumeRenderer::printRenderingParameters(int sampleCount, double duration)
{
	//save textual output of all the parameters (volume, samples, lamp,...)
	string paramsFile = RenderingSettings::sessionName;
	paramsFile.append("_params.txt");
	ofstream o(paramsFile.c_str(), ios::out);

	glm::int64 successfulPathCount = 0;
	for (int i = 0; i < RenderingSettings::THREAD_COUNT; i++) {
		successfulPathCount += successfulPathCounter[i];
	}

	o << "Integrator: " << endl;
	switch (RenderingSettings::integrator) {
		case TEST_RENDERING: o << "Test Rendering" << endl; break;
		case PATH_TRACING_NO_SCATTERING: o << "Path Tracing, no medium interaction" << endl; break;
		case PATH_TRACING_NEE_MIS_NO_SCATTERING: o << "Path Tracing with NEE and MIS, no medium interaction" << endl; break;
		case PATH_TRACING_RANDOM_WALK: o << "Path Tracing in homogeneous medium" << endl; break;
		case PATH_TRACING_NEE_MIS: o << "Path Tracing with NEE and MIS in homogeneous medium" << endl; break;
		case PATH_TRACING_MVNEE: o << "Path Tracing with MVNEE and MIS in homogeneous medium" << endl; break;
		case PATH_TRACING_MVNEE_FINAL: o << "Path Tracing with MVNEE and MIS in homogeneous medium: final optimized version" << endl; break;
		/*case PATH_TRACING_MVNEE_OPTIMIZED: o << "Path Tracing with MVNEE and MIS in homogeneous medium - Optimized version" << endl; break;
		case PATH_TRACING_MVNEE_OPTIMIZED2: o << "Path Tracing with MVNEE and MIS in homogeneous medium - Optimized2 second version" << endl; break;
		case PATH_TRACING_MVNEE_OPTIMIZED3: o << "Path Tracing with MVNEE and MIS in homogeneous medium - Optimized3 third version" << endl; break;*/
		case PATH_TRACING_MVNEE_GAUSS_PERTURB: o << "Path Tracing with MVNEE and MIS in homogeneous medium - Gauss perturbation!" << endl; break;
		case PATH_TRACING_MVNEE_Constants_ALPHA: o << "Path Tracing with MVNEE and MIS in homogeneous medium - GGX perturbation with Constants ALPHA!" << endl; break;
		default: o << "DEFAULT: Integrator unknown!" << endl; break;
	}
	o << endl;
	o << "Image Dimension: width = " << RenderingSettings::WIDTH << ", height = " << RenderingSettings::HEIGHT << endl;
	o << "Num Samples: " << sampleCount << endl;
	o << "maximum path length: " << RenderingSettings::MAX_SEGMENT_COUNT << endl;
	o << "maximum expected MVNEE segments: " << RenderingSettings::MAX_MVNEE_SEGMENTS << endl;
	o << "count of successful paths: " << successfulPathCount << endl;
	o << endl;
	scene->lightSource->printParameters(o);	
	o << endl;
	o << "mu_s: " << MediumParameters::mu_s << endl;
	o << "mu_a: " << MediumParameters::mu_a << endl;
	o << "mu_t: " << MediumParameters::mu_t << endl;
	o << "henyey-greenstein g: " << MediumParameters::hg_g << endl;
	o << "sigma_for_hg: " << sigmaForHG << endl;
	o << endl;
	o << "Thread count for execution = ";
	if (!RenderingSettings::RENDER_PARALLEL || RenderingSettings::THREAD_COUNT == 1) {
		o << 1 << endl;
	}
	else {
		o << RenderingSettings::THREAD_COUNT << endl;
	}	
	o << "Time elapsed: " << duration << endl; 

	//close the file
	o.close();
}

void VolumeRenderer::printCounters()
{
	glm::int64 successfulPathCount = 0;
	glm::int64 pathTooShortCount = 0;
	glm::int64 successfullySampledCount = 0;
	glm::int64 rejectedSuccessfulCount = 0;
	glm::int64 occludedPathCount = 0;
	for (int i = 0; i < RenderingSettings::THREAD_COUNT; i++) {
		successfulPathCount += successfulPathCounter[i];
		pathTooShortCount += segmentsTooShortCounter[i];
		successfullySampledCount += successfullyCreatedPaths[i];
		rejectedSuccessfulCount += rejectedSuccessfulPaths[i];
		occludedPathCount += occludedPathCounter[i];
	}

	//save textual output of all the parameters (volume, samples, lamp,...)
	string paramsFile = RenderingSettings::sessionName;
	paramsFile.append("_counters.txt");
	ofstream o(paramsFile.c_str(), ios::out);
	
	
	o << "count of successfully sampled paths: " << successfullySampledCount << endl;
	o << "     count of successful paths: " << successfulPathCount << endl;
	o << "     count of rejected successfully sampled paths: " << rejectedSuccessfulCount << endl;
	o << "---------------------" << endl;
	o << "count of occluded paths: " << occludedPathCount << endl;
	o << "count of paths with segment counts too short: " << pathTooShortCount << endl;

	o.close();
}

void VolumeRenderer::writeBufferToFloatFile(const string& fileName, int width, int height, vec3* buffer) {
	string fileN = fileName;
	fileN.append(".flt");

	ofstream o;
	o.open(fileN.c_str(), ios::out | ios::binary);
	if (o.good()) {
		//first write width and height in int
		o.write((char*)&width, sizeof(int));
		o.write((char*)&height, sizeof(int));
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int index = y * width + x;
				float xValue = buffer[index].x;
				o.write((char*)&xValue, sizeof(float));

				float yValue = buffer[index].y;
				o.write((char*)&yValue, sizeof(float));

				float zValue = buffer[index].z;
				o.write((char*)&zValue, sizeof(float));
			}
		}

		o.close();
	}
	else {
		cout << "error: couldnt open file!" << endl;
	}

}


void VolumeRenderer::readFloatFileToBuffer(const string& fileName, int* width, int* height, vec3* buffer) {
	ifstream i;
	i.open(fileName.c_str(), ios::in | ios::binary);
	if (i.good()) {
		//read width and height
		int w;
		int h;
		i.read((char*)&w, sizeof(int));
		i.read((char*)&h, sizeof(int));
		*width = w;
		*height = h;

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int index = y * w + x;
				float xValue, yValue, zValue;
				i.read((char*)&xValue, sizeof(float));
				i.read((char*)&yValue, sizeof(float));
				i.read((char*)&zValue, sizeof(float));
				buffer[index] = vec3(xValue, yValue, zValue);
			}
		}

		i.close();
	}
	else {
		cout << "error: couldnt open file!" << endl;
	}
}

//sample arbitrary normal distribution N(0, std) by sampling std * N(0, 1)!!
double VolumeRenderer::sampleArbitraryGauss1D(const double& stdDeviation, const int threadID) {
	double gaussSample = gaussStandardDist[threadID](mt[threadID]);
	return stdDeviation * gaussSample;
}

/** Calculate PDF for sampling the given value with a gaussian with given standard deviation  */
double VolumeRenderer::gaussPDF(const float& x, const double& stdDeviation)
{
	double exponent = (((double)(x)*(double)(x)) / (2.0 * stdDeviation * stdDeviation));
	return (1.0 / (sqrt(2.0 * M_PI) * stdDeviation)) * exp(-exponent);
}

/** Perturb a vertex on the tangent plane using a gaussian sampled distance, gaussian is defined by N(0, sigma), where sigma is the std-deviation
* @param input: vector that has to be perturbed
* @param u: first tangent vector as perturbation direction 1
* @param v: second tangent vector as perturbation direction 2
* @return output: perturbed input vector
*/
void VolumeRenderer::perturbVertexGaussian2D(const double& sigma, const vec3& u, const vec3& v, const vec3& input, vec3* output, const int threadID)
{
	//find x and y distance
	float uDist = (float)sampleArbitraryGauss1D(sigma, threadID);
	float vDist = (float)sampleArbitraryGauss1D(sigma, threadID);

	vec3 out = input + uDist * u + vDist * v;

	*output = out;
}