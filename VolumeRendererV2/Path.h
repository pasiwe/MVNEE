#pragma once

#include "Settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using glm::vec3;
using namespace std;

enum VertexType {
	TYPE_ORIGIN,
	TYPE_MEDIUM,
	TYPE_SURFACE,
	TYPE_MVNEE
};

struct PathVertex {
	vec3 vertex;	
	vec3 surfaceNormal;
	VertexType vertexType;
	int geometryID;

	PathVertex() : vertex(0.0f), vertexType(TYPE_MEDIUM), geometryID(-1), surfaceNormal(vec3(0.0f)) {

	}

	PathVertex(const vec3& vertex, const VertexType& vertexType, int geometryID, vec3 surfaceNormal) : vertex(vertex), 
		vertexType(vertexType),
		geometryID(geometryID),
		surfaceNormal(surfaceNormal) 
	{

	}
};

/**
*	Path that stores the vertices of the path.
*/
class Path
{

private:

	/** Number of segments (vertex-to-vertex-conenction) of the path */
	int segmentLength;

	/** Number of actual vertices in the path*/
	int vertexCount;

	int pathTracingVertexCount;

	/** Maximum number of vertices for this path */
	int maxPathVertices;

	/** Vertices of the path. Starting at the image plane vertex, ending on a light vertex. */
	PathVertex pathVertices[RenderingSettings::MAX_SEGMENT_COUNT + 1];

public:
	Path()
	{
		segmentLength = -1;
		maxPathVertices = RenderingSettings::MAX_SEGMENT_COUNT + 1;
		vertexCount = 0;
		pathTracingVertexCount = 0;
	}

	Path(const PathVertex& startVertex)
	{
		segmentLength = 0;
		maxPathVertices = RenderingSettings::MAX_SEGMENT_COUNT + 1;

		vertexCount = 0;
		pathVertices[vertexCount] = startVertex;
		vertexCount++;
		pathTracingVertexCount = 1;
	}

	~Path()
	{

	}

	/* Resets all attributes, so the Path can be reused without having to create a new instance */
	void reset() {
		segmentLength = -1;
		vertexCount = 0;
		pathTracingVertexCount = 0;
	}

	inline void addVertex(const PathVertex& nextPathVertex) {
		if (vertexCount < maxPathVertices) {
			pathVertices[vertexCount] = nextPathVertex;
			vertexCount++;
			segmentLength++;
			if (nextPathVertex.vertexType != TYPE_MVNEE) {
				pathTracingVertexCount++;
			}
		}
		else {
			cout << "Path Array is full!" << endl;
		}
	}

	inline void addMediumVertex(const vec3& vertex, const VertexType& type) {
		assert(type != TYPE_SURFACE);
		if (vertexCount < maxPathVertices) {
			pathVertices[vertexCount] = PathVertex(vertex, type, -1, vec3(0.0f));
			vertexCount++;
			segmentLength++;
			if (type != TYPE_MVNEE) {
				pathTracingVertexCount++;
			}
		}
		else {
			cout << "Path Array is full!" << endl;
		}
	}

	inline void addSurfaceVertex(const vec3& vertex, const int geometryID, const vec3& surfaceNormal) {
		if (vertexCount < maxPathVertices) {
			pathVertices[vertexCount] = PathVertex(vertex, TYPE_SURFACE, geometryID, surfaceNormal);
			vertexCount++;
			segmentLength++;
			pathTracingVertexCount++;
		}
		else {
			cout << "Path Array is full!" << endl;
		}
	}

	inline PathVertex* getPathVertices() {
		return pathVertices;
	}

	inline int getSegmentLength()
	{
		return segmentLength;
	}

	inline int getVertexCount()
	{
		return vertexCount;
	}

	inline VertexType getTypeAt(int index) {
		assert(index < vertexCount && index >= 0);
		return pathVertices[index].vertexType;
	}

	inline void getVertex(int index, PathVertex& output) {
		assert(index < vertexCount);
		if (index < vertexCount) {
			output = pathVertices[index];
		}
		else {
			cout << "Path: index out of bounds!" << endl;
		}
	}

	inline vec3 getVertexPosition(int index) {
		assert(index < vertexCount);
		if (index < vertexCount) {
			return pathVertices[index].vertex;
		}
		else {
			cout << "Path: index out of bounds!" << endl;
			return vec3(0.0f);
		}
	}

	/**
	* Cut the end MVNEE vertices by reducing the segment length and vertex count to the path tracing vertex count.
	*/
	inline void cutMVNEEVertices() {
		vertexCount = pathTracingVertexCount;
		segmentLength = pathTracingVertexCount - 1;
		
	}

	/**
	* Cut the end vertices by reducing the segment length and vertex count.
	*/
	void reduceSegmentLengthTo(int newSegmentLength) {
		segmentLength = newSegmentLength;
		vertexCount = newSegmentLength + 1;
	}
};