/*
For Reference and understanding
http://beltoforion.de/article.php?a=barnes-hut-galaxy-simulator
*/

#include <GLFW/glfw3.h>
#include <iostream>
#include "Quad.cpp"
#include <time.h>
#include "tbb/tbb.h"
#include "tbb/task_group.h"
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace tbb;

#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 1000


// Set number of particles
const int numberOfParticles = 150;

//Kernel function to add calculate the mass and force from all the particle in the array to the quad
__global__
void calculation(Quad *quadtree, Particle *galaxy) {

	quadtree->calculateMass();

	//index of the current thread within its block
	int index = threadIdx.x;
	//number of threads in the block
	int stride = blockDim.x;	
	for (int i = index; i < numberOfParticles; i+= stride) {
		galaxy[i].force = quadtree->CalculateForce(galaxy[i]);
		galaxy[i].x = quadtree->newPositionInX(galaxy[i], galaxy[i].force);
		galaxy[i].y = quadtree->newPositionInX(galaxy[i], galaxy[i].force);
	}	
}

//Kernel function to insert particles in quad
__global__
void insert(Quad *quadtree, Particle *galaxy) {
	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < numberOfParticles; i += stride) {
		Node *node = new Node(galaxy[i]);
		quadtree->insert(node);
	}
}

int main(void)
{

	// Create the quadtree and randomly distribute the particles in 2d
	Particle *galaxy1, *galaxy2;

	//Allocate Unified Memory, acccessible from CPU or GPU
	cudaMallocManaged(&galaxy1, numberOfParticles*sizeof(Particle));
	cudaMallocManaged(&galaxy2, numberOfParticles * sizeof(Particle));

	//Mark the extremity of the the board to divide
	for (int i = 0; i < numberOfParticles; i++) {
		Particle *p = new Particle(rand() % 100 + 200, rand() % 100 + 200, rand() % 1000);
		galaxy1[i] = *p;
	}

	for (int i = 0; i < numberOfParticles; i++) {
		Particle *p = new Particle(rand() % 200 + 700, rand() % 200 + 700, rand() % 1000);
		galaxy2[i] = *p;
	}

	// use Barnes-hut to find the forces and etc.

	GLFWwindow* window;

	if (!glfwInit()) {
		return -1;
	}

	//Create a windowed mode window and its OpenGL context
	window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Hello World", NULL, NULL);

	if (!window) {
		glfwTerminate();
		return -1;
	}

	//Make the window's context current
	glfwMakeContextCurrent(window);

	glViewport(0.0f, 0.0f, SCREEN_WIDTH, SCREEN_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, 0, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	//Loop until the user closes the window
	while (!glfwWindowShouldClose(window)) {

		// Creet a pointVertex to show the particle in the openGL
		GLfloat *pointVertex = new GLfloat[numberOfParticles * 4];

		Quad *quadtree1, *quadtree2;

		cudaMallocManaged(&quadtree1, sizeof(Quad));
		cudaMallocManaged(&quadtree2, sizeof(Quad));

		quadtree1 = new Quad(Particle(0, 0, 0), Particle(2000, 2000, 0));
		quadtree2 = new Quad(Particle(0, 0, 0), Particle(2000, 2000, 0));

		//Run kernel on the GPU, 256 threads should be reasonable size
		int count = 200;
	
		insert<<<(count/256)+1, 256>>>(quadtree1, galaxy1);
		insert<<<(count / 256) + 1, 256>>>(quadtree2, galaxy2);

		cudaDeviceSynchronize();

		//Run kernel on the GPU, 256 threads should be reasonable size
		calculation<<<(count / 256) + 1, 256>>>(quadtree1,galaxy1);
		calculation<<<(count / 256) + 1, 256>> >(quadtre2, galaxy2);

		cudaDeviceSynchronize();
		
		//put the particle in the vertix array
		int j = 0;
		for (int i = 0; i < numberOfParticles; i++) {
			if (rand() % 2 == 1) {
				galaxy1[i].x += 4;
			}
			else {
				galaxy1[i].x -= 4;
			}
			if (rand() % 2 == 1) {
				galaxy1[i].y += 4;
			}
			else {
				galaxy1[i].x -= 4;
			}
			pointVertex[j] = galaxy1[i].x;
			pointVertex[j + 1] = galaxy1[i].y;
			j += 2;
		}
		for (int i = 0; i < numberOfParticles; i++) {
			if (rand() % 2 == 1) {
				galaxy2[i].x += 4;
			}
			else {
				galaxy2[i].x -= 4;
			}
			if (rand() % 2 == 1) {
				galaxy2[i].y += 4;
			}
			else {
				galaxy2[i].x -= 4;
			}
			pointVertex[j] = galaxy2[i].x;
			pointVertex[j + 1] = galaxy2[i].y;
			j += 2;
		}

		glClear(GL_COLOR_BUFFER_BIT);

		glEnable(GL_POINT_SMOOTH);

		glEnableClientState(GL_VERTEX_ARRAY);
		glPointSize(5.0f);
		glVertexPointer(2, GL_FLOAT, 0, pointVertex);
		glDrawArrays(GL_POINTS, 0, numberOfParticles * 2);
		glDisableClientState(GL_VERTEX_ARRAY);

		glDisable(GL_POINT_SMOOTH);

		glfwSwapBuffers(window);
		glfwPollEvents();

		delete[] pointVertex;
		//free Memory
		cudaFree(quadtree1);
		cudaFree(quadtree2);
	}

	glfwTerminate();

	//free Memory
	cudaFree(galaxy1);
	cudaFree(galaxy2);

	return 0;
}