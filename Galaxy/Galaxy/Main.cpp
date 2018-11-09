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

using namespace std;
using namespace tbb;

#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 1000

// Set number of particles
const int numberOfParticles = 150;

void calculation(Quad *quadtree, Particle *galaxy) {
	quadtree->calculateMass();
	//Parallel_for data parallism
	parallel_for(0, numberOfParticles, [&](size_t i) {
		galaxy[i].force = quadtree->CalculateForce(galaxy[i]);
		//galaxy[i].x = quadtree->newPositionInX(galaxy[i], galaxy[i].force);
		//galaxy[i].y = quadtree->newPositionInX(galaxy[i], galaxy[i].force);
	});
}

void insert(Quad *quadtree, Particle *galaxy) {
	//Parallel_for data parallism
	parallel_for(0, numberOfParticles, [&](size_t i) {
		Node *node = new Node(galaxy[i]);
		quadtree->insert(node);
	});
}

int main(void)
{

	// Create the quadtree and randomly distribute the particles in 2d
	Particle *galaxy1 = new Particle[numberOfParticles];
	Particle *galaxy2 = new Particle[numberOfParticles];

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

		Quad *quadtree1 = new Quad(Particle(0, 0, 0), Particle(2000, 2000, 0));
		Quad *quadtree2 = new Quad(Particle(0, 0, 0), Particle(2000, 2000, 0));

		//Calculations threads (task parallism)
		task_group insertGroup;
		insertGroup.run([&] { insert(quadtree1, galaxy1); }); // spawn a task
		insertGroup.run([&] { insert(quadtree2, galaxy2); }); // spawn another task
		insertGroup.wait(); // wait for both tasks to complete

							//Changing possitions, display threads ( task parallism)
		task_group calculatioGroup;
		calculatioGroup.run([&] { calculation(quadtree1, galaxy1); }); // spawn a task
		calculatioGroup.run([&] { calculation(quadtree2, galaxy2); }); // spawn another task
		calculatioGroup.wait(); // wait for both tasks to complete			

								//put the particle in the vertix array
		int j = 0;
		for (int i = 0; i < numberOfParticles; i++) {
			pointVertex[j] = galaxy1[i].x;
			pointVertex[j + 1] = galaxy1[i].y;
			j += 2;
		}
		for (int i = 0; i < numberOfParticles; i++) {
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
		delete quadtree1;
		delete quadtree2;
	}

	glfwTerminate();

	return 0;
}