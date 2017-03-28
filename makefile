all:
	g++ -o cannyedge cannyedge.c image_template.h -lm
	mpic++ -o cannyedge_MPI cannyedge_MPI.c
