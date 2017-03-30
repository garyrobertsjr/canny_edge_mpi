#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "sys/time.h"
#include "mpi.h"
#include "image_template.h"

int inBounds(int x, int y, int h, int w, int tgr, int bgr){
	if(x < 0 || x > w)
		return 0;
	else if(y < 0-tgr || y > h+bgr)
		return 0;
	else
		return 1;
}

int isConnected(float *hyst, int x, int y, int height, int width, int comm_rank, int comm_size){
	int tgr, bgr;
	
	if(comm_rank==0){
		tgr=0;
		bgr=1;
	}
	else if(comm_rank==comm_size-1){
		tgr=1;
		bgr=0;
	}
	else{
		tgr=1;
		bgr=1;
	}

	for(int y_offset=-1; y_offset<=1; y_offset++){
		for(int x_offset=-1; x_offset<=1; x_offset++){
			if(inBounds(x+x_offset, y+y_offset, height, width, tgr, bgr) &&
				*(hyst+(y+y_offset)*width+x_offset+x)==255)
				return 1;
		}
	}
	return 0;
}

void print_matrix(float *matrix, int height, int width){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			printf("%.3f ", *(matrix+(i*width)+j));
		}
		printf("\n");
	}
}

// Returns neg if b > a abd pos of a > b
int floatcomp(const void *a, const void *b){
	if(*(const float*)a < *(const float*)b)
		return -1;
	return *(const float*)a > *(const float*)b;
}

// returns null if OOB, else value
float left(float *arr, int x, int y, int width){
	if(x-1>=0){
		return *(arr+y*width+(x-1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float right(float *arr, int x, int y, int width){
	if(x+1<width){
		return *(arr+y*width+(x+1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float top(float *arr, int x, int y, int height, int width){
	if(y-1>=-1){
		return *(arr+(y-1)*width+x);
	}
	return (float)NULL;
}

// returns null if OOB, else value
float bottom(float *arr, int x, int y, int height, int width){
	if(y+1<=height){
		return *(arr+(y+1)*width+x);
	}
	return (float)NULL;
}

// returns null if OOB, else value
float topLeft(float *arr, int x, int y, int height, int width){
	if(x-1>=0 && y-1>=-1){
		return *(arr+(y-1)*width+(x-1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float bottomRight(float *arr, int x, int y, int height, int width){
	if(x+1<width && y+1<=height){
		return *(arr+(y+1)*width+(x+1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float topRight(float *arr, int x, int y, int height, int width){
	if(x+1<width && y-1>=-1){
		return *(arr+(y-1)*width+(x+1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float bottomLeft(float *arr, int x, int y, int height, int width){
	if(x-1>=0 && y+1<=height){
		return *(arr+(y+1)*width+(x-1));
	}
	return (float)NULL;
}

void suppress(float *mag, float *phase, float **sup, int height, int width){
	memcpy(*sup, mag, sizeof(float)*height*width);	
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			float theta = *(phase+i*width+j);
			if(theta<0)
				theta+=M_PI;
			
			theta*=(180/M_PI);			

			if(theta<=22.5 || theta >157.5){
				if(left(mag,j,i,width) > *(mag+i*width+j)
				   	|| right(mag,j,i,width) > *(mag+i*width+j))
					*(*sup+i*width+j)=0; 
			}
			else if(theta>22.5 && theta<=67.5){
				if(topLeft(mag,j,i,width,height) > *(mag+i*width+j)
					|| bottomRight(mag,j,i,height,width) > *(mag+i*width+j))
					*(*sup+i*width+j)=0; 
			}
			else if(theta>67.5 && theta<=112.5){
				if(top(mag,j,i,height,width) > *(mag+i*width+j) 
					|| bottom(mag,j,i,height,width) > *(mag+i*width+j))
					*(*sup+i*width+j)=0; 
			}
			else if(theta>112.5 && theta<=157.5){
				if(topRight(mag,j,i,height,width) > *(mag+i*width+j) 
					|| bottomLeft(mag,j,i,height,width) > *(mag+i*width+j))
					*(*sup+i*width+j)=0; 
			}
		}
	}
}

void hysteresis(float *sup, float **hyst, int height, int width, float t_high, float t_low){
	*hyst = (float*)malloc(sizeof(float)*height*width);
	memcpy(*hyst, sup, sizeof(float)*height*width);	
	
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			if(*(sup+i*width+j)>=t_high)
				*(*hyst+i*width+j)=255;
			else if(*(sup+i*width+j)<=t_low)
				*(*hyst+i*width+j)=0;
			else if(*(sup+i*width+j)<t_high && *(sup+i*width+j)>t_low)
				*(*hyst+i*width+j)=125;
		}
	}
}

void find_edges(float *hyst, float **edges, int height, int width, int comm_rank, int comm_size){
	*edges = (float*)malloc(sizeof(float)*height*width);
	memcpy(*edges, hyst, sizeof(float)*height*width);	
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			if(isConnected(hyst,j,i,height, width, comm_rank, comm_size))
				*(*edges+i*width+j)=255;
			else
				*(*edges+i*width+j)=0;
		}
	}
}

void gradient_phase(float *v_grad, float *h_grad, float **mag, float **phase,
			int height, int width){
	*mag = (float*)malloc(sizeof(float)*height*width);
	*phase = (float*)malloc(sizeof(float)*height*width);
	
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){

			// Calucate magnitude
			*(*mag+i*width+j)=sqrt(pow(*(v_grad+i*width+j),2) 
						+ pow(*(h_grad+i*width+j),2));

			// Calculate  Phase
			*(*phase+i*width+j)=atan2(*(v_grad+i*width+j),*(h_grad+i*width+j));
		}
	}
	
}

float *sr_ghost(float *image, int height, int width, int ngr, int comm_size, int comm_rank){
	MPI_Status status;
	float *output;
	if(comm_rank==0 || comm_rank==(comm_size-1))
		output = (float*)malloc(sizeof(float)*width*(height + ngr));
	else
		output = (float*)malloc(sizeof(float)*width*(height + 2*ngr));
	if(comm_rank==0){
		// send bottom and receive top from rank+1
		MPI_Sendrecv(image+width*(height-ngr), ngr*width, MPI_FLOAT, comm_rank+1, comm_rank,
				output+width*height, ngr*width, MPI_FLOAT, comm_rank+1, comm_rank+1, 
				MPI_COMM_WORLD, &status);
	
		memcpy(output, image, sizeof(float)*width*height);

		return output;
	}
	else if(comm_rank==(comm_size-1)){
		// sent top and receive bottom from rank-1
		MPI_Sendrecv(image, ngr*width, MPI_FLOAT, comm_rank-1, comm_rank,
				output, ngr*width, MPI_FLOAT, comm_rank-1, comm_rank-1, MPI_COMM_WORLD, 
				&status);
	
		memcpy(output+ngr*width, image, sizeof(float)*width*height);
		return output+ngr*width;
	}
	else{
		//send top and receive top from rank+1	
		MPI_Sendrecv(image, ngr*width, MPI_FLOAT, comm_rank-1, comm_rank,
				output, ngr*width, MPI_FLOAT, comm_rank-1, comm_rank-1, MPI_COMM_WORLD, 
				&status);
		// send bottom and receive top from rank+1
		MPI_Sendrecv(image+width*(height-ngr), ngr*width, MPI_FLOAT, comm_rank+1, comm_rank,
				output+width*(height+ngr), ngr*width, MPI_FLOAT, comm_rank+1, comm_rank+1, 
				MPI_COMM_WORLD, &status);
		
		memcpy(output+ngr*width, image, sizeof(float)*width*height);
		return output+ngr*width;
	}
}

void convolve(float *image, float **output, float *kernel, int height, 
		int width, int k_height, int k_width, int comm_size, int comm_rank){

	*output = (float*)malloc(sizeof(float)*width*height);
	int tgr,bgr;

	if(comm_rank==0){
		tgr = 0;
		bgr = floor(k_height/2);
	}
	else if(comm_rank==comm_size-1){
		tgr=floor(k_height/2);
		bgr = 0;
	}
	else{
		tgr = floor(k_height/2);
		bgr = tgr;
	}
	// Iter pixels and convolve
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			float sum = 0;
			// Iter kernel
			for(int k=0; k<k_height;k++){
				for(int m=0; m<k_width; m++){
					int offseti = -1*floor(k_height/2)+k;
					int offsetj = -1*floor(k_width/2)+m;
					
					if(inBounds(j+offsetj, i+offseti, height, width, tgr, bgr)){
						sum+= *(image+(i+offseti)*width+j+offsetj)*(*(kernel+(k*k_width)+m));
					}
				}
			}
			*(*output+(i*width)+j)=sum;
		}
	}
}

void create_gaussians(float **g_kernel, float **dg_kernel, float sigma, int *w){
	float a = ceil(2.5*sigma-0.5);
	int sum = 0;
	
	*w=2*a+1;
	*g_kernel=(float*)malloc(sizeof(float)*(*w));

	// Calculate gaussian	
	for(int i=0; i<(*w); i++){
		(*g_kernel)[i] = exp((-1*(i-a)*(i-a))/
			  (2*sigma*sigma));
		sum+=(*g_kernel)[i];			   
	}
	
	// Normalize
	for(int i=0; i<(*w); i++){
		(*g_kernel)[i]/=sum;	
	}

	// Calculate Derivative
	sum = 0;
	
	*dg_kernel=(float*)malloc(sizeof(float)*(*w));
	
	for(int i=0; i<(*w); i++){
		(*dg_kernel)[i] = (-1*(i-a))*exp((-1*(i-a)*(i-a))/
			  (2*sigma*sigma));
		sum-=i*(*dg_kernel)[i];			   
	}
	
	// Normalize
	for(int i=0; i<(*w); i++){
		(*dg_kernel)[i]/=sum;	
	}
	
}

int main(int argc, char **argv){
	if(argc != 3)
		printf("cannyedge_MPI <file> <sigma>\n");
	else{
		int height, width, k_width, k_height=1, comm_size, comm_rank;
		struct timeval start, end;
		float *g_kernel, *dg_kernel, *t_hor, *t_ver, *image, 
		      *h_grad, *v_grad, *mag, *phase, *sup, *hyst, 
		      *subimage, *edges, *th_grad, *fh_grad, *tv_grad,
		      *fv_grad, *tmag, *tphase, *ft_sup, *t_sup, t_high,
		       t_low, *sorted, *ft_hyst, *t_edges, *ft_edges, *test;	

		MPI_Init(&argc,&argv);
		MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
		
		create_gaussians(&g_kernel, &dg_kernel, atof(argv[2]), &k_width);
		
		if(comm_rank==0){
			read_image_template(argv[1],
				    	&image,
				    	&width,
				    	&height);

			h_grad = (float*)malloc(sizeof(float)*width*height);
			v_grad = (float*)malloc(sizeof(float)*width*height);
			mag = (float*)malloc(sizeof(float)*width*height);
			phase = (float*)malloc(sizeof(float)*width*height);
			sup = (float*)malloc(sizeof(float)*width*height);
			hyst = (float*)malloc(sizeof(float)*width*height);
			edges = (float*)malloc(sizeof(float)*width*height);
			
			printf("Gaussian Kernel:\n");
			print_matrix(g_kernel, 1, k_width);
			printf("Derivative Kernel:\n");
			print_matrix(dg_kernel,1,k_width);
			printf("Kernel: Width %d\n", k_width);
		
			printf("Processing %s using %d nodes...\n", argv[1], comm_size);
			gettimeofday(&start, NULL);
		}
	
		// Broadcast height, width, k_width, kernel, dkernel
		MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
		// Dispatch chunks or original to nodes
		// NOTE: Need to add additional room for ghost rows
		//       will handle S/R in head of each func.
		subimage = (float*)malloc(sizeof(float)*width*(height/comm_size));
		
		MPI_Scatter(image, width*(height/comm_size), MPI_FLOAT, 
				subimage, width*(height/comm_size), 
				MPI_FLOAT, 0, MPI_COMM_WORLD);

		//TODO: Refactor naming conventions for convolution
		// HORIZONTAL GRADIENT //
		th_grad = sr_ghost(subimage, height/comm_size, width, floor(k_width/2), comm_size, comm_rank);
		convolve(th_grad, &t_hor, g_kernel, height/comm_size, width, k_width, 1,
				comm_size, comm_rank);
		

		fh_grad = sr_ghost(th_grad, height/comm_size, width, 0, comm_size, comm_rank);
		convolve(t_hor, &fh_grad, dg_kernel, height, width, 1, k_width,
				comm_size, comm_rank);
		
		MPI_Gather(fh_grad, width*(height/comm_size), MPI_FLOAT, h_grad,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		// VERTICAL GRADIENT //
		tv_grad = sr_ghost(subimage, height/comm_size, width, 0, comm_size, comm_rank);
		convolve(tv_grad, &t_ver, g_kernel, height/comm_size, width, 1, k_width,
				comm_size, comm_rank);
		

		t_ver = sr_ghost(t_ver, height/comm_size, width, floor(k_width/2), comm_size, comm_rank);
		convolve(t_ver, &fv_grad, dg_kernel, height, width, k_width, 1,
				comm_size, comm_rank);

		MPI_Gather(fv_grad, width*(height/comm_size), MPI_FLOAT, v_grad,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		// MAGNITUDE AND PHASE //
		tmag = (float*)malloc(sizeof(float)*(height/comm_size)*width);
		tphase = (float*)malloc(sizeof(float)*(height/comm_size)*width);
		gradient_phase(fv_grad, fh_grad, &tmag, &tphase, height/comm_size, width);
		
		MPI_Gather(tmag, width*(height/comm_size), MPI_FLOAT, mag,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);

		MPI_Gather(tphase, width*(height/comm_size), MPI_FLOAT, phase,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		// SUPPRESS //
		ft_sup = (float*)malloc(sizeof(float)*width*(height/comm_size));
		t_sup = sr_ghost(tmag, height/comm_size, width, 1, comm_size, comm_rank);
		suppress(t_sup, tphase, &ft_sup, height/comm_size, width);	
	
		MPI_Gather(ft_sup, width*(height/comm_size), MPI_FLOAT, sup,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
	
		// HYST //
		
		// Calc thresholds on node 0
		if(comm_rank==0){
			sorted = (float*)malloc(sizeof(float)*height*width);
			memcpy(sorted, sup, sizeof(float)*height*width);
			qsort(sorted, height*width, sizeof(float), floatcomp);

			t_high = *(sorted+(int)(.9*height*width));
			t_low = t_high/5;
			
			free(sorted);
		}
		
		// Updates thresholds for all nodes
		MPI_Bcast(&t_high, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&t_low, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		hysteresis(ft_sup, &ft_hyst, height/comm_size, width, t_high, t_low);

		MPI_Gather(ft_hyst, width*(height/comm_size), MPI_FLOAT, hyst,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
		
	
		// EDGES //

		t_edges = sr_ghost(ft_hyst, height/comm_size, width, 1, comm_size, comm_rank);
		find_edges(t_edges, &ft_edges, height/comm_size, width, comm_size, comm_rank);	
	
		MPI_Gather(ft_edges, width*(height/comm_size), MPI_FLOAT, edges,
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		MPI_Finalize();
			
		if(comm_rank==0){
			gettimeofday(&end, NULL);
			write_image_template("h_grad_MPI.pgm", h_grad, width, height);
			write_image_template("v_grad_MPI.pgm", v_grad, width, height);
			write_image_template("mag_MPI.pgm", mag, width, height);
			write_image_template("phase_MPI.pgm", phase, width, height);
			write_image_template("suppress_MPI.pgm", sup, width, height);
			write_image_template("hysteresis_MPI.pgm", hyst, width, height);
			write_image_template("edges_MPI.pgm",edges, width, height);
			printf("%ld\n", (end.tv_sec *1000000 + end.tv_usec)
				-(start.tv_sec * 1000000 + start.tv_usec));
		}
	}
}
