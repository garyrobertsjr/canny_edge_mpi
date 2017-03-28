#include "math.h"
#include "stdlib.h"
#include "stdio.h"
#include "time.h"
#include "sys/time.h"
#include "mpi.h"
#include "image_template.h"

int inBounds(int x, int y, int h, int w){
	if(x < 0 || x >= w)
		return 0;
	else if(y < 0 || y >= h)
		return 0;
	else
		return 1;
}

int isConnected(float *hyst, int x, int y, int height, int width){
	for(int y_offset=-3; y_offset<=3; y_offset++){
		for(int x_offset=-3; x_offset<=3; x_offset++){
			if(inBounds(x+x_offset, y+y_offset, height, width) &&
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
	if(y-1>=0){
		return *(arr+(y-1)*width+x);
	}
	return (float)NULL;
}

// returns null if OOB, else value
float bottom(float *arr, int x, int y, int height, int width){
	if(y+1<height){
		return *(arr+(y+1)*width+x);
	}
	return (float)NULL;
}

// returns null if OOB, else value
float topLeft(float *arr, int x, int y, int height, int width){
	if(x-1>=0 && y-1>=0){
		return *(arr+(y-1)*width+(x-1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float bottomRight(float *arr, int x, int y, int height, int width){
	if(x+1<width && y+1<height){
		return *(arr+(y+1)*width+(x+1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float topRight(float *arr, int x, int y, int height, int width){
	if(x+1<width && y-1>=0){
		return *(arr+(y-1)*width+(x+1));
	}
	return (float)NULL;
}

// returns null if OOB, else value
float bottomLeft(float *arr, int x, int y, int height, int width){
	if(x-1<=0 && y+1<height){
		return *(arr+(y+1)*width+(x-1));
	}
	return (float)NULL;
}

void suppress(float *mag, float *phase, float **sup, int height, int width){
	*sup = (float*)malloc(sizeof(float)*height*width);
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
					|| bottomRight(mag,j,i,width,height) > *(mag+i*width+j))
					*(*sup+i*width+j)=0; 
			}
			else if(theta>67.5 && theta<=112.5){
				if(top(mag,j,i,height,width) > *(mag+i*width+j) 
					|| bottom(mag,j,i,height,width) > *(mag+i*width+j))
					*(*sup+i*width+j)=0; 
			}
			else if(theta>112.5 && theta<=157.5){
				if(topRight(mag,j,i,height,width) > *(mag+i*width+j) 
					|| bottomLeft(mag,i,j,height,width) > *(mag+i*width+j))
					*(*sup+i*width+j)=0; 
			}
		}
	}
}

void hysteresis(float *sup, float **hyst, int height, int width){
	float *sorted = (float*)malloc(sizeof(float)*height*width);
	int t_high, t_low;
	
	*hyst = (float*)malloc(sizeof(float)*height*width);
	memcpy(*hyst, sup, sizeof(float)*height*width);	
	memcpy(sorted, sup, sizeof(float)*height*width);	
	
	qsort(sorted, height*width, sizeof(float), floatcomp);
	t_high = *(sorted+(int)(.965*height*width));
	t_low = t_high/5;
	

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
	free(sorted);
}

void find_edges(float *hyst, float **edges, int height, int width){
	*edges = (float*)malloc(sizeof(float)*height*width);
	memcpy(*edges, hyst, sizeof(float)*height*width);	
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			if(isConnected(hyst,j,i,height, width))
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

void convolve(float *image, float **output, float *kernel, int height, 
		int width, int k_height, int k_width){
	
	*output = (float*)malloc(sizeof(float)*height*width);
	
	// Iter pixels
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			float sum = 0;
			// Iter kernel
			for(int k=0; k<k_height;k++){
				for(int m=0; m<k_width; m++){
					int offseti = -1*floor(k_height/2)+k;
					int offsetj = -1*floor(k_width/2)+m;
					
					if(inBounds(j+offsetj, i+offseti, height, width))
						sum+= *(image+(i+offseti)*width+j+offsetj)*(*(kernel+(k*k_width)+m));
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
		printf("cannyedge <file> <sigma>\n");
	else{
		int height, width, k_width, comm_size, comm_rank;
		struct timeval start, end;
		float *g_kernel, *dg_kernel, *t_hor, *t_ver, *image, 
		      *h_grad, *v_grad, *mag, *phase, *sup, *hyst, 
		      *subimage, *edges;	

		read_image_template(argv[1],
				    &image,
				    &width,
				    &height);
		
		create_gaussians(&g_kernel, &dg_kernel, atof(argv[2]), &k_width);
	
		printf("Gaussian Kernel:\n");
		print_matrix(g_kernel, 1, k_width);
		printf("Derivative Kernel:\n");
		print_matrix(dg_kernel,1,k_width);

		gettimeofday(&start, NULL);
		
		MPI_Init(&argc,&argv);
		MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

		printf("Processing %s using %d nodes...\n", argv[1], comm_size);


		// Dispatch chunks or original to nodes
		MPI_Scatter(image, width*(height/comm_size), MPI_FLOAT, subimage, 
				width*(height/comm_size), MPI_FLOAT, 0, MPI_COMM_WORLD);


		printf("TEST\n");
		// Convolve Each Subimage
		//convolve(subimage, &t_hor, g_kernel, height/comm_size, width, k_width, 1);
		
		// Pass ghost rows

		// Convolve Each subimage
		//convolve(t_hor, &h_grad, dg_kernel, height, width, 1, k_width);
		
		// Gather subimages and write
		//write_image_template("h_grad.pgm", h_grad, width, height);
		
		MPI_Finalize();

		printf("%ld\n", (end.tv_sec *1000000 + end.tv_usec)
			-(start.tv_sec * 1000000 + start.tv_usec));
	}
}
