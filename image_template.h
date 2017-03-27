/* This program was originally written by
Sumedh Naik (now at Intel) at Clemson University
as a part of his thesis titled, "Connecting Architectures,
fitness, Optimizations and Performance using an Anisotropic
Diffusion Filter. This header was also used
in Dr. Pallipuram's dissertation work. */

#include "stdio.h" 
#include "math.h"
#include "stdlib.h"
#include "string.h"

#define BUFFER 512

// Function Declaration


void read_image(char *name, unsigned char **image, int *im_width, int *im_height);
template <class T>
void read_image_template(char *name, T **image, int *im_width, int *im_height);
void write_image(char *name, unsigned char *image, int im_width, int im_height);
template <class T>
void write_image_template(char *name, T *image, int im_width, int im_height);

//Function Definition

/*Call this function alone to read images*/

template <class T>
void read_image_template(char *name, T **image, int *im_width, int *im_height)
{
        unsigned char *temp_img;

	int i;

        read_image(name, &temp_img, im_width, im_height);

        *image=(T *)malloc(sizeof(int)*(*im_width)*(*im_height));


        for(i=0;i<(*im_width)*(*im_height);i++)
        {
                (*image)[i]=(T)temp_img[i];
        }
        free(temp_img);
}

void read_image(char *name, unsigned char **image, int *im_width, int *im_height)
{
	FILE *fip;
	char buf[BUFFER];
	char *parse;
	int im_size;
	
	fip=fopen(name,"rb");
	if(fip==NULL)
	{
		fprintf(stderr,"ERROR:Cannot open %s\n",name);
		exit(0);
	}
	fgets(buf,BUFFER,fip);
	do
	{
		fgets(buf,BUFFER,fip);
	}
	while(buf[0]=='#');
	parse=strtok(buf," ");
	(*im_width)=atoi(parse);

	parse=strtok(NULL,"\n");
	(*im_height)=atoi(parse);

	fgets(buf,BUFFER,fip);
	parse=strtok(buf," ");
	
	im_size=(*im_width)*(*im_height);
	(*image)=(unsigned char *)malloc(sizeof(unsigned char)*im_size);
	fread(*image,1,im_size,fip);
	
	fclose(fip);
}

/* Call this function alone to write an image*/
template <class T>
void write_image_template(char *name, T *image, int im_width, int im_height)
{
	int i;
        unsigned char *temp_img=(unsigned char*)malloc(sizeof(unsigned char)*im_width*im_height);
        for(i=0;i<(im_width*im_height);i++)
        {
                temp_img[i]=(unsigned char)image[i];
        }
        write_image(name,temp_img,im_width,im_height);

        free(temp_img);
}

template <class T>
void write_img_template(char *name, T *image,int im_width,int im_height)
{

int i,j;
FILE *fop;
int im_size=im_width*im_height;

//fop=fopen(name,"w+");
//fwrite(image,sizeof(T),im_size,fop);

printf("%d\n%d\n",im_width,im_height);
for(i=0;i<im_height;i++)
{
//printf("\n");
for(j=0;j<im_width;j++)
	printf("%f\n",image[i*im_width+j]);
}

//fclose(fop);
} 

void write_image(char *name, unsigned char *image, int im_width, int im_height)
{
	FILE *fop; 
	int im_size=im_width*im_height;
	
	fop=fopen(name,"w+");
	fprintf(fop,"P5\n%d %d\n255\n",im_width,im_height);
	fwrite(image,sizeof(unsigned char),im_size,fop);
	
	fclose(fop);
}

