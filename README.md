# Tunisian-License-Plate-Recognition
This project aims to detect and recognize the regular Tunisian license plates with high accuracy using Mask-RCNN and Image processing techniques. It is part of the Computer Vision for License Plate Recognition Challenge from Zindi (https://zindi.africa/competitions/ai-hack-tunisia-2-computer-vision-challenge-2) designed specifically for the AI Tunisia Hack 2019. 
The data is composed of two datasets (data folder):

  - A set of vehicle images (900 images) taken from the internet and annotated manually. The annotations are the coordinates of the bounding box containing the license plate.
  - A set of license plate images (900 images) where the annotations are the text written in the license plate.
  

## Steps for installing dependencies:

	1- Create conda environment for project with python version 3
	
		conda create -n "name" python=3
		
	2- Install openCV, pandas, numpy, matplotlib and jupyter
	
		command for jupyter:

			conda install -c conda-forge jupyterlab

		command for openCV:

			conda install -c conda-forge opencv

		commands for the rest:

			conda install pandas, numpy

		command for matplotlib:

			conda install -c conda-forge matplotlib
			
		command for pillow:
		
			conda install -c anaconda pillow
			
		command for keras:
		
			conda install -c conda-forge keras
			
		conda install -c anaconda tensorflow-gpu
			
		conda install -c anaconda cudatoolkit


	3- Launch jupyter notebook and start exploring the data
	
## Supported tensorflow and keras versions
This model works fine with tensorflow=1.14 and keras=2.2.5

## Note:
This project still under development

## license
This project is free to explore, contribute and may be redistributed under the terms specified in the [LICENSE](LICENSE.txt) file.
