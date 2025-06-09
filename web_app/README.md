# CAM Classificator project

Chorioallantoic membrane is a highly vascularized tissue responsible for gas exchange. 
Monitoring the growth of blood vessels and their branching levels can serve as a basis for
research in the fields of cosmetology, oncology, and allergology. Properly prepared embryos, 
after undergoing examinations, are photographed and described by qualified specialists, and
the analysis of these images is tedious and time-consuming.

Noticing this problem, we propose a solution based on software that utilizes artificial neural networks,
aimed at analyzing images of the chorionic and amniotic membrane and assessing the suitability of the
material for further research, as well as recognizing changes occurring in the studied tissues as a result
of the examination process. 

This is a web application using neural networks to analyze the provided images. This will enable researchers from the Faculty of Chemistry, and potentially other entities in the future, to utilize the CAM model more efficiently.

## Running the application

### Running the containerized application
- Make sure that the environment variables inside docker-compose are correct
- Run `docker compose up -d`


### Running locally
There are two ways, depending whether we want our browser app to be a standalone 
application with live updated, or we want one application handling everything.

#### With two applications
- First start the server -> ./backend/README.md
- Then start the web application -> ./frontend/README.md **Standalone Start** section

#### As one application
- Build the web application -> ./frontend/README.md **Build** section
- Then start the server -> ./backend/README.md

