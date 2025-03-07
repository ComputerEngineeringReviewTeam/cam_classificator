```
/ai                             # ! Main module for the AI model development
  ├── /dataset
  │   ├── cam_label.py          # Classes for loading labels from JSON or CSV
  │   ├── cam_dataset.py	    # CamDataset class implementing dataset for loading data
  │   └── dataset_helpers.py    # Helper functions for creating datasets, dataloaders and splitting data into train/test subsets
  ├── /nn
  │   ├── cam_nn.py             # Main script for training and testing the neural network
  │   ├── camnet.py             # CamNet class defining the neural network model
  │   └── custom_loss.py        # Custom loss function for the neural network
  ├── /tools
  │   └── cam_snapshot.py       # CamSnapshot class for saving snapshots of model and config parameters
  ├── config.py                 # Configuration file for various settings like device, model, training, and testing #TODO: split logically
  └── paths.py                  # Defines paths for labels, images, and model

/data                           # ! Data used for training and testing the model
  ├── /photos			        # All the CAM tissue photos, filenames are image IDs
  │   ├── <image_id>1.jpg
  │   ├── <image_id2>.jpg
  │   └── ...
  └── data.json                 # JSON file with labels for each image

/data-scrapper                  
  └── scrape.py                 # Script for scraping data from web app

/data_acquisition               # ! Flask app for acquiring, labeling and storing data
  ├── app/
  │   ├── config/
  │   │   └── config.py
  │   ├── domain/
  │   │   ├── common/
  │   │   │   ├── authentication/
  │   │   │   │   ├── blueprints/
  │   │   │   │   ├── forms/
  │   │   │   │   ├── decorators/
  │   │   │   │   ├── routes/
  │   │   │   │   ├── static/
  │   │   │   │   │   ├── css/
  │   │   │   │   │   └── js/
  │   │   │   │   └── templates/
  │   │   │   └── mongodb/
  │   │   └── training_data/   
  │   │       ├── blueprints/      
  │   │       ├── dto/
  │   │       ├── forms/     
  │   │       ├── queries/
  │   │       ├── repositories/
  │   │       ├── routes/  
  │   │       ├── services/
  │   │       ├── static/    
  │   │       │   ├── css/   
  │   │       │   └── js/    
  │   │       └── templates/ 
  │   ├── static/
  │   │   └── css/
  │   ├── templates/
  │   └── __init__.py
  ├── .env
  ├── .env-local-base
  └── run.py

/logs                           # ! Directory for storing logs generated during training and testing
  └── ...

/docs
  └── project_map.md            # This file

/models                         # ! Directory for storing trained models
  └── ...
```