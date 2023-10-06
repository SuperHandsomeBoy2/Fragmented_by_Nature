# Fragmented_by_Nature
Data sources for three novel worldwide urban indexes--the share of natural barriers, average dyadic nonconvexity, and the average road detour.

Research Paper: Fragmented by Nature: Metropolitan Geography,Inner Urban Connectivity, and Environmental Outcomes

Author: Luyao Wang, Albert Saiz, Weipeng Li

The code in this folder is used to calculate the detour index for 13000+ urban areas all over the world.

* Please run "pip install -r requirements.txt" to install all the dependencies.

## [demo.py](demo.py)
* The file "[demo.py](demo.py)" can be run directly. 

## [global_graphical_index_with_road_map.py](global_graphical_index_with_road_map.py)
* Please unzip "[Data/global_data/global_data.rar](Data/global_data/global_data.rar)" first.

* The road net data used in this paper comes from The Global Roads Inventory Project (GRIP) dataset, 
it should be downloaded from: https://www.globio.info/ before running this demo.

* Introduction of the data is accessible: https://www.globio.info/download-grip-dataset .

* If the road net data is not downloaded, please download it and extract to the folder: [Data/global_data/](Data/global_data/)
    
* Part of the city road net is automatically downloaded from https://www.openstreetmap.org/ 
when the data of https://www.globio.info/ is insufficient.

