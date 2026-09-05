Expected output- 

1)non_traversable.py - gives the nontraversable mask  

2)casuality_info.py - gives info about all the casualities in the image (age,severity,location,priority_score) and total no. of casualities  

3)elevation_mask.py - gives seperate masks of 4 different elevation levels (not that useful , made only for learning purpose)  
4)get_elevation.py - first it detects all the elevation contours and puts them in a list then it detects the shapes and using a loop it tries to see if the shape center lies in any of the elevation contours (according to me this should work but it fails some of the times).  
5)priority_travel.py - it sorts all the casualites on basis of priority score and then connects them with straight line
