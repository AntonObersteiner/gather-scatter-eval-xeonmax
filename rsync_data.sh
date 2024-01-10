server_name=crobat
base_directory=/home/aobersteiner/gather

#rsync the data folder into the main directory
#(it’s in the same place, despite not being tracked by git)

rsync -avuP $server_name:$base_directory/data ./
