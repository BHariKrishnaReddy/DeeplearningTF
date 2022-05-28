# script that provides jovyan, the default Docker container username, permission to write to the directory
sudo chgrp -R 100 DeeplearningTF/
sudo chmod -R g+w DeeplearningTF/
