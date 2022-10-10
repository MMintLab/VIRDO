download_dataset(){
  (
    cd 'data'
<<<<<<< HEAD
    wget https://www.dropbox.com/s/xtyotb99gqb72xp/virdo_simul_dataset.pickle
=======
    curl https://www.dropbox.com/s/xtyotb99gqb72xp/virdo_simul_dataset.pickle?dl=0 -L -O -J
    unzip -o data.zip virdo_simul_dataset.pickle
>>>>>>> no lfs
  )
}

download_pretrained(){
  (
    cd 'pretrained_model'
<<<<<<< HEAD
    wget https://www.dropbox.com/s/7h2sqc6ouzlk94y/pretrained_model.zip
=======
    curl https://www.dropbox.com/sh/s1r9gxd9dz4wdkk/AADK6mHvrPtFYo_xD4EmZZdda?dl=0 -L -O -J
>>>>>>> no lfs
    unzip -o pretrained_model.zip
  )
}
