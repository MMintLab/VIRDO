download_dataset(){
  (
    cd 'data'
    curl https://www.dropbox.com/s/xtyotb99gqb72xp/virdo_simul_dataset.pickle?dl=0 -L -O -J
    unzip -o data.zip virdo_simul_dataset.pickle
  )
}

download_pretrained(){
  (
    cd 'pretrained_model'
    curl https://www.dropbox.com/sh/s1r9gxd9dz4wdkk/AADK6mHvrPtFYo_xD4EmZZdda?dl=0 -L -O -J
    unzip -o pretrained_model.zip
  )
}
