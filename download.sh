download_dataset(){
  (
    cd 'data'
    wget https://www.dropbox.com/s/xtyotb99gqb72xp/virdo_simul_dataset.pickle
  )
}

download_pretrained(){
  (
    cd 'pretrained_model'
    wget https://www.dropbox.com/s/7h2sqc6ouzlk94y/pretrained_model.zip
    unzip -o pretrained_model.zip
  )
}
