# music_style_transfer_with_instruments
1) revise audioLDM2-music(https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/audioldm2)
  - customize DDIM scheduling: use Z_0(latent vector of reference music) guidance
  - guidance strength decay exponentially as reverse process proceeding


  
2) fine-tune using *instrument specific* LoRAs
  - each of LoRAs are trained with seperate datasets(musics played with only one unique instrument)
  - They would be attached to the U-net/cross_attn layers in inference time
  - Each audioldm2-music which has a LoRA individually would be concatenated to work as an auto-regressive model 
