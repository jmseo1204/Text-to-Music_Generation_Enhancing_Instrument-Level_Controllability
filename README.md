# Paper was uploaded
You can see exact contents of this project on 'Enhancing Instrument-Level Controllability in Text-to-Music Generation for Professional Songwriting Applications.pdf'




# music_style_transfer_with_instruments
1) generate music fused with several instruments using auto-regressive method
  - use audioLDM2-music(https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/audioldm2) as a backbone
  - customize DDIM scheduling: use Z_0(latent vector of reference music) guidance
  - guidance strength decay exponentially as reverse process proceeding
  - deploy output latent vector of generated music to add a new instrument sound on top of it (chain model auto-regressively)



  
2) fine-tune using *instrument specific* LoRAs
  - each of LoRAs are trained with seperate datasets(musics played with only one unique instrument)
  - They would be attached to the U-net/cross_attn layers in inference time
  - Each audioldm2-music which has a LoRA individually would be concatenated to work as an auto-regressive model 



# Scripts description
1) preprocess_dataset.ipynb
- *If you only want to do style transfer, than this script is useless. This is only for LoRA training*
- utilized datasets(https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset/data)
- make audio clips of 10sec and convert to log_mel_spectrogram
- matching each clip to text caption which describe the original music
- devide datasets based on instruments


2) AudioLDM2 with LoRA.ipynb
- LoRA training using peft library

3) music_synthesis_with_instruments.ipynb
- make a style-converted new music which is played by one instrument from the reference music
- mix each style-converted musics with a proper fusion weight
