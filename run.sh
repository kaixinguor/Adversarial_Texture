
##### RCA
# python evaluation_texture.py --method RCA --load_path pretrained/RCA2.npy --suffix yolov2_RCA2 --prepare_data

# python evaluation_texture.py --method RCA --load_path training_results/yolov2_RCA_result/patch2000.npy --suffix yolov2_RCA



##### TCA
python evaluation_texture.py --method TCA --load_path pretrained/TCA.npy # --prepare_data

# python evaluation_texture.py --method TCA --load_path training_results/yolov2_TCA_result/patch_epoch800.npy # --prepare_data




##### EGA

# python evaluation_texture.py --method EGA --load_path pretrained/EGA.pkl --prepare_data

# python evaluation_texture.py --method EGA --load_path training_results/yolov2_TCEGA_result/yolov2_TCEGA_epoch2000.pkl


##### TC-EGA

# python evaluation_texture.py --method TCEGA --load_path pretrained/EGA.pkl --load_path_z pretrained/TCEGA_z.npy --prepare_data

# python evaluation_texture.py --method TCEGA --load_path training_results/yolov2_TCEGA_result/yolov2_TCEGA_epoch2000.pkl