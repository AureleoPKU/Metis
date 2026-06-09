#convert lerobot only with real robot data to rlds, totally 932 episodes
# python lerobot2rlds.py \
#    --src-dir /home/zhaojunkai/DATA/wild_move_to \
#    --output-dir /home/zhaojunkai/DATA/transfored_rlds \
#    --task-name wild_move_to


#convert lerobot with synthstic vl to rlds
python lerobot2rlds.py \
   --src-dir /home/zhaojunkai/DATA/wild_move_to \
   --output-dir /share/project/zjk/DATA \
   --task-name wild_move_to_6932