# seed = [24, 42, 123, 321, 340]
# Level 1
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config Danger_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config GoToFav_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config ObjInBox_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config OpenDoor_ours_set --seed 24

# Level 2
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config DangerGoToFav2Room_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config DangerObjInBox_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config DangerOpenDoor_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config GoToFavoriteObjInBox_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config OpenDoorGoToFavorite_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config OpenDoorObjInBox_ours_set --seed 24


# Level 3
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config DangerGoToFavObjInBox_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config DangerOpenDoorObjInBox_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config GoToFavoriteObjInBox_ours_set --seed 24
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config OpenDoorGoToFavoriteObjInBox_ours_set --seed 24

# Level 4
OMP_NUM_THREADS=1 python  ../scripts/train_rl.py --config Boss_ours_set --seed 24