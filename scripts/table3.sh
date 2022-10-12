# Runs for Table 3 (modifications of relaxations)

# hBox-Diag
python3 cert.py --label tweak-mnist-FC-hdiag --seed 10 --mode train-provable --dataset mnist --net FC --train_domain hdiag --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hdiag

python3 cert.py --label tweak-mnist-CONV-hdiag --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain hdiag --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hdiag

# hBox-Diag-C
python3 cert.py --label tweak-mnist-FC-hdiag-c --seed 10 --mode train-provable --dataset mnist --net FC --train_domain hdiag-c --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hdiag-c

python3 cert.py --label tweak-mnist-CONV-hdiag-c --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain hdiag-c --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hdiag-c

# hBox-Switch
python3 cert.py --label tweak-mnist-FC-hswitch --seed 10 --mode train-provable --dataset mnist --net FC --train_domain hswitch --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hswitch

python3 cert.py --label tweak-mnist-CONV-hswitch --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain hswitch --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hswitch

# DeepZ-Box
python3 cert.py --label tweak-mnist-FC-zonobox --seed 10 --mode train-provable --dataset mnist --net FC --train_domain zbox --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zbox

python3 cert.py --label tweak-mnist-CONV-zonobox --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain zbox --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zbox

# DeepZ-Diag
python3 cert.py --label tweak-mnist-FC-zdiag --seed 10 --mode train-provable --dataset mnist --net FC --train_domain zdiag --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zdiag

python3 cert.py --label tweak-mnist-CONV-zdiag --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain zdiag --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zdiag

# DeepZ-Diag-C
python3 cert.py --label tweak-mnist-FC-zdiag-c --seed 10 --mode train-provable --dataset mnist --net FC --train_domain zdiag-c --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zdiag-c

python3 cert.py --label tweak-mnist-CONV-zdiag-c --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain zdiag-c --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zdiag-c

# DeepZ-Switch
python3 cert.py --label tweak-mnist-FC-zswitch --seed 10 --mode train-provable --dataset mnist --net FC --train_domain zswitch --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zswitch

python3 cert.py --label tweak-mnist-CONV-zswitch --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain zswitch --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zswitch

# DeepZ-Soft
python3 cert.py --label tweak-mnist-FC-zsoft --seed 10 --mode train-provable --dataset mnist --net FC --train_domain zono --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zono --soft_slope_gamma 1

python3 cert.py --label tweak-mnist-CONV-zsoft --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain zono --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zono --soft_slope_gamma 1

# DeepZ-IBP (R)
python3 cert.py --label tweak-mnist-FC-zono-ibp --seed 10 --mode train-provable --dataset mnist --net FC --train_domain zono-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zono-ibp

python3 cert.py --label tweak-mnist-CONV-zono-ibp --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain zono-ibp --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zono-ibp

# CROWN-0 
python3 cert.py --label tweak-mnist-FC-deeppoly0 --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 0

python3 cert.py --label tweak-mnist-CONV-deeppoly0 --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 0

# CROWN 0-c
python3 cert.py --label tweak-mnist-FC-deeppoly0c --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 0-c

python3 cert.py --label tweak-mnist-CONV-deeppoly0c --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 0-c

# CROWN-0-Tria
python3 cert.py --label tweak-mnist-FC-deeppoly0tria --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant lower-tria

python3 cert.py --label tweak-mnist-CONV-deeppoly0tria --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant lower-tria

# CROWN-0-Tria-C
python3 cert.py --label tweak-mnist-FC-deeppoly0triaC --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant lower-tria-c

python3 cert.py --label tweak-mnist-CONV-deeppoly0triaC --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant lower-tria-c

# CROWN-1
python3 cert.py --label tweak-mnist-FC-deeppoly1 --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 1

python3 cert.py --label tweak-mnist-CONV-deeppoly1 --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 1

# CROWN 1-c
python3 cert.py --label tweak-mnist-FC-deeppoly1c --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 1-c

python3 cert.py --label tweak-mnist-CONV-deeppoly1c --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant 1-c

# CROWN-1-Tria
python3 cert.py --label tweak-mnist-FC-deeppoly1tria --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant upper-tria

python3 cert.py --label tweak-mnist-CONV-deeppoly1tria --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant upper-tria

# CROWN-1-Tria-C
python3 cert.py --label tweak-mnist-FC-deeppoly1triaC --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant upper-tria-c

python3 cert.py --label tweak-mnist-CONV-deeppoly1triaC --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly --crown_variant upper-tria-c

# CROWN-Soft
python3 cert.py --label tweak-mnist-FC-deeppoly-soft --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly  --soft_slope_gamma 1

python3 cert.py --label tweak-mnist-CONV-deeppoly-soft --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly  --soft_slope_gamma 1

# CROWN-Soft-IBP
python3 cert.py --label tweak-mnist-FC-deeppoly-soft-ibp --seed 12 --mode train-provable --dataset mnist --net FC --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 1 --kappa2 0.5 --verify_domains crown-ibp --soft_slope_gamma 10

python3 cert.py --label tweak-mnist-CONV-deeppoly-soft-ibp --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 1 --kappa2 0.5 --verify_domains crown-ibp --soft_slope_gamma 10