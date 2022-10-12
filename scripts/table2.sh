# Examples for certifiable training with convex relaxations (cert.py)
# Used to generate Table 2; variations possible

# Example eval: evaluating the pretrained model: Box training of FC with eps_test=0.3
python3 cert.py --label pretrained-mnist-FC-box-eval --seed 10 --mode eval --dataset mnist --net FC --verify_domains box --batch_size 100 --eps_test 0.3 --load_model ./pretrained/mnist_FC_box.pt 

### MNIST ###

#Box training of FC with eps_test=0.3
python3 cert.py --label mnist-FC-box --seed 10 --mode train-provable --dataset mnist --net FC --train_domain box --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains box

#hBox training of FC with eps_test=0.3
python3 cert.py --label mnist-FC-hbox --seed 10 --mode train-provable --dataset mnist --net FC --train_domain hbox --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hbox

#CROWN-IBP 1->1 (CROWN-IBP (R)) training of FC with eps_test=0.3
python3 cert.py --label mnist-FC-crownibp1to1 --seed 12 --mode train-provable --dataset mnist --net FC --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 1 --kappa2 0.5 --verify_domains crown-ibp

#DeepZ training of FC with eps_test=0.3
python3 cert.py --label mnist-FC-zono --seed 10 --mode train-provable --dataset mnist --net FC --train_domain zono --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zono

#CROWN training of FC with eps_test=0.3
python3 cert.py --label mnist-FC-deeppoly --seed 10 --mode train-provable --dataset mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly

#CROWN-IBP 1->0 (CROWN-IBP) training of FC with eps_test=0.3
python3 cert.py --label mnist-FC-crownibp1to0 --seed 12 --mode train-provable --dataset mnist --net FC --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --beta2 0 --kappa2 0 --verify_domains box

#Box training of CONV with eps_test=0.3
python3 cert.py --label mnist-CONV-box --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain box --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains box

#hBox training of CONV with eps_test=0.3
python3 cert.py --label mnist-CONV-hbox --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain hbox --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hbox

#CROWN-IBP 1->1 (CROWN-IBP (R)) training of CONV with eps_test=0.3
python3 cert.py --label mnist-CONV-crownibp1to1 --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 1 --kappa2 0.5 --verify_domains crown-ibp

#DeepZ training of CONV with eps_test=0.3
python3 cert.py --label mnist-CONV-zono --seed 10 --mode train-provable --dataset mnist --net CONV --train_domain zono --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zono

#CROWN training of CONV with eps_test=0.3
python3 cert.py --label mnist-CONV-deeppoly --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly

#CROWN-IBP 1->0 (CROWN-IBP) training of CONV with eps_test=0.3
python3 cert.py --label mnist-CONV-crownibp1to0 --seed 12 --mode train-provable --dataset mnist --net CONV --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 0 --kappa2 0 --verify_domains box

### FashionMNIST ###

#Box training of FC with eps_test=0.3
python3 cert.py --label fashion-mnist-FC-box --seed 10 --mode train-provable --dataset fashion-mnist --net FC --train_domain box --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains box

#hBox training of FC with eps_test=0.3
python3 cert.py --label fashion-mnist-FC-hbox --seed 10 --mode train-provable --dataset fashion-mnist --net FC --train_domain hbox --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hbox

#CROWN-IBP 1->1 (CROWN-IBP (R)) training of FC with eps_test=0.3
python3 cert.py --label fashion-mnist-FC-crownibp1to1 --seed 12 --mode train-provable --dataset fashion-mnist --net FC --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 1 --kappa2 0.5 --verify_domains crown-ibp

#DeepZ training of FC with eps_test=0.3
python3 cert.py --label fashion-mnist-FC-zono --seed 10 --mode train-provable --dataset fashion-mnist --net FC --train_domain zono --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C eye --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --kappa2 0 --verify_domains zono

#CROWN training of FC with eps_test=0.3
python3 cert.py --label fashion-mnist-FC-deeppoly --seed 10 --mode train-provable --dataset fashion-mnist --net FC --train_domain deeppoly --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly

#CROWN-IBP 1->0 (CROWN-IBP) training of FC with eps_test=0.3
python3 cert.py --label fashion-mnist-FC-crownibp1to0 --seed 12 --mode train-provable --dataset fashion-mnist --net FC --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --beta2 0 --kappa2 0 --verify_domains box

#Box training of CONV with eps_test=0.3
python3 cert.py --label fashion-mnist-CONV-box --seed 10 --mode train-provable --dataset fashion-mnist --net CONV --train_domain box --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains box

#hBox training of CONV with eps_test=0.3
python3 cert.py --label fashion-mnist-CONV-hbox --seed 10 --mode train-provable --dataset fashion-mnist --net CONV --train_domain hbox --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains hbox

#CROWN-IBP 1->1 (CROWN-IBP (R)) training of CONV with eps_test=0.3
python3 cert.py --label fashion-mnist-CONV-crownibp1to1 --seed 10 --mode train-provable --dataset fashion-mnist --net CONV --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 1 --kappa2 0.5 --verify_domains crown-ibp

#DeepZ training of CONV with eps_test=0.3
python3 cert.py --label fashion-mnist-CONV-zono --seed 10 --mode train-provable --dataset fashion-mnist --net CONV --train_domain zono --batch_size 50 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-05 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains zono

#CROWN training of CONV with eps_test=0.3
python3 cert.py --label fashion-mnist-CONV-deeppoly --seed 12 --mode train-provable --dataset fashion-mnist --net CONV --train_domain deeppoly --batch_size 50 --eps_train 0.3 --eps_test 0.3 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 100 --lr 0.0005 --lr_milestones 130,190 --kappa2 0 --verify_domains deeppoly

#CROWN-IBP 1->0 (CROWN-IBP) training of CONV with eps_test=0.3
python3 cert.py --label fashion-mnist-CONV-crownibp1to0 --seed 12 --mode train-provable --dataset fashion-mnist --net CONV --train_domain crown-ibp --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 10 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --beta2 0 --kappa2 0 --verify_domains box

### SVHN ### 

# Box training of CONV with eps_test=8/255
python3 cert.py --label svhn-CONV-box --seed 1 --mode train-provable --dataset svhn --net CONV --train_domain box --batch_size 50 --eps_train 0.03451 --eps_test 0.03137 --reg L1 --reg_lambda 5e-5 --C diffs --n_epochs 2200 --warmup_epochs 10 --mix_epochs 1100 --lr 0.0005 --lr_milestones 0.6,0.9 --kappa2 0 --verify_domains box

# CROWN-IBP (R) training of CONV with eps_test=8/255
python3 cert.py --label svhn-CONV-crownibp-r --seed 1 --mode train-provable --dataset svhn --net CONV --train_domain crown-ibp --batch_size 50 --eps_train 0.03451 --eps_test 0.03137 --C diffs --n_epochs 2200 --warmup_epochs 10 --mix_epochs 1100 --lr 0.001 --lr_milestones 0.6,0.9 --kappa2 0.5 --beta2 1 --verify_domains crown-ibp

# CROWN-IBP training of CONV with eps_test=8/255
python3 cert.py --label svhn-CONV-crownibp --seed 1 --mode train-provable --dataset svhn --net CONV --train_domain crown-ibp --batch_size 50 --eps_train 0.03451 --eps_test 0.03137 --C diffs --n_epochs 2200 --warmup_epochs 10 --mix_epochs 1100 --lr 0.0005 --lr_milestones 0.6,0.9 --kappa2 0 --beta2 0 --verify_domains box

# DeepZ training of CONV with eps_test=8/255
python3 cert.py --label svhn-CONV-deepz --seed 1 --mode train-provable --dataset svhn --net CONV --train_domain zono --batch_size 20 --eps_train 0.03137 --eps_test 0.03137 --reg L1 --reg_lambda 5e-6 --C diffs --n_epochs 100 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --tenhalf --kappa2 0 --verify_domains zono --eps1 0.001

# hBox training of CONV with eps_test=8/255
python3 cert.py --label svhn-CONV-hbox --seed 1 --mode train-provable --dataset svhn --net CONV --train_domain hbox --batch_size 20 --eps_train 0.03137 --eps_test 0.03137 --reg L1 --reg_lambda 5e-6 --C diffs --n_epochs 100 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --tenhalf --kappa2 0 --verify_domains hbox --eps1 0.001

# CROWN training of CONV with eps_test=8/255
python3 cert.py --label svhn-CONV-crown --seed 1 --mode train-provable --dataset svhn --net CONV --train_domain deeppoly --batch_size 20 --eps_train 0.03137 --eps_test 0.03137 --reg L1 --reg_lambda 5e-6 --C diffs --n_epochs 100 --warmup_epochs 0 --mix_epochs 50 --lr 0.001 --tenhalf --kappa2 0 --verify_domains deeppoly --eps1 0.001

### CIFAR-10 ###

# Box training of CONVPLUS with eps_test=8/255
python3 cert.py --label cifar10-CONVPLUS-box --seed 1 --mode train-provable --dataset cifar10 --net CONVPLUS --train_domain box --batch_size 512 --eps_train 0.03451 --eps_test 0.03137 --C diffs --n_epochs 3200 --warmup_epochs 320 --mix_epochs 1600 --lr 0.0005 --lr_milestones 2600,3040 --kappa1 1 --kappa2 0 --verify_domains box --do_transform

# CROWN-IBP (R) training of CONVPLUS with eps_test=8/255
python3 cert.py --label cifar10-CONVPLUS-crownibp-r --seed 1 --mode train-provable --dataset cifar10 --net CONVPLUS --train_domain crown-ibp --batch_size 512 --eps_train 0.03451 --eps_test 0.03137 --C diffs --n_epochs 3200 --warmup_epochs 320 --mix_epochs 1600 --lr 0.0005 --lr_milestones 2600,3040 --kappa1 0 --kappa2 0 --beta2 1 --verify_domains crown-ibp --do_transform

# CROWN-IBP training of CONVPLUS with eps_test=8/255
python3 cert.py --label cifar10-CONVPLUS-crownibp --seed 1 --mode train-provable --dataset cifar10 --net CONVPLUS --train_domain crown-ibp --batch_size 512 --eps_train 0.03451 --eps_test 0.03137 --C diffs --n_epochs 3200 --warmup_epochs 320 --mix_epochs 1600 --lr 0.0005 --lr_milestones 2600,3040 --kappa1 0 --kappa2 0 --beta2 0 --verify_domains box --do_transform

# DeepZ training of CONVPLUS with eps_test=8/255
python3 cert.py --label cifar10-CONVPLUS-deepz --seed 1 --mode train-provable --dataset cifar10 --net CONVPLUS --train_domain cauchy-zono --batch_size 50 --eps_train 0.03137 --eps_test 0.03137 --C eye --n_epochs 240 --warmup_epochs 0 --mix_epochs 80 --lr 0.0005 --tenhalf --kappa1 1 --kappa2 0 --verify_domains cauchy-zono --eps1 0.001

# hBox training of CONVPLUS with eps_test=8/255
python3 cert.py --label cifar10-CONVPLUS-hbox --seed 1 --mode train-provable --dataset cifar10 --net CONVPLUS --train_domain cauchy-hbox --batch_size 50 --eps_train 0.03137 --eps_test 0.03137 --C eye --n_epochs 240 --warmup_epochs 0 --mix_epochs 80 --lr 0.0005 --tenhalf --kappa1 1 --kappa2 0 --verify_domains cauchy-hbox --eps1 0.001
