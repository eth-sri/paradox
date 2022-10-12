# Generates scripts used in Table 4 (loose parametrized relaxations)

template="python3 cert.py --label {} --seed {} --mode train-provable --dataset mnist --net CONV --train_domain box --batch_size 100 --eps_train 0.3 --eps_test 0.3 --reg L1 --reg_lambda 5e-06 --C diffs --n_epochs 200 --warmup_epochs 0 --mix_epochs 50 --lr 0.0005 --lr_milestones 130,190 --kappa2 0.5 --verify_domains box {}"

def getjob(mode, omega, seed, F=None):
    if mode == 'C':
        label = f'parametrized-C-{omega}'
        ad =  f'--loosebox_widen {omega}'
    else:
        label = f'parametrized-DC-{omega}-{F}' 
        ad = f'--loosebox_round {omega} --dcs_per_one {F}'
    cmd = template.format(label, seed, ad) 
    print(cmd)

for seed in [101, 102, 103, 104]:
    for omega in [0.01, 0.5, 1, 1.5, 2, 3, 5]:
        for F in [0, 1, 10, 100, 300, 1000]:
            if F == 0:
                getjob('C', omega, seed)
            else:
                getjob('DC', omega, seed, F=F)