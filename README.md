Train a ResNet18 model on CIFAR10:
*python train.py --data_root dataset_path --dataset CIFAR10 --weight_decay 3.5e-3 --lr 0.01 --batch_size 128 --epoch 120 --model resnet18*

Train a WRN-34-10 model on CIFAR10:
*python train.py --data_root dataset_path --dataset CIFAR10 --weight_decay 5e-4 --lr 0.1 --batch_size 128 --epoch 60 --model wrn-34-10*

Test a model under PGD-50 attack:
*python test.py --weights checkpoint_path --attack PGD --step 50 --dataset dataset_name --model model_name*