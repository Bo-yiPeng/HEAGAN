
python fully_train_arch.py \
--gpu_ids 7 \
--num_workers 8 \
--gen_bs 128 \
--dis_bs 64 \
--dataset stl10 \
--bottom_width 6 \
--img_size 48 \
--max_epoch_G 121 \
--n_critic 5 \
--arch arch_stl10 \
--draw_arch False \
--genotypes_exp stl10_D.npy \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--val_freq 5 \
--num_eval_imgs 50000 \
--exp_name arch_train_stl10 \
--data_path /data/datasets/stl10 \
--cr 1 \
--genotype_of_G arch_searchG_cifar10_2025_07_20_11_18_55/Model/best_fid_gen.npy \
--use_basemodel_D False
# --genotype_of_G best_gen_0.npy \