import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import random
import heapq
from utils.Single_objective_sort import single_sort
from utils.nsga import NSGA_2
from utils.utils import count_parameters_in_MB
from archs.fully_super_network import Generator
from pytorch_gan_metrics import get_inception_score_and_fid, get_fid, get_inception_score
from torchvision.utils import make_grid, save_image
from pyswarm import pso  # 如果使用PSO库（例如`pyswarm`）来实现PSO
# 添加adaptive损失函数导入
from robust_loss_pytorch.adaptive import AdaptiveLossFunction
logger = logging.getLogger(__name__)

try_balance = False
Extra_subnet = 2
num_normal = 7


class GenTrainer():
    def __init__(self, args, gen_net, dis_net, gen_optimizer, dis_optimizer, train_loader, gan_alg, dis_genotype,
                 gen_fixed_genotype):
        self.args = args
        self.gen_net = gen_net
        self.dis_net = dis_net
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.train_loader = train_loader
        self.gan_alg = gan_alg
        self.dis_genotype = dis_genotype
        self.gen_fixed_genotype = gen_fixed_genotype
        self.historical_population = {}
        self.genotypes = np.stack([gan_alg.search() for i in range(args.num_individual)],
                                  axis=0)
        self.best_is_evo = 0
        self.best_fid_evo = 999
        # 初始化adaptive损失函数
        self.adaptive_loss_fn = AdaptiveLossFunction(
            num_dims=1,              # 判别器输出维度（通常为1）
            float_dtype=np.float32,
            device=0,                # 直接使用GPU索引0
            alpha_lo=0.1,            # 可调参数：损失形状下限
            alpha_hi=1.9,            # 可调参数：损失形状上限
            scale_lo=1e-5,           # 可调参数：尺度下限
            scale_init=2.0           # 初始尺度
        )
        # 添加全局最优跟踪
        self.global_best_genotype = None
        self.global_best_fitness = -np.inf
        self.fitness_history = []
    def train(self, epoch, writer_dict, fid_stat, schedulers=None):
        writer = writer_dict['writer']
        gen_step = 0

        # train mode
        gen_net = self.gen_net.train()
        dis_net = self.dis_net.train()


        # number of g
        num_g = int(len(self.gan_alg.Normal_G_fixed[0]) // self.args.num_op_g)
        genotype_G = np.stack([self.gan_alg.search() for i in range(num_g)],
                              axis=0)
        # 添加调试打印
        print("genotype_G type:", type(genotype_G))  # 应为 int
        print("genotype_G value:", genotype_G)  # 应为单个整数
        for iter_idx, (imgs, _) in enumerate(tqdm(self.train_loader)):
            global_steps = writer_dict['train_global_steps']

            real_imgs = imgs.type(torch.cuda.FloatTensor)
            i = np.random.randint(0, self.args.num_individual, 1)[0]
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (imgs.shape[0], self.args.latent_dim)))

            # # ============================train D-D================================
            # self.dis_optimizer.zero_grad()
            # # real_validity = dis_net(real_imgs, genotype_D)
            # real_validity = dis_net(real_imgs)
            # n_i = iter_idx % num_g
            # fake_imgs = gen_net(z, genotype_G[n_i]).detach()

            # assert fake_imgs.size() == real_imgs.size()
            # # fake_validity = dis_net(fake_imgs, genotype_D)
            # fake_validity = dis_net(fake_imgs)
            # # Hinge loss
            # # d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
            # #          torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            # # d_loss.backward()
            # # 修复adaptive损失的输入格式
            # # 检查并调整输入张量的形状
            # print(f"real_validity shape: {real_validity.shape}")
            # print(f"fake_validity shape: {fake_validity.shape}")
            
            # # 确保输入张量是 [batch_size, 1] 的形状
            # if real_validity.dim() == 1:
            #     real_validity = real_validity.unsqueeze(1)
            # if fake_validity.dim() == 1:
            #     fake_validity = fake_validity.unsqueeze(1)
            
            # # 方法1：分别计算真实和虚假样本的损失
            # d_loss_real = self.adaptive_loss_fn.lossfun(real_validity - 1.0).mean()  # 期望真实样本输出为1
            # d_loss_fake = self.adaptive_loss_fn.lossfun(-fake_validity - 1.0).mean()  # 期望虚假样本输出为-1
            # d_loss = d_loss_real + d_loss_fake
            # d_loss.backward()
            # self.dis_optimizer.step()
            # writer.add_scalar('d_loss', d_loss.item(), global_steps)

            # # ===============================train G-G===============================
            # if global_steps % self.args.n_critic == 0:
            #     self.gen_optimizer.zero_grad()
            #     gen_z = torch.cuda.FloatTensor(np.random.normal(
            #         0, 1, (self.args.gen_bs, self.args.latent_dim)))
            #     for k in range(num_g):
            #         gen_imgs = gen_net(gen_z, genotype_G[k])
            #         # gen_imgs = gen_net(gen_z, self.gen_fixed_genotype)
            #         # fake_validity = dis_net(gen_imgs, genotype_D)
            #         fake_validity = dis_net(gen_imgs)
            #         # Hinge loss
            #         # g_loss = -torch.mean(fake_validity)
            #         # g_loss.backward()
            #         # 新的adaptive损失：
            #         # 生成器损失需要特殊处理（例如最大化判别器输出）
            #         # 确保生成器损失的输入格式正确
            #         if fake_validity.dim() == 1:
            #             fake_validity = fake_validity.unsqueeze(1)
                    
            #         # 生成器损失：期望判别器输出为正值（真实）
            #         g_loss = self.adaptive_loss_fn.lossfun(-fake_validity).mean()
            #         g_loss.backward()
            #     self.gen_optimizer.step()
            #     genotype_G = np.stack([self.gan_alg.search() for i in range(num_g)],
            #                           axis=0)
            # ============================train D-D================================
            self.dis_optimizer.zero_grad()
            # real_validity = dis_net(real_imgs, genotype_D)
            real_validity = dis_net(real_imgs)
            n_i = iter_idx % num_g
            fake_imgs = gen_net(z, genotype_G[n_i]).detach()

            assert fake_imgs.size() == real_imgs.size()
            # fake_validity = dis_net(fake_imgs, genotype_D)
            fake_validity = dis_net(fake_imgs)
            # Hinge loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                     torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            self.dis_optimizer.step()
            writer.add_scalar('d_loss', d_loss.item(), global_steps)

            # ===============================train G-G===============================
            if global_steps % self.args.n_critic == 0:
                self.gen_optimizer.zero_grad()
                gen_z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.gen_bs, self.args.latent_dim)))
                for k in range(num_g):
                    gen_imgs = gen_net(gen_z, genotype_G[k])
                    # gen_imgs = gen_net(gen_z, self.gen_fixed_genotype)
                    # fake_validity = dis_net(gen_imgs, genotype_D)
                    fake_validity = dis_net(gen_imgs)
                    # Hinge loss
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                self.gen_optimizer.step()
                genotype_G = np.stack([self.gan_alg.search() for i in range(num_g)],
                                      axis=0)
                # learning rate
                if schedulers:
                    gen_scheduler, dis_scheduler = schedulers
                    g_lr = gen_scheduler.step(global_steps)
                    d_lr = dis_scheduler.step(global_steps)
                    writer.add_scalar('LR/g_lr', g_lr, global_steps)
                    writer.add_scalar('LR/d_lr', d_lr, global_steps)
                writer.add_scalar('g_loss', g_loss.item(), global_steps)
                gen_step += 1
            # verbose
            if gen_step and iter_idx % self.args.print_freq == 0:
                tqdm.write(
                    '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' %
                    (epoch, self.args.max_epoch_D, iter_idx % len(self.train_loader), len(self.train_loader),
                     d_loss.item(), g_loss.item()))
            writer_dict['train_global_steps'] = global_steps + 1
        if epoch % 5 == 0:
            is_warm, is_std_warm, fid_warm = self.validate(self.gen_fixed_genotype, fid_stat)

            logger.info(
                f'IS_warm: {round(is_warm, 3)}, FID_warm: {round(fid_warm, 3)},@ epoch {epoch}.')

    def get_E_DGz(self, geno):
        gen_net = self.gen_net.eval()
        dis_net = self.dis_net.eval()
        gen_z = torch.cuda.FloatTensor(np.random.normal(
            0, 1, (100, self.args.latent_dim)))
        gen_imgs = gen_net(gen_z, geno)
        fake_validity = dis_net(gen_imgs)
        # Hinge loss
        E_DGz = torch.mean(fake_validity)
        return E_DGz
    def update_global_best(self, genotype, is_value, fid_value):
        """更新全局最优解"""
        fitness = 0.5 * is_value / 5 - 0.5 * fid_value / 90  # 复合适应度
        
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_genotype = genotype.copy()
            
            # 同时更新gan_alg中的全局最优
            self.gan_alg.update_global_best(genotype, fitness)
            
            print(f"发现全局最优: IS={is_value:.3f}, FID={fid_value:.3f}, fitness={fitness:.3f}")
            
            # 保存全局最优
            file_path = os.path.join(self.args.path_helper['ckpt_path'], "global_best_genotype.npy")
            np.save(file_path, genotype)
            
            return True
        return False
    def Judge_relevance(self, fid_stat):
        Random_population = np.stack([self.gan_alg.search() for i in range(100)],
                                     axis=0)
        is_values, fid_values, D_Gz = np.zeros(len(Random_population)), np.zeros(
            len(Random_population)), np.zeros(len(Random_population))

        for idx, genotype_G in enumerate(tqdm(Random_population)):
            is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            D_Gz[idx] = self.get_E_DGz(genotype_G)
        logger.info(f'IS: {is_values}, FID: {fid_values}, D_G(z): {D_Gz}.')

    def search_evol_arch(self, epoch, fid_stat):
        offsprings = self.gen_offspring(self.genotypes)
        genotypes = np.concatenate((self.genotypes, offsprings), axis=0)
        is_values, fid_values, params = np.zeros(len(genotypes)), np.zeros(
            len(genotypes)), np.zeros(len(genotypes))
        keep_N, selected_N = len(offsprings), self.args.num_selected
        for idx, genotype_G in enumerate(tqdm(genotypes)):
            is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)
            param_szie = count_parameters_in_MB(
                Generator(self.args, genotype_G))
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_szie

        logger.info(f'mean_IS_values: {np.mean(is_values)}, mean_FID_values: {np.mean(fid_values)},@ epoch {epoch}.')
        obj = [fid_values, params]
        keep, selected = CARS_NSGA(is_values, obj, keep_N), CARS_NSGA(
            is_values, obj, selected_N)
        for i in selected:
            logger.info(
                f'genotypes_{i}, IS_values: {is_values[i]}, FID_values: {fid_values[i]}, param_szie: {params[i]}|| @ epoch {epoch}.')
        self.genotypes = genotypes[keep]
        return genotypes[selected]

    def judege_model_size(self, genotype_G, limit):
        '''
        Test whether the model size constraint is met
        '''
        param_szie = count_parameters_in_MB(Generator(self.args, genotype_G))
        if param_szie <= limit:
            return True
        return False
    def pso_optimize(self, population, fid_stat):
        # PSO优化函数
        def fitness_function(x):
            genotype = self.vector_to_genotype(x)
            is_value, _, fid_value = self.validate(genotype, fid_stat)
            return 0.5 * is_value / 5 - 0.5 * fid_value / 90  # 多目标优化（最小化FID，最大化IS）

        # 定义PSO参数
        lb = [0] * (3*7)  # 基因型维度
        ub = [6] * (3*7)  # 各基因位的取值范围
        
        # 运行PSO
        best_pos, best_fit = pso(fitness_function, lb, ub, 
                                swarmsize=20, maxiter=10, 
                                debug=True)
        
        return self.vector_to_genotype(best_pos)

    def vector_to_genotype(self, vector):
        # 将PSO向量转换为基因型
        return np.array(vector).reshape(3,7).astype(int)

    def genotype_to_vector(self, genotype):
        # 将基因型转换为PSO向量
        return genotype.flatten()
    def my_search_evolv2(self, parents, fid_stat, num):
        '''
        input: parent
        num:
        return:
        '''
        offsprings = self.gen_offspring(parents)
        genotypes = np.concatenate((parents, offsprings), axis=0)
        is_values, fid_values, params = np.zeros(len(genotypes)), np.zeros(
            len(genotypes)), np.zeros(len(genotypes))
        keep_N, selected_N = len(offsprings), self.args.num_selected
        for idx, genotype_G in enumerate(tqdm(genotypes)):
            encode_G = self.gan_alg.encode(genotype_G)
            if encode_G in self.historical_population:
                data = self.historical_population[encode_G]
                is_value, fid_value, param_szie = data[0], data[1], data[2]
            else:
                is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)  # Obtain subnetwork performance
                param_szie = count_parameters_in_MB(Generator(self.args, genotype_G))
                self.historical_population[encode_G] = [is_value, fid_value, param_szie]
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_szie
            if is_value > self.best_is_evo:
                self.best_is_evo = is_value
                file_path = os.path.join(self.args.path_helper['ckpt_path'],
                                         "best_is_gen_.npy")
                np.save(file_path, genotype_G)
            if fid_value < self.best_fid_evo:
                self.best_fid_evo = fid_value
                file_path = os.path.join(self.args.path_helper['ckpt_path'],
                                         "best_fid_gen.npy")
                np.save(file_path, genotype_G)
        sum_is, sum_fid, sum_param = 0, 0, 0
        for i in range(keep_N):
            sum_is += is_values[i]
            sum_fid += fid_values[i]
            sum_param += params[i]
        avg_is = sum_is / keep_N
        avg_fid = sum_fid / keep_N
        avg_params = sum_param / keep_N
        logger.info(
            f'mean_IS: {round(avg_is, 3)}, mean_FID: {round(avg_fid, 3)}, max_IS: {round(np.max(is_values), 3)}, min_FID: {round(np.min(fid_values), 3)},mean_params: {round(avg_params, 3)}@ evolutionary_algebra: {num}.')

        keep = NSGA_2(is_values.tolist(), (-fid_values).tolist(), keep_N)
        selected = NSGA_2(is_values.tolist(), (-fid_values).tolist(), selected_N)

        # con
        # a = np.array(is_values)
        # b = np.array(fid_values)
        # weigh = a/10-b/180
        # keep = NSGA_2(weigh.tolist(), (-params).tolist(), keep_N)
        # selected = NSGA_2(weigh.tolist(), (-params).tolist(), selected_N)
        # keep = single_sort(is_values, keep_N, False)
        # selected = single_sort(is_values, selected_N, False)

        # keep = single_sort(fid_values, keep_N, True)
        # selected = single_sort(fid_values, selected_N, True)

        # a = np.array(is_values)
        # b = np.array(fid_values)
        # weigh = a/10-b/180
        # keep = single_sort(weigh, keep_N, False)
        # selected = single_sort(weigh, selected_N, False)
        for i in selected:
            logger.info(
                f'genotypes_{i}, IS: {round(is_values[i], 3)}, FID: {round(fid_values[i], 3)}, param_size: {round(params[i], 3)}.')

        self.genotypes = genotypes[keep]
        return genotypes[keep], genotypes[selected], np.mean(is_values[keep]), np.mean(fid_values[keep]), is_values[
            keep], fid_values[keep]
        # # 1. 如果有全局最优，将其加入父代
        # if self.global_best_genotype is not None:
        #     parents = np.vstack([parents, self.global_best_genotype.reshape(1, 3, 7)])
        
        # # 2. 生成子代
        # offsprings = self.gen_offspring(parents)
        # genotypes = np.concatenate((parents, offsprings), axis=0)
        
        # is_values, fid_values, params = np.zeros(len(genotypes)), np.zeros(len(genotypes)), np.zeros(len(genotypes))
        # keep_N, selected_N = len(offsprings), self.args.num_selected
        
        # # 3. 评估所有个体
        # for idx, genotype_G in enumerate(tqdm(genotypes, desc=f"Evaluating generation {num}")):
        #     encode_G = self.gan_alg.encode(genotype_G)
        #     if encode_G in self.historical_population:
        #         data = self.historical_population[encode_G]
        #         is_value, fid_value, param_szie = data[0], data[1], data[2]
        #     else:
        #         is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)
        #         param_szie = count_parameters_in_MB(Generator(self.args, genotype_G))
        #         self.historical_population[encode_G] = [is_value, fid_value, param_szie]
            
        #     is_values[idx] = is_value
        #     fid_values[idx] = fid_value
        #     params[idx] = param_szie
            
        #     # 4. 更新全局最优
        #     self.update_global_best(genotype_G, is_value, fid_value)
            
        #     # 更新单独的最优记录
        #     if is_value > self.best_is_evo:
        #         self.best_is_evo = is_value
        #         file_path = os.path.join(self.args.path_helper['ckpt_path'], "best_is_gen_.npy")
        #         np.save(file_path, genotype_G)
        #     if fid_value < self.best_fid_evo:
        #         self.best_fid_evo = fid_value
        #         file_path = os.path.join(self.args.path_helper['ckpt_path'], "best_fid_gen.npy")
        #         np.save(file_path, genotype_G)
        
        # # 5. 记录历史
        # current_best_fitness = 0.5 * np.max(is_values) / 5 - 0.5 * np.min(fid_values) / 90
        # self.fitness_history.append((num, np.mean(is_values), np.mean(fid_values), current_best_fitness))
        
        # # 6. 选择策略 - 使用综合得分
        # composite_scores = 0.5 * np.array(is_values) / 5 - 0.5 * np.array(fid_values) / 90
        # sorted_indices = np.argsort(composite_scores)[::-1]
        
        # keep_N = min(keep_N, len(genotypes))
        # selected_N = max(1, min(self.args.num_selected, len(genotypes)))
        
        # keep = sorted_indices[:keep_N].tolist()
        # selected = sorted_indices[:selected_N].tolist()
        
        # # 7. 日志输出
        # avg_is = np.mean(is_values[keep])
        # avg_fid = np.mean(fid_values[keep])
        # avg_params = np.mean(params[keep])
        
        # logger.info(f'第{num}代: mean_IS: {avg_is:.3f}, mean_FID: {avg_fid:.3f}, '
        #            f'max_IS: {np.max(is_values):.3f}, min_FID: {np.min(fid_values):.3f}, '
        #            f'mean_params: {avg_params:.3f}, global_best_fitness: {self.global_best_fitness:.3f}')
        
        # for i in selected:
        #     logger.info(f'genotypes_{i}, IS: {is_values[i]:.3f}, FID: {fid_values[i]:.3f}, '
        #                f'composite_score: {composite_scores[i]:.3f}, param_size: {params[i]:.3f}')
        
        # # 8. 更新种群
        # self.genotypes = genotypes[keep]
        
        # return genotypes[keep], genotypes[selected], avg_is, avg_fid, is_values[keep], fid_values[keep]
    # 进化算法搜索方法
    # def my_search_evol(self, parents, fid_stat, num):
    #     offsprings = self.gen_offspring(parents)
    #     genotypes = np.concatenate((parents, offsprings), axis=0)
    #     is_values, fid_values, params = np.zeros(len(genotypes)), np.zeros(
    #         len(genotypes)), np.zeros(len(genotypes))
    #     keep_N, selected_N = len(offsprings), self.args.num_selected
    #     for idx, genotype_G in enumerate(tqdm(genotypes)):
    #         encode_G = self.gan_alg.encode(genotype_G)
    #         if encode_G in self.historical_population:
    #             data = self.historical_population[encode_G]
    #             is_value, fid_value, param_szie = data[0], data[1], data[2]
    #         else:
    #             is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)
    #             param_szie = count_parameters_in_MB(Generator(self.args, genotype_G))
    #             self.historical_population[encode_G] = [is_value, fid_value, param_szie]
    #         is_values[idx] = is_value
    #         fid_values[idx] = fid_value
    #         params[idx] = param_szie
    #         if is_value > self.best_is_evo:
    #             self.best_is_evo = is_value
    #             file_path = os.path.join(self.args.path_helper['ckpt_path'],
    #                                      "best_is_gen_.npy")
    #             np.save(file_path, genotype_G)
    #         if fid_value < self.best_fid_evo:
    #             self.best_fid_evo = fid_value
    #             file_path = os.path.join(self.args.path_helper['ckpt_path'],
    #                                      "best_fid_gen.npy")
    #             np.save(file_path, genotype_G)
    #     sum_is, sum_fid, sum_param = 0, 0, 0
    #     for i in range(keep_N):
    #         sum_is += is_values[i]
    #         sum_fid += fid_values[i]
    #         sum_param += params[i]
    #     avg_is = sum_is / keep_N
    #     avg_fid = sum_fid / keep_N
    #     avg_params = sum_param / keep_N
    #     logger.info(
    #         f'mean_IS: {round(avg_is, 3)}, mean_FID: {round(avg_fid, 3)}, max_IS: {round(np.max(is_values), 3)}, min_FID: {round(np.min(fid_values), 3)},mean_params: {round(avg_params, 3)}@ evolutionary_algebra: {num}.')
    #     # 使用PSO优化前10%的个体
    #     # top_indices = np.argsort(is_values)[-len(genotypes)//10:]
    #     # for idx in top_indices:
    #     #     optimized = self.pso_optimize(
    #     #         self.genotype_to_vector(genotypes[idx]), 
    #     #         fid_stat
    #     #     )
    #     #     genotypes[idx] = optimized
    #     #选择
    #     keep = NSGA_2(is_values.tolist(), (-fid_values).tolist(), keep_N)
    #     selected = NSGA_2(is_values.tolist(), (-fid_values).tolist(), selected_N)
    #     self.genotypes = genotypes[keep]
    #     return genotypes[keep], genotypes[selected], np.mean(is_values[keep]), np.mean(fid_values[keep]), np.max(
    #         is_values[keep]), np.min(fid_values[keep])
    #     # 计算综合得分：IS - FID
    #     composite_scores = 0.5 * np.array(is_values) / 5 - 0.5 * np.array(fid_values) / 90
        
    #     # 添加调试信息
    #     print(f"第 {num} 代: IS范围={np.min(is_values):.3f}-{np.max(is_values):.3f}, FID范围={np.min(fid_values):.3f}-{np.max(fid_values):.3f}")
    #     print(f"综合得分(IS-FID)范围: {np.min(composite_scores):.3f}-{np.max(composite_scores):.3f}")
        
    #     # 由于IS-FID为负值，且绝对值越小越好，所以我们要选择数值最大的（最接近0的）
    #     # 即按降序排列，选择最大的值（绝对值最小的负数）
    #     sorted_indices = np.argsort(composite_scores)[::-1]  # 降序排列，选择最大值
        
    #     # 确保选择数量不超过总个体数
    #     keep_N = min(keep_N, len(genotypes))
    #     # 修改选择策略：选择前30%作为精英
    #     selected_N = max(1, int(len(genotypes) * 0.3))  # 确保至少选择1个个体
    #     selected_N = min(selected_N, len(genotypes))      # 不超过总个体数
        
    #     # 选择前keep_N个作为保留，前30%个作为精英
    #     keep = sorted_indices[:keep_N].tolist()
    #     selected = sorted_indices[:selected_N].tolist()
        
    #     print(f"选择结果: keep={len(keep)}个, selected={len(selected)}个")
    #     print(f"最佳得分(IS-FID): {composite_scores[sorted_indices[0]]:.3f}")
    #     print(f"最差保留得分(IS-FID): {composite_scores[sorted_indices[keep_N-1]]:.3f}")
        
    #     # 记录选择的个体信息
    #     for i in selected:
    #         logger.info(
    #             f'genotypes_{i}, IS: {round(is_values[i], 3)}, FID: {round(fid_values[i], 3)}, composite_score(IS-FID): {round(composite_scores[i], 3)}, param_size: {round(params[i], 3)}.')

    #     self.genotypes = genotypes[keep]
    #     return genotypes[keep], genotypes[selected], np.mean(is_values[keep]), np.mean(fid_values[keep]), is_values[
    #         keep], fid_values[keep]
    # PSO进化算法
    def my_search_evol(self, parents, fid_stat, num):
        '''
        使用PSO算法进行架构搜索
        '''
        offsprings = self.gen_offspring(parents)
        genotypes = np.concatenate((parents, offsprings), axis=0)
        is_values, fid_values, params = np.zeros(len(genotypes)), np.zeros(
            len(genotypes)), np.zeros(len(genotypes))
        keep_N, selected_N = len(offsprings), self.args.num_selected
        
        # 评估所有个体
        for idx, genotype_G in enumerate(tqdm(genotypes)):
            encode_G = self.gan_alg.encode(genotype_G)
            if encode_G in self.historical_population:
                data = self.historical_population[encode_G]
                is_value, fid_value, param_szie = data[0], data[1], data[2]
            else:
                is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)
                param_szie = count_parameters_in_MB(Generator(self.args, genotype_G))
                self.historical_population[encode_G] = [is_value, fid_value, param_szie]
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_szie
            
            if is_value > self.best_is_evo:
                self.best_is_evo = is_value
                file_path = os.path.join(self.args.path_helper['ckpt_path'],
                                        "best_is_gen_.npy")
                np.save(file_path, genotype_G)
            if fid_value < self.best_fid_evo:
                self.best_fid_evo = fid_value
                file_path = os.path.join(self.args.path_helper['ckpt_path'],
                                        "best_fid_gen.npy")
                np.save(file_path, genotype_G)
        
        # 统计信息
        sum_is, sum_fid, sum_param = 0, 0, 0
        for i in range(keep_N):
            sum_is += is_values[i]
            sum_fid += fid_values[i]
            sum_param += params[i]
        avg_is = sum_is / keep_N
        avg_fid = sum_fid / keep_N
        avg_params = sum_param / keep_N
        logger.info(
            f'mean_IS: {round(avg_is, 3)}, mean_FID: {round(avg_fid, 3)}, max_IS: {round(np.max(is_values), 3)}, min_FID: {round(np.min(fid_values), 3)},mean_params: {round(avg_params, 3)}@ evolutionary_algebra: {num}.')
        
        # 计算综合得分：IS - FID (与选中代码相同的评估方式)
        composite_scores = 0.5 * np.array(is_values) / 5 - 0.5 * np.array(fid_values) / 90
        
        # 添加调试信息
        print(f"第 {num} 代: IS范围={np.min(is_values):.3f}-{np.max(is_values):.3f}, FID范围={np.min(fid_values):.3f}-{np.max(fid_values):.3f}")
        print(f"综合得分(IS-FID)范围: {np.min(composite_scores):.3f}-{np.max(composite_scores):.3f}")
        
        # 使用PSO算法优化选择
        optimized_genotypes, selected_indices = self.pso_search(genotypes, composite_scores, keep_N, selected_N)
        
        print(f"PSO选择结果: keep={len(optimized_genotypes)}个, selected={len(selected_indices)}个")
        print(f"最佳得分(IS-FID): {composite_scores[selected_indices[0]]:.3f}")
        
        # 记录选择的个体信息
        for i in range(len(selected_indices)):
            idx = selected_indices[i]
            logger.info(
                f'genotypes_{idx}, IS: {round(is_values[idx], 3)}, FID: {round(fid_values[idx], 3)}, composite_score(IS-FID): {round(composite_scores[idx], 3)}, param_size: {round(params[idx], 3)}.')

        self.genotypes = optimized_genotypes
        selected_genotypes = genotypes[selected_indices]
        
        return optimized_genotypes, selected_genotypes, np.mean(is_values[selected_indices]), np.mean(fid_values[selected_indices]), is_values[selected_indices], fid_values[selected_indices]

    def pso_search(self, genotypes, fitness_scores, keep_N, selected_N):
        """
        使用PSO算法进行个体选择和优化
        fitness_scores: 已计算好的适应度分数 (IS-FID综合得分)
        """
        num_particles = len(genotypes)
        dim = 21  # 基因型维度 (3x7)
        
        # PSO参数
        w = 0.9  # 惯性权重
        c1 = 2.0  # 认知系数
        c2 = 2.0  # 社会系数
        max_iter = 5  # PSO迭代次数
        
        # 将基因型转换为连续空间
        particles = np.array([genotype.flatten().astype(float) for genotype in genotypes])
        velocities = np.random.uniform(-1, 1, (num_particles, dim))
        
        # 初始化个体最优和全局最优
        personal_best_positions = particles.copy()
        personal_best_fitness = fitness_scores.copy()
        
        global_best_idx = np.argmax(fitness_scores)
        global_best_position = particles[global_best_idx].copy()
        global_best_fitness = fitness_scores[global_best_idx]
        
        print(f"PSO初始化: 全局最优适应度 = {global_best_fitness:.4f}")
        
        # PSO主循环
        for iteration in range(max_iter):
            improved_count = 0
            
            for i in range(num_particles):
                # 更新速度
                r1, r2 = np.random.random(dim), np.random.random(dim)
                velocities[i] = (w * velocities[i] + 
                            c1 * r1 * (personal_best_positions[i] - particles[i]) +
                            c2 * r2 * (global_best_position - particles[i]))
                
                # 限制速度
                velocities[i] = np.clip(velocities[i], -2, 2)
                
                # 更新位置
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 6)
                
                # 转换为离散基因型
                new_genotype = np.round(particles[i]).astype(int).reshape(3, 7)
                
                # 简单评估新基因型 (使用相似性估计，避免重复计算)
                new_fitness = self.estimate_fitness_simple(new_genotype, genotypes, fitness_scores)
                
                # 更新个体最优
                if new_fitness > personal_best_fitness[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_fitness[i] = new_fitness
                    improved_count += 1
                    
                    # 更新全局最优
                    if new_fitness > global_best_fitness:
                        global_best_position = particles[i].copy()
                        global_best_fitness = new_fitness
                        print(f"PSO迭代{iteration}: 发现更优解, 适应度 = {global_best_fitness:.4f}")
            
            # 动态调整惯性权重
            w = 0.9 - 0.4 * iteration / max_iter
            
            print(f"PSO迭代{iteration}: {improved_count}个粒子得到改进")
        
        # 选择最优个体 - 基于改进后的适应度排序
        sorted_indices = np.argsort(personal_best_fitness)[::-1]
        
        # 转换优化后的粒子回基因型
        optimized_genotypes = []
        for i in range(keep_N):
            idx = sorted_indices[i]
            optimized_genotype = np.round(personal_best_positions[idx]).astype(int).reshape(3, 7)
            optimized_genotypes.append(optimized_genotype)
        
        optimized_genotypes = np.array(optimized_genotypes)
        
        # 选择精英个体
        selected_N = min(selected_N, len(sorted_indices))
        selected_indices = sorted_indices[:selected_N]
        
        print(f"PSO完成: 最终最优适应度 = {global_best_fitness:.4f}")
        
        return optimized_genotypes, selected_indices

    def estimate_fitness_simple(self, target_genotype, reference_genotypes, reference_fitness):
        """
        基于相似度简单估计基因型适应度
        """
        # 计算与参考基因型的汉明距离
        similarities = []
        for ref_genotype in reference_genotypes:
            hamming_dist = np.sum(target_genotype.flatten() != ref_genotype.flatten())
            similarity = 1.0 / (1.0 + hamming_dist)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # 使用加权平均估计适应度
        if np.sum(similarities) > 0:
            weights = similarities / np.sum(similarities)
            estimated_fitness = np.sum(weights * reference_fitness)
        else:
            estimated_fitness = np.mean(reference_fitness)
        
        # 添加小幅随机扰动
        noise = np.random.normal(0, 0.01)
        estimated_fitness += noise
        
        return estimated_fitness
    def validate(self, genotype_G, fid_stat):
        """
    处理多种基因型输入格式并评估性能
    :param genotype_input: 可以是以下格式之一：
        - 3x7的numpy数组
        - 21维向量（将被reshape为3x7）
        - 列表形式（自动转换为numpy数组）
    :param fid_stat: FID统计量
    :return: (IS均值, IS标准差, FID值)
    """
        # 类型转换和形状验证
        try:
            if isinstance(genotype_G, np.ndarray):
            # 处理向量输入 (21,)
                if genotype_G.ndim == 1 and genotype_G.size == 21:
                    genotype = genotype_G.reshape(3,7)
            # 处理矩阵输入 (3,7)
                elif genotype_G.shape == (3,7):
                    genotype = genotype_G.copy()
                else:
                    raise ValueError("Invalid genotype array shape")
            elif isinstance(genotype_G, (list, tuple)):
            # 列表类型强制转换为numpy数组
                genotype = np.array(genotype_G).reshape(3,7)
            else:
                raise TypeError("Unsupported genotype input type")
            
            # 验证数值范围
            if not np.issubdtype(genotype.dtype, np.integer):
                genotype = genotype.astype(int)
            if (genotype < 0).any() or (genotype > 6).any():
                raise ValueError("Gene values must be integers in [0,6]")
            
        except Exception as e:
            raise ValueError(f"Invalid genotype format: {str(e)}") from e
        
        '''
        Calculate IS and FID
        '''
        # eval mode
        gen_net = self.gen_net.eval()
        eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
        fakes = []
        with torch.no_grad():
            for iter_idx in tqdm(range(eval_iter), desc='sample images'):
                z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
                gen_imgs = (gen_net(z, genotype_G) + 1) / 2
                fakes.append(gen_imgs.cpu())
        fakes = torch.cat(fakes, dim=0)

        # get inception score
        (mean, std), fid_score = get_inception_score_and_fid(fakes, fid_stat, verbose=True)
        return mean, std, fid_score

    def gen_offspring(self, alphas, offspring_ratio=1.0):
        """Generate offsprings.
        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """

        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            a, b = np.random.randint(
                0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
            while (a == b):
                a, b = np.random.randint(
                    0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
            alphas_c = self.uniform_crossover_mutation(alphas[a], alphas[b])
            # alphas_c = self.single_point_crossover_mutation(alphas[a], alphas[b])
            # alphas_c = self.two_point_crossover_mutation(alphas[a], alphas[b])
            encode_G = self.gan_alg.encode(alphas_c)
            if self.judege_model_size(alphas_c, limit=self.args.max_model_size) and (
                    encode_G not in self.historical_population):
                offsprings.append(alphas_c)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    def judge_repeat(self, alphas, new_alphas):
        """Judge if two individuals are the same.
        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param new_alphas: An individual
        :type new_alphas: nn.Tensor
        :return: True or false
        :rtype: nn.Tensor
        """
        diff = np.reshape(np.absolute(
            alphas - np.expand_dims(new_alphas, axis=0)), (alphas.shape[0], -1))
        diff = np.sum(diff, axis=1)
        return np.sum(diff == 0)

    def crossover(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        # alpha a
        new_alphas = alphas_a.copy()
        # alpha b
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        while (new_alphas[layer][index] == alphas_a[layer][index]):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            new_alphas[layer][index] = alphas_b[layer][index]
            if index >= 2 and index < 4 and new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
            elif index >= 4 and new_alphas[layer][4] == 0 and new_alphas[layer][5] == 0 and new_alphas[layer][6] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
        return new_alphas

    def uniform_crossover_mutation(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        new_alphas = alphas_a.copy()
        for i in range(3):
            for j in range(2):
                p = random.random()
                if p >= 0.5:
                    new_alphas[i][j] = alphas_b[i][j]
            for k in range(5):
                p = random.random()
                if p >= 0.5:
                    new_alphas[i][k + 2] = alphas_b[i][k + 2]
        # Mutation operation
        num_mute = random.randint(1, self.args.mute_max_num)
        for i in range(num_mute):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            if index < 2:
                new_alphas[layer][index] = random.choice(self.gan_alg.Up_G_fixed[2 * layer + index])
            else:
                new_alphas[layer][index] = random.choice(self.gan_alg.Normal_G_fixed[5 * layer + index - 2])
            # print(layer, index)
        return new_alphas

    def single_point_crossover_mutation(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        new_alphas = alphas_a.copy()
        a1 = random.randint(0, 2)  # cross point a1,a2
        a2 = random.randint(0, 7)
        for i in range(a1):
            for j in range(7):
                new_alphas[i][j] = alphas_b[i][j]
        for k in range(a2):
            new_alphas[a1][k] = alphas_b[a1][k]
        # Mutation operation
        num_mute = random.randint(1, self.args.mute_max_num)
        for i in range(num_mute):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            if index < 2:
                new_alphas[layer][index] = random.choice(self.gan_alg.Up_G_fixed[2 * layer + index])
            else:
                new_alphas[layer][index] = random.choice(self.gan_alg.Normal_G_fixed[5 * layer + index - 2])
            # print(layer, index)
        return new_alphas

    def two_point_crossover_mutation(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        alphas_a1 = alphas_a.ravel()
        alphas_b1 = alphas_b.ravel()
        new_alphas = alphas_a1.copy()
        a1 = random.randint(0, len(alphas_a1))  # cross point a1,a2
        a2 = random.randint(a1, len(alphas_a1))
        for i in range(a1, a2):
            new_alphas[i] = alphas_b1[i]
        new_alphas = new_alphas.reshape((3, 7))
        # Mutation operation
        num_mute = random.randint(1, self.args.mute_max_num)
        for i in range(num_mute):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            if index < 2:
                new_alphas[layer][index] = random.choice(self.gan_alg.Up_G_fixed[2 * layer + index])
            else:
                new_alphas[layer][index] = random.choice(self.gan_alg.Normal_G_fixed[5 * layer + index - 2])
            # print(layer, index)
        return new_alphas

    def mutation(self, alphas_a, ratio=0.5):
        """Mutation for An individual."""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        if index < 2:
            new_alphas[layer][index] = random.randint(0, 2)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 2)
        elif index >= 2 and index < 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][4] == 0 and new_alphas[layer][5] == 0 and new_alphas[layer][6] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        return new_alphas

    def select_best(self, epoch):
        values = []
        for genotype_G in self.genotypes:
            ssim_value, psnr_value = self.validate(genotype_G)
            # logger.info(f'ssim_value: {ssim_value}, psnr_value: {psnr_value}|| @ epoch {epoch}.')
            values.append(ssim_value)
        max_index = values.index(max(values))
        return self.genotypes[max_index]

    def get_fitness(self, genotypes, fid_stat):
        '''
        get fid
        '''
        fids = np.zeros(len(genotypes))
        for idx, genotype_G in enumerate(genotypes):
            # eval mode
            gen_net = self.gen_net.eval()
            eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
            fakes = []
            with torch.no_grad():
                for iter_idx in tqdm(range(eval_iter), desc='sample images'):
                    z = torch.cuda.FloatTensor(np.random.normal(
                        0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
                    gen_imgs = (gen_net(z, genotype_G) + 1) / 2
                    fakes.append(gen_imgs.cpu())
            fakes = torch.cat(fakes, dim=0)
            fid_score = get_fid(fakes, fid_stat)
            # fid = get_fid()
            fids[idx] = fid_score
        return fids

    def get_index(self, genotypes, fid_stat):
        # Obtain a ranking of performance from good to bad
        fids_G = self.get_fitness(genotypes, fid_stat)
        index = fids_G.argsort()
        return index

    def get_operation_score(self, genotypes, index):
        '''
        Obtain scores for each operation
        genotypes:Genes obtained from sampling
        index:Sorted Index
        '''
        scores_up = np.zeros((3, 2, 3), dtype=int)
        scores_normal = np.zeros((3, 5, 7), dtype=int)
        jj = 0
        num = len(genotypes) / 2
        for kk in index:
            for i in range(3):
                for j in range(2):
                    scores_up[i][j][genotypes[kk][i][j]] += 1
                for k in range(2):
                    scores_normal[i][k][genotypes[kk][i][k + 2]] += 1
                for k in range(2, 5):
                    scores_normal[i][k][genotypes[kk][i][k + 2]] += 1
            jj += 1
            if jj == num:
                break
        return scores_up, scores_normal

    def modify_fixed(self, Normal_G_fixed, Up_G_fixed, scores_up, scores_normal):
        '''
        Normal_G_fixed,Up_G_fixed: Fixed operation pool
        scores_up,scores_normal: Operation score
        Function: Modify fixed operation pool
        '''
        index_up = scores_up.argsort()
        index_normal = scores_normal.argsort()
        for i in range(3):
            for j in range(2):  # remove one
                Up_G_fixed[2 * i + j].remove(index_up[i][j][0])
        for i in range(3):
            for j in range(5):
                for kk in range(3):  # remove three
                    Normal_G_fixed[5 * i + j].remove(index_normal[i][j][kk])
        return None

    def greedy_modify(self, Normal_G_fixed, Up_G_fixed, rank, genotypes):
        '''
        Normal_G_fixed,Up_G_fixed: Fixed operation pool
        rank: Performance sorting
        '''
        # clear
        self.gan_alg.Normal_G_fixed = []
        for i in range(15):
            self.gan_alg.Normal_G_fixed.append([])
        self.gan_alg.Up_G_fixed = []
        for i in range(6):
            self.gan_alg.Up_G_fixed.append([])

        for r in rank:
            for i in range(3):
                for j in range(2):
                    if genotypes[r][i][j] not in self.gan_alg.Up_G_fixed[2 * i + j] and len(
                            self.gan_alg.Up_G_fixed[2 * i + j]) < 2:
                        self.gan_alg.Up_G_fixed[2 * i + j].append(genotypes[r][i][j])
            for i in range(3):
                for j in range(5):
                    if genotypes[r][i][j + 2] not in self.gan_alg.Normal_G_fixed[5 * i + j] and len(
                            self.gan_alg.Normal_G_fixed[5 * i + j]) < 4:
                        self.gan_alg.Normal_G_fixed[5 * i + j].append(genotypes[r][i][j + 2])
        return None

    def directly_modify_fixed(self, fid_stat):
        #  clear
        self.clear_bag()
        # sample
        genotypes_to_eva = np.stack([self.gan_alg.search() for i in range(42)], axis=0)
        rank = self.get_index(genotypes_to_eva, fid_stat)  # get rank
        scores_up, scores_normal = self.get_operation_score(genotypes_to_eva, rank)
        self.modify_fixed(self.gan_alg.Normal_G_fixed, self.gan_alg.Up_G_fixed, scores_up, scores_normal)
        self.clear_bag()
        self.genotype_fixG = self.gan_alg.search(remove=False)

    def greedy_modify_fixed(self, num, fid_stat):
        self.clear_bag()
        genotypes_to_eva = np.stack([self.gan_alg.search() for i in range(num)], axis=0)
        rank = self.get_index(genotypes_to_eva, fid_stat)

        self.greedy_modify(self.gan_alg.Normal_G_fixed, self.gan_alg.Up_G_fixed, rank, genotypes_to_eva)

        self.clear_bag()
        self.genotype_fixG = self.gan_alg.search(remove=False)

    def clear_bag(self):
        self.gan_alg.Normal_G = []
        for i in range(15):
            self.gan_alg.Normal_G.append([])
        self.gan_alg.Up_G = []
        for i in range(6):
            self.gan_alg.Up_G.append([])


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * \
                 (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr
